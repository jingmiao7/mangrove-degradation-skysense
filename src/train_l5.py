"""Controlled backbone/pretraining, LoRA, and augmentation ablations."""
import argparse, csv, gc, json, math, os, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MANGROVE_DATA", PROJECT / "data")).expanduser().resolve()
SPLIT_DIR = Path(os.environ.get("MANGROVE_SPLITS", PROJECT / "configs/splits")).expanduser().resolve()
OUTPUT = Path(os.environ.get("OUTPUT_DIR", PROJECT / "outputs")).expanduser().resolve()
PRETRAIN = Path(os.environ.get("SKYSENSE_PRETRAIN", PROJECT / "weights/skysense_model_backbone_s2.pth")).expanduser().resolve()
YEARS = (2017, 2019, 2021, 2023)
MEAN = torch.tensor([172.6143,304.1498,203.0544,486.6588,1492.7885,
                     1896.2063,1988.8759,2119.2815,653.4074,281.6476]).view(1,10,1,1)
STD = torch.tensor([79.7107,120.3797,132.46,191.0323,730.7147,
                    945.5046,1017.748,1060.7884,409.2817,223.9724]).view(1,10,1,1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune.backbone_loader import build_skysense_backbone
from standalone_vit import VisionTransformer, MultiheadAttention
from train_full_lora import (ConvGnRelu, FullModel, LoRALinear, PixelWeightedL1Loss,
                             ResidualBlock, UNetGatedDecoder)
from finetune.train_pure_unet_v3 import PureUNetV2

def memory_efficient_attention_forward(self, x):
    """Equivalent scaled dot-product attention using the B200 fused kernel."""
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    x = F.scaled_dot_product_attention(
        q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
    )
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(x))

MultiheadAttention.forward = memory_efficient_attention_forward

FULL_AUG = {"gaussian": True, "edge": True, "flip": True, "band": True}
NO_AUG = {key: False for key in FULL_AUG}
EXPERIMENTS = {
    "B0_pure_unet": {"group":"backbone", "kind":"pure", "aug":FULL_AUG},
    "B1_random_vit_fullft": {"group":"backbone", "kind":"vit_full", "pretrained":False, "aug":FULL_AUG},
    "B2_pretrained_vit_fullft": {"group":"backbone", "kind":"vit_full", "pretrained":True, "aug":FULL_AUG},
    "L0_frozen": {"group":"lora", "kind":"lora", "target":"none", "start":24, "aug":FULL_AUG},
    "L1_attention_all24": {"group":"lora", "kind":"lora", "target":"attention", "start":0, "aug":FULL_AUG},
    "L2_ffn_all24": {"group":"lora", "kind":"lora", "target":"ffn", "start":0, "aug":FULL_AUG},
    "L3_both_last6": {"group":"lora", "kind":"lora", "target":"both", "start":18, "aug":FULL_AUG},
    "L4_both_last12": {"group":"lora", "kind":"lora", "target":"both", "start":12, "aug":FULL_AUG},
    "L5_both_all24_full": {"group":"lora", "kind":"lora", "target":"both", "start":0,
                            "decoder":"gated", "aug":FULL_AUG},
    "H0_simple_single_scale": {"group":"decoder", "kind":"lora", "target":"both", "start":0,
                               "decoder":"simple", "aug":FULL_AUG},
    "H1_multiscale_unet_no_gating": {"group":"decoder", "kind":"lora", "target":"both", "start":0,
                                      "decoder":"no_gating", "aug":FULL_AUG},
    "D0_no_augmentation": {"group":"augmentation", "kind":"lora", "target":"both", "start":0, "aug":NO_AUG},
    "D2_without_gaussian": {"group":"augmentation", "kind":"lora", "target":"both", "start":0,
                            "aug":{"gaussian":False,"edge":True,"flip":True,"band":True}},
    "D3_without_edge": {"group":"augmentation", "kind":"lora", "target":"both", "start":0,
                        "aug":{"gaussian":True,"edge":False,"flip":True,"band":True}},
    "D4_without_flip": {"group":"augmentation", "kind":"lora", "target":"both", "start":0,
                        "aug":{"gaussian":True,"edge":True,"flip":False,"band":True}},
    "D5_without_band_dropout": {"group":"augmentation", "kind":"lora", "target":"both", "start":0,
                                "aug":{"gaussian":True,"edge":True,"flip":True,"band":False}},
}
FORMAL_EXPERIMENTS = [name for name, config in EXPERIMENTS.items() if config["group"] != "augmentation"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

def year_of(name):
    match = re.search(r"(2017|2019|2021|2023)", name)
    return int(match.group(1)) if match else -1

class ControlledDataset(Dataset):
    def __init__(self, split, aug, seed):
        path = SPLIT_DIR / f"{split}_256_all.json"
        self.items = json.loads(path.read_text()); self.aug = dict(aug)
        self.seed = int(seed); self.epoch = 0; self.training = split == "train"
    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        name = self.items[index]["tile_name"]
        image = torch.from_numpy(np.load(DATA/f"tiles_256/images/{name}.npy")).float()
        label = torch.from_numpy(np.load(DATA/f"tiles_256/labels/{name}.npy")).float()
        mask_path = DATA/f"tiles_256/labels/{name}_mask.npy"
        if mask_path.exists(): label[torch.from_numpy(np.load(mask_path)) == 0] = -1
        if self.training: image, label = self._augment(image, label, index)
        return image, label, name, year_of(name)
    def _augment(self, image, label, index):
        # Always draw every random variable so retained transforms match across ablations.
        g = torch.Generator().manual_seed(self.seed + self.epoch*100_003 + index)
        do_noise = torch.rand((),generator=g).item()<.5; noise_std=torch.rand((),generator=g).item()*50
        noise = torch.randn(image.shape,generator=g,dtype=image.dtype)
        do_edge=torch.rand((),generator=g).item()<.5; borders=torch.randint(0,65,(4,),generator=g).tolist()
        flip_h=torch.rand((),generator=g).item()<.5; flip_v=torch.rand((),generator=g).item()<.5
        do_band=torch.rand((),generator=g).item()<.2; n_drop=int(torch.randint(1,3,(),generator=g).item())
        bands=torch.randperm(10,generator=g)[:n_drop]
        if self.aug["gaussian"] and do_noise: image = image + noise*noise_std
        if self.aug["edge"] and do_edge:
            top,bottom,left,right=borders
            if top: image[:,:top,:]=0; label[:top,:]=-1
            if bottom: image[:,-bottom:,:]=0; label[-bottom:,:]=-1
            if left: image[:,:,:left]=0; label[:,:left]=-1
            if right: image[:,:,-right:]=0; label[:,-right:]=-1
        if self.aug["flip"] and flip_h: image=image.flip(1); label=label.flip(0)
        if self.aug["flip"] and flip_v: image=image.flip(2); label=label.flip(1)
        if self.aug["band"] and do_band: image[bands]=0
        return image,label

class NormalizedPureUNet(nn.Module):
    def __init__(self):
        super().__init__(); self.network=PureUNetV2(10)
        self.register_buffer("mean",MEAN); self.register_buffer("std",STD)
    def forward(self,image): return self.network((image-self.mean)/self.std)


class SimpleSingleScaleHead(nn.Module):
    """Minimal head using only the final ViT feature map."""
    def __init__(self, feat_ch=1024):
        super().__init__()
        self.proj = ConvGnRelu(feat_ch, 128, 1, 0)
        self.refine = ResidualBlock(128, 64)
        self.dropout = nn.Dropout2d(0.1)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, feats, img_norm):
        x = self.refine(self.proj(feats[-1]))
        x = F.interpolate(x, size=img_norm.shape[-2:], mode="bilinear", align_corners=False)
        return self.final(self.dropout(x))


class AverageFusion(nn.Module):
    """Parameter-free equal fusion followed by the same residual refinement."""
    def __init__(self, channels):
        super().__init__()
        self.refine = ResidualBlock(channels, channels)

    def forward(self, x_prev, x_skip):
        return self.refine(0.5 * (x_prev + x_skip))


class UNetNoGatingDecoder(nn.Module):
    """Same multiscale decoder path as the main head, without learned gates."""
    def __init__(self, feat_ch=1024, img_ch=10):
        super().__init__()
        self.compress4 = ConvGnRelu(feat_ch, 128, 1, 0)
        self.compress3 = ConvGnRelu(feat_ch, 128, 1, 0)
        self.compress2 = ConvGnRelu(feat_ch, 128, 1, 0)
        self.compress1 = ConvGnRelu(feat_ch, 128, 1, 0)
        self.fuse43 = AverageFusion(128)
        self.fuse32 = AverageFusion(128)
        self.fuse21 = AverageFusion(128)
        self.bottleneck = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256))
        self.up1_pre = ResidualBlock(256, 192)
        self.skip1_proj = ConvGnRelu(128, 64, 1, 0)
        self.up1_fuse = nn.Sequential(ResidualBlock(192 + 64, 128), ResidualBlock(128, 128))
        self.up2_pre = ResidualBlock(128, 96)
        self.img_proj = ConvGnRelu(img_ch, 32, 1, 0)
        self.up2_fuse = nn.Sequential(ResidualBlock(96 + 32, 64), ResidualBlock(64, 32))
        self.dropout = nn.Dropout2d(0.1)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, feats, img_norm):
        c1, c2, c3, c4 = feats
        f1, f2 = self.compress1(c1), self.compress2(c2)
        f3, f4 = self.compress3(c3), self.compress4(c4)
        x = self.fuse43(f4, f3)
        x = self.fuse32(x, f2)
        x = self.fuse21(x, f1)
        x = self.bottleneck(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1_pre(x)
        f1_up = self.skip1_proj(F.interpolate(f1, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.up1_fuse(torch.cat([x, f1_up], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2_pre(x)
        img_feat = self.img_proj(img_norm)
        x = self.up2_fuse(torch.cat([x, img_feat], dim=1))
        return self.final(self.dropout(x))


def build_decoder(config):
    decoder = config.get("decoder", "gated")
    if decoder == "gated": return UNetGatedDecoder()
    if decoder == "simple": return SimpleSingleScaleHead()
    if decoder == "no_gating": return UNetNoGatingDecoder()
    raise ValueError(f"Unknown decoder: {decoder}")

def fresh_vit():
    return VisionTransformer(img_size=(256,256),patch_size=4,in_channels=10,embed_dims=1024,
        num_layers=24,num_heads=16,mlp_ratio=4,out_indices=(5,11,17,23),qkv_bias=True,
        drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,with_cls_token=False,with_cp=True,norm_eval=True)

def add_lora(backbone,target,start,rank=8,alpha=16):
    parameters=[]
    if target=="none": return parameters
    blocks=backbone.layers if hasattr(backbone,"layers") else backbone.blocks
    for index,block in enumerate(blocks):
        if index<start: continue
        if target in ("attention","both"):
            for attribute in ("qkv","proj"):
                replacement=LoRALinear(getattr(block.attn,attribute),rank,alpha)
                setattr(block.attn,attribute,replacement); parameters += [replacement.lora_A,replacement.lora_B]
        if target in ("ffn","both"):
            if hasattr(block,"ffn") and hasattr(block.ffn,"layers"):
                first,second=block.ffn.layers[0][0],block.ffn.layers[1]
                r1,r2=LoRALinear(first,rank,alpha),LoRALinear(second,rank,alpha)
                block.ffn.layers[0][0],block.ffn.layers[1]=r1,r2
            else:
                r1,r2=LoRALinear(block.mlp.fc1,rank,alpha),LoRALinear(block.mlp.fc2,rank,alpha)
                block.mlp.fc1,block.mlp.fc2=r1,r2
            parameters += [r1.lora_A,r1.lora_B,r2.lora_A,r2.lora_B]
    return parameters

def set_activation_checkpointing(backbone, enabled):
    """Toggle recomputation without changing the model or learned parameters."""
    blocks = backbone.layers if hasattr(backbone, "layers") else backbone.blocks
    changed = 0
    for block in blocks:
        if hasattr(block, "with_cp"):
            block.with_cp = bool(enabled)
            changed += 1
    if changed != len(blocks):
        raise RuntimeError(f"Checkpoint toggle reached {changed}/{len(blocks)} blocks")


def build_model(config, activation_checkpointing=False):
    if config["kind"]=="pure":
        model=NormalizedPureUNet(); groups=[{"params":model.parameters(),"lr":1e-3,"weight_decay":1e-4}]
    elif config["kind"]=="vit_full":
        backbone=build_skysense_backbone(str(PRETRAIN)) if config["pretrained"] else fresh_vit()
        set_activation_checkpointing(backbone, activation_checkpointing)
        for parameter in backbone.parameters(): parameter.requires_grad=True
        head=build_decoder(config); model=FullModel(backbone,head,MEAN,STD)
        groups=[{"params":backbone.parameters(),"lr":1e-4,"weight_decay":.01},
                {"params":head.parameters(),"lr":1e-3,"weight_decay":1e-4}]
    else:
        backbone=build_skysense_backbone(str(PRETRAIN)); set_activation_checkpointing(backbone, activation_checkpointing)
        adapted=add_lora(backbone,config["target"],config["start"])
        head=build_decoder(config); model=FullModel(backbone,head,MEAN,STD); groups=[]
        if adapted: groups.append({"params":adapted,"lr":5e-4,"weight_decay":.01})
        groups.append({"params":head.parameters(),"lr":1e-3,"weight_decay":1e-4})
    trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
    return model,groups,trainable

class Moments:
    def __init__(self):
        self.n=0; self.sy=0.; self.sp=0.; self.sabs=0.; self.ssq=0.; self.sy2=0.
    def add(self,prediction,target):
        p,y=prediction.double(),target.double(); error=p-y; self.n+=y.numel()
        self.sy+=y.sum().item(); self.sp+=p.sum().item(); self.sabs+=error.abs().sum().item()
        self.ssq+=error.square().sum().item(); self.sy2+=y.square().sum().item()
    def result(self):
        if not self.n: return {"n":0,"r2":None,"mae":None,"rmse":None,"bias":None}
        sst=self.sy2-self.sy**2/self.n
        return {"n":self.n,"r2":1-self.ssq/max(sst,1e-12),"mae":self.sabs/self.n,
                "rmse":math.sqrt(self.ssq/self.n),"bias":(self.sp-self.sy)/self.n}

@torch.no_grad()
def evaluate(model,loader,amp):
    model.eval(); moments={"all":Moments(),**{str(year):Moments() for year in YEARS}}
    for image,label,_,year in loader:
        image,label=image.cuda(),label.cuda()
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp): prediction=model(image).squeeze(1)
        valid=label>=0
        if valid.any():
            p,y=prediction[valid].float().cpu(),label[valid].cpu(); moments["all"].add(p,y)
            moments[str(int(year[0]))].add(p,y)
    return {key:value.result() for key,value in moments.items()}

def save_best(path,model,experiment,epoch,val):
    config=EXPERIMENTS[experiment]
    if config["kind"] in ("pure","vit_full"):
        state=model.state_dict()
    else:
        state={name:value for name,value in model.state_dict().items()
               if name.startswith("head.") or "lora_A" in name or "lora_B" in name}
    torch.save({"experiment":experiment,"epoch":epoch,"val":val,"model_state":state},path); del state; gc.collect()

def run_one(experiment,seed,epochs,batch_size,amp,force,activation_checkpointing):
    config=EXPERIMENTS[experiment]; run_dir=OUTPUT/experiment/f"seed_{seed}"; run_dir.mkdir(parents=True,exist_ok=True)
    result_path=run_dir/"result.json"
    if result_path.exists() and not force:
        print(f"SKIP {experiment} seed={seed}",flush=True); return
    seed_all(seed); model,groups,trainable=build_model(config,activation_checkpointing)
    total=sum(p.numel() for p in model.parameters()); n_train=sum(p.numel() for p in trainable)
    model=model.cuda(); optimizer=AdamW(groups); scheduler=CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
    criterion=PixelWeightedL1Loss(); train_ds=ControlledDataset("train",config["aug"],seed)
    val_ds=ControlledDataset("val",NO_AUG,seed); test_ds=ControlledDataset("test",NO_AUG,seed)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,generator=generator,num_workers=0,pin_memory=False,drop_last=True)
    val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0); test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=0)
    metadata={"experiment":experiment,"config":config,"seed":seed,"epochs":epochs,"batch_size":batch_size,
              "amp_bfloat16":amp,"attention_backend":"torch_sdpa",
              "activation_checkpointing":activation_checkpointing,
              "data_loader_workers":0,"total_parameters":total,"trainable_parameters":n_train,
              "train_samples":len(train_ds),"val_samples":len(val_ds),"test_samples":len(test_ds)}
    (run_dir/"config.json").write_text(json.dumps(metadata,indent=2)); history=run_dir/"history.jsonl"; history.write_text("")
    best_r2=-float("inf"); best_epoch=-1; checkpoint=run_dir/"best.pth"
    print(f"START {experiment} seed={seed} trainable={n_train/1e6:.2f}M",flush=True)
    for epoch in range(1,epochs+1):
        started=time.time(); model.train(); train_ds.set_epoch(epoch); loss_sum=0.; batches=0
        for image,label,_,_ in train_loader:
            image,label=image.cuda(),label.cuda(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp): loss=criterion(model(image),label)
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.); optimizer.step(); loss_sum+=loss.item(); batches+=1
        scheduler.step(); val=evaluate(model,val_loader,amp)
        record={"epoch":epoch,"train_loss":loss_sum/max(batches,1),"elapsed_seconds":time.time()-started,"val":val}
        with history.open("a") as destination: destination.write(json.dumps(record)+"\n")
        r2=val["all"]["r2"]
        print(f"{experiment} ep={epoch:02d} loss={record['train_loss']:.5f} val_R2={r2:.5f} RMSE={val['all']['rmse']:.5f} sec={record['elapsed_seconds']:.1f}",flush=True)
        if r2>best_r2: best_r2,best_epoch=r2,epoch; save_best(checkpoint,model,experiment,epoch,val)
    payload=torch.load(checkpoint,map_location="cuda"); model.load_state_dict(payload["model_state"],strict=False)
    test=evaluate(model,test_loader,amp); result={**metadata,"best_epoch":best_epoch,"best_validation":payload["val"],"test":test}
    result_path.write_text(json.dumps(result,indent=2)); print(f"DONE {experiment} best_ep={best_epoch} test_R2={test['all']['r2']:.5f}",flush=True)
    del model,optimizer,scheduler,trainable,groups; torch.cuda.empty_cache(); gc.collect()

def summarize():
    rows=[]
    for path in sorted(OUTPUT.glob("*/seed_42/result.json")):
        result=json.loads(path.read_text())
        if result["config"]["group"] == "augmentation": continue
        for split,root in (("validation",result["best_validation"]),("test",result["test"])):
            for scope,metric in root.items():
                rows.append({"experiment":result["experiment"],"group":result["config"]["group"],"seed":result["seed"],
                    "split":split,"scope":scope,"best_epoch":result["best_epoch"],"total_parameters":result["total_parameters"],
                    "trainable_parameters":result["trainable_parameters"],**metric})
    if rows:
        with (OUTPUT/"ablation_metrics_long.csv").open("w",newline="") as destination:
            writer=csv.DictWriter(destination,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--experiments",nargs="+",default=["L5_both_all24_full"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[42]); parser.add_argument("--epochs",type=int,default=60)
    parser.add_argument("--batch-size",type=int,default=4); parser.add_argument("--no-amp",action="store_true"); parser.add_argument("--force",action="store_true")
    parser.add_argument("--activation-checkpointing",action="store_true",
                        help="Trade speed for lower GPU memory; disabled by default on the 180 GiB B200")
    args=parser.parse_args(); unknown=sorted(set(args.experiments)-set(EXPERIMENTS))
    if unknown: raise ValueError(f"Unknown experiments: {unknown}")
    
    for required in (DATA/"tiles_256/images", DATA/"tiles_256/labels", SPLIT_DIR/"train_256_all.json", SPLIT_DIR/"val_256_all.json", SPLIT_DIR/"test_256_all.json", PRETRAIN):
        if not required.exists(): raise FileNotFoundError(f"Required input is missing: {required}")
    OUTPUT.mkdir(parents=True,exist_ok=True); (OUTPUT/"experiment_matrix.json").write_text(json.dumps({name:EXPERIMENTS[name] for name in FORMAL_EXPERIMENTS},indent=2))
    for seed in args.seeds:
        for experiment in args.experiments:
            run_one(experiment,seed,args.epochs,args.batch_size,not args.no_amp,args.force,args.activation_checkpointing)
            summarize()

if __name__=="__main__": main()
