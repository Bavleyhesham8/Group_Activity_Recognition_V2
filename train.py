
import torch, torch.nn as nn, time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR,LinearLR,SequentialLR
from pathlib import Path

from vgar.data.dataset      import build_loaders
from vgar.models.pipeline   import VATFormer
from vgar.training.losses   import compute_loss
from vgar.configs.default   import DEFAULT_CONFIG
from vgar.data.features     import IDX_TO_CLASS, N_CLASSES


def freeze(m):
    for p in m.parameters(): p.requires_grad_(False)

def unfreeze(m):
    for p in m.parameters(): p.requires_grad_(True)


def make_opt_sch(params,config,n_epochs,steps_per_epoch):
    opt = AdamW(params,lr=config["lr"],weight_decay=config["weight_decay"])
    ws  = config["warmup_epochs"]*steps_per_epoch
    ts  = n_epochs*steps_per_epoch
    w   = LinearLR(opt,0.1,1.0,total_iters=ws)
    c   = CosineAnnealingLR(opt,T_max=max(ts-ws,1),eta_min=config["lr_min"])
    sch = SequentialLR(opt,[w,c],milestones=[ws])
    return opt,sch


def run_epoch(model,loader,cw,config,epoch,opt=None,sch=None,device="cpu"):
    training = opt is not None
    model.train(training)
    tot_loss=0.; correct=0; total=0; ld_sum={}

    for batch in loader:
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v
               for k,v in batch.items()}
        with torch.set_grad_enabled(training):
            out   = model(batch)
            loss,ld = compute_loss(out,batch,cw,config,epoch)

        if training:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["grad_clip"])
            opt.step()
            if sch: sch.step()

        bs = batch["label"].shape[0]
        tot_loss += loss.item()*bs
        correct  += (out["logits"].argmax(-1)==batch["label"]).sum().item()
        total    += bs
        for k,v in ld.items():
            ld_sum[k] = ld_sum.get(k,0.)+v*bs

    avg = {k:v/max(total,1) for k,v in ld_sum.items()}
    return correct/max(total,1), avg


@torch.no_grad()
def evaluate(model,loader,cw,config,epoch,device):
    model.eval()
    all_p,all_l=[],[]
    for batch in loader:
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v
               for k,v in batch.items()}
        out = model(batch)
        all_p.append(out["logits"].argmax(-1).cpu())
        all_l.append(batch["label"].cpu())
    if not all_p:
        return {"accuracy":0.,"f1":0.,"per_class_f1":[0.]*N_CLASSES}
    P=torch.cat(all_p); L=torch.cat(all_l)
    acc=(P==L).float().mean().item()
    f1s=[]
    for c in range(N_CLASSES):
        tp=((P==c)&(L==c)).sum().item()
        fp=((P==c)&(L!=c)).sum().item()
        fn=((P!=c)&(L==c)).sum().item()
        pr=tp/(tp+fp+1e-8); rc=tp/(tp+fn+1e-8)
        f1s.append(2*pr*rc/(pr+rc+1e-8))
    return {"accuracy":acc,"f1":sum(f1s)/len(f1s),"per_class_f1":f1s}


def train_phase(model,trl,vll,tel,cw,config,name,components,epochs,start_ep,device):
    print(f"\n{'='*55}")
    print(f"Phase: {name}  |  Epochs: {epochs}")
    print(f"Components: {components}")
    print(f"{'='*55}")

    freeze(model)
    comp_map={
        "actor_enc":  model.actor_enc,
        "player_int": model.player_int,
        "classifier": model.classifier,
    }
    params=[]
    for c in components:
        if c in comp_map:
            unfreeze(comp_map[c])
            params+=list(comp_map[c].parameters())

    if not params:
        print("  No trainable params."); return {}

    lr = config["lr"] if "actor_enc" in components else config["lr"]*0.3
    opt,sch = make_opt_sch(
        [p for p in params if p.requires_grad],
        {**config,"lr":lr}, epochs, len(trl))

    best_f1=0.; best_state=None; history=[]

    for ep in range(epochs):
        g = start_ep+ep
        t0= time.time()
        tr_acc,tr_ld = run_epoch(model,trl,cw,config,g,opt,sch,device)
        val_m        = evaluate(model,vll,cw,config,g,device)

        print(f"  Ep{g:3d} | "
              f"TrAcc:{tr_acc:.3f} | "
              f"ValAcc:{val_m['accuracy']:.3f} | "
              f"ValF1:{val_m['f1']:.3f} | "
              f"Ltot:{tr_ld.get('L_total',0):.4f} | "
              f"Ljnt:{tr_ld.get('L_joint',0):.4f} | "
              f"Lsd:{tr_ld.get('L_side',0):.4f} | "
              f"Lact:{tr_ld.get('L_action',0):.4f} | "
              f"{time.time()-t0:.1f}s")

        history.append({"epoch":g,"tr_acc":tr_acc,**val_m,**tr_ld})
        if val_m["f1"]>best_f1:
            best_f1=val_m["f1"]
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state); model.to(device)

    te_m = evaluate(model,tel,cw,config,start_ep+epochs,device)
    print(f"\n  Phase '{name}' — Test Acc:{te_m['accuracy']:.4f} F1:{te_m['f1']:.4f}")
    return {"best_val_f1":best_f1,"test":te_m,"history":history}


def train_full_model(config=None):
    if config is None: config=DEFAULT_CONFIG.copy()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nLoading data...")
    trl,vll,tel,cw = build_loaders(config)
    cw = cw.to(device)

    print("\nBuilding model...")
    model = VATFormer(config).to(device)
    total_p  = sum(p.numel() for p in model.parameters())
    train_p  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_p:,}")
    print(f"  Trainable parameters: {train_p:,}")

    # ── Parameter breakdown ────────────────────────────────
    for name,mod in [("ActorEncoder",model.actor_enc),
                     ("PlayerInteraction",model.player_int),
                     ("TemporalClassifier",model.classifier)]:
        n = sum(p.numel() for p in mod.parameters())
        print(f"    {name:22s}: {n:>9,}")

    phases=[
        {"name":"1 — Actor Warm-up",
         "components":["actor_enc","classifier"],
         "epochs":config["phase1_epochs"]},
        {"name":"2 — Actor + Interaction",
         "components":["actor_enc","player_int","classifier"],
         "epochs":config["phase2_epochs"]},
        {"name":"3 — Full End-to-End",
         "components":["actor_enc","player_int","classifier"],
         "epochs":config["phase3_epochs"]},
        {"name":"4 — Fine-Tune",
         "components":["actor_enc","player_int","classifier"],
         "epochs":config["phase4_epochs"]},
    ]

    all_history=[]; ep=0
    for ph in phases:
        res = train_phase(model,trl,vll,tel,cw,config,
                          ph["name"],ph["components"],
                          ph["epochs"],ep,device)
        all_history.extend(res.get("history",[]))
        ep += ph["epochs"]

    print("\n"+"="*55)
    print("FINAL TEST EVALUATION")
    print("="*55)
    te = evaluate(model,tel,cw,config,ep,device)
    print(f"  Test Accuracy : {te['accuracy']:.4f}")
    print(f"  Test Macro F1 : {te['f1']:.4f}")
    print("\n  Per-class F1:")
    for c,f in enumerate(te["per_class_f1"]):
        print(f"    {IDX_TO_CLASS[c]:15s}: {f:.4f}")

    sp = Path("/kaggle/working/vgar/models/final_model.pt")
    sp.parent.mkdir(exist_ok=True)
    torch.save({"state":model.state_dict(),"config":config,"test":te},sp)
    print(f"\n  Saved → {sp}")
    return model,all_history,te


if __name__=="__main__":
    train_full_model()
