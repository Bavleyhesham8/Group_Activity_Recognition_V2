
import torch, torch.nn.functional as F
from vgar.data.features import N_CLASSES


def focal_loss(logits,labels,gamma=2.0,alpha=None,smoothing=0.05):
    C = logits.size(1)
    with torch.no_grad():
        sm = torch.full_like(logits,smoothing/(C-1))
        sm.scatter_(1,labels.unsqueeze(1),1.-smoothing)
    lp  = F.log_softmax(logits,-1)
    p   = lp.exp()
    lpt = (sm*lp).sum(-1)
    pt  = (sm*p).sum(-1)
    fw  = (1-pt).pow(gamma)
    if alpha is not None:
        fw = fw*alpha[labels]
    return (-fw*lpt).mean()


def compute_loss(outputs,batch,class_weights,config,epoch):
    dev    = batch["label"].device
    alpha  = class_weights.to(dev)
    labels = batch["label"]

    gamma    = config.get("focal_gamma",2.0)
    smoothing= config.get("label_smoothing",0.05)
    w_side   = config.get("w_side",0.5)
    w_action = config.get("w_action",0.7)

    L_joint  = focal_loss(outputs["logits"],labels,gamma,alpha,smoothing)
    L_side   = F.cross_entropy(outputs["side"],  batch["side"])
    L_action = F.cross_entropy(outputs["action"],batch["action"])

    total = L_joint + w_side*L_side + w_action*L_action

    return total,{
        "L_total":  total.item(),
        "L_joint":  L_joint.item(),
        "L_side":   L_side.item(),
        "L_action": L_action.item(),
    }
