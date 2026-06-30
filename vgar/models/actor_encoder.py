
import torch, torch.nn as nn, torch.nn.functional as F
from vgar.data.features import N_JOINTS, BONES, FEAT_DIM, FEAT_SLICES, N_PLAYERS

# ── Reduced dimensions ─────────────────────────────────────
# Old: POSE_DIM=96, MOTION_DIM=64, CTX_DIM=32, ACTOR_DIM=192
# New: POSE_DIM=64, MOTION_DIM=48, CTX_DIM=24, ACTOR_DIM=128
ACTOR_DIM  = 128
POSE_DIM   = 64
MOTION_DIM = 48
CTX_DIM    = 24


def _build_adj(device):
    A = torch.zeros(N_JOINTS,N_JOINTS,device=device)
    for j1,j2 in BONES:
        A[j1,j2]=1.; A[j2,j1]=1.
    A = A+torch.eye(N_JOINTS,device=device)
    D = A.sum(-1,keepdim=True).clamp(min=1.).sqrt()
    return A/D/D.T


class SkeletonGCNLayer(nn.Module):
    def __init__(self,in_d,out_d):
        super().__init__()
        self.W  = nn.Linear(in_d,out_d,bias=False)
        self.bn = nn.LayerNorm(out_d)
        self._adj = None

    def _get_adj(self,dev):
        if self._adj is None or self._adj.device!=dev:
            self._adj = _build_adj(dev)
        return self._adj

    def forward(self,x):
        A = self._get_adj(x.device)
        return F.gelu(self.bn(self.W(torch.matmul(A,x))))


class PoseBranch(nn.Module):
    """
    2-layer GCN over joints + relative pose MLP.
    Slimmed: 3→32→48 (was 3→32→64→64)
    """
    def __init__(self):
        super().__init__()
        self.gcn1  = SkeletonGCNLayer(3, 32)
        self.gcn2  = SkeletonGCNLayer(32,48)
        self.pool  = nn.Linear(48,POSE_DIM)
        self.rel   = nn.Sequential(
            nn.Linear(N_JOINTS*2,48),nn.GELU(),nn.Linear(48,24))
        self.fuse  = nn.Linear(POSE_DIM+24,POSE_DIM)

    def forward(self,kp_global,kp_rel):
        BN,T,_ = kp_global.shape
        xg = kp_global.reshape(BN*T,N_JOINTS,3)
        h  = self.gcn2(self.gcn1(xg))           # (BN*T,J,48)
        g  = self.pool(h.mean(1)).reshape(BN,T,POSE_DIM)
        r  = self.rel(kp_rel)                    # (BN,T,24)
        return F.gelu(self.fuse(torch.cat([g,r],-1)))


class MotionBranch(nn.Module):
    """
    vel(34)+accel(34)+bones(32)+angles(19)+jump(3)=122 → 48
    Slimmed: 122→96→48 (was 122→128→64)
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(122,96),nn.GELU(),nn.LayerNorm(96),
            nn.Linear(96,MOTION_DIM))

    def forward(self,feat):
        s = FEAT_SLICES
        x = torch.cat([
            feat[:,:,s['velocity'][0]   :s['velocity'][1]],
            feat[:,:,s['acceleration'][0]:s['acceleration'][1]],
            feat[:,:,s['bones'][0]      :s['bones'][1]],
            feat[:,:,s['angles'][0]     :s['angles'][1]],
            feat[:,:,s['jump_proxy'][0] :s['jump_proxy'][1]],
        ],-1)
        return self.mlp(x)


class ContextBranch(nn.Module):
    """
    bbox(4)+bvel(4)+zone(6)+team(2)+net_s(1)+net_a(1)+form(6)+pres(1)=25 → 24
    Slimmed: 25→40→24 (was 25→48→32)
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(25,40),nn.GELU(),nn.LayerNorm(40),
            nn.Linear(40,CTX_DIM))

    def forward(self,feat):
        s = FEAT_SLICES
        x = torch.cat([
            feat[:,:,s['bbox'][0]      :s['bbox'][1]],
            feat[:,:,s['bbox_vel'][0]  :s['bbox_vel'][1]],
            feat[:,:,s['zone'][0]      :s['zone'][1]],
            feat[:,:,s['team'][0]      :s['team'][1]],
            feat[:,:,s['net_signed'][0]:s['net_signed'][1]],
            feat[:,:,s['net_abs'][0]   :s['net_abs'][1]],
            feat[:,:,s['formation'][0] :s['formation'][1]],
            feat[:,:,s['present'][0]   :s['present'][1]],
        ],-1)
        return self.mlp(x)


class ActorEncoder(nn.Module):
    """
    Encodes each player×frame into ACTOR_DIM=128.
    Parameters: ~120K (was ~350K)
    """
    def __init__(self,config):
        super().__init__()
        self.pose   = PoseBranch()
        self.motion = MotionBranch()
        self.ctx    = ContextBranch()
        self.fuse   = nn.Sequential(
            nn.Linear(POSE_DIM+MOTION_DIM+CTX_DIM, ACTOR_DIM),
            nn.LayerNorm(ACTOR_DIM), nn.GELU())
        self.drop   = nn.Dropout(config.get("dropout_rate",0.15))

    def forward(self,feat,kp_raw,present):
        B,T,N,_ = feat.shape
        feat_bn = feat.permute(0,2,1,3).reshape(B*N,T,FEAT_DIM)
        pres_bn = present.permute(0,2,1).reshape(B*N,T)

        s       = FEAT_SLICES
        kg      = feat_bn[:,:,s['kp_global'][0]:s['kp_global'][1]]
        kr      = feat_bn[:,:,s['kp_relative'][0]:s['kp_relative'][1]]
        # reshape for GCN: (BN, T, J*3)
        kg_flat = kg  # already (BN,T,51)

        p = self.pose(kg_flat, kr)    # (BN,T,POSE_DIM)
        m = self.motion(feat_bn)      # (BN,T,MOTION_DIM)
        c = self.ctx(feat_bn)         # (BN,T,CTX_DIM)

        actor = self.fuse(torch.cat([p,m,c],-1))  # (BN,T,128)
        actor = self.drop(actor) * pres_bn.unsqueeze(-1)
        return actor.reshape(B,N,T,ACTOR_DIM).permute(0,2,1,3)
