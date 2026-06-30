
import torch, torch.nn as nn, torch.nn.functional as F
from vgar.data.features import N_CLASSES, N_SIDES, N_ACTIONS, N_KEYFRAMES
from vgar.models.actor_encoder import ACTOR_DIM
from vgar.models.relation_transformer import N_HEADS, FF_DIM

# ── Reduced temporal dim ───────────────────────────────────
# Old: TEMPORAL_DIM=256
# New: TEMPORAL_DIM=128  (saves ~30% of classifier params)
TEMPORAL_DIM = 128


class TemporalTransformer(nn.Module):
    """
    Bidirectional transformer over T frame tokens.

    KEY FRAME FIX:
      - The Volleyball dataset label is defined at the TARGET/MIDDLE frame.
      - num_frames ≈ 20, target = middle = num_frames//2.
      - We sample 10 frames centered on target (guaranteed by sample_keyframes).
      - key_frame_pos tells us WHICH of the 10 sampled frames IS the target.
      - We extract that frame's hidden state and fuse with CLS via a learned gate.

    This is different from old code which took ONLY the last causal frame state.
    """
    def __init__(self,in_dim,n_layers,n_heads,ff_dim,dropout):
        super().__init__()
        self.in_proj   = nn.Linear(in_dim,TEMPORAL_DIM)
        self.cls_token = nn.Parameter(torch.randn(1,1,TEMPORAL_DIM)*0.02)
        # +1 for CLS position
        self.pos_embed = nn.Embedding(N_KEYFRAMES+1,TEMPORAL_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=TEMPORAL_DIM, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer,n_layers)
        self.norm = nn.LayerNorm(TEMPORAL_DIM)
        # Gated fusion: CLS (global) + key_frame (local action peak)
        self.kf_gate = nn.Sequential(
            nn.Linear(TEMPORAL_DIM*2,TEMPORAL_DIM),nn.Sigmoid())

    def forward(self,x,key_frame_pos=None):
        """
        x:             (B,T,in_dim)
        key_frame_pos: (B,)  int in [0..T-1], index of target frame
                             in the 10 sampled frames
        """
        B,T,_ = x.shape
        dev   = x.device

        h   = self.in_proj(x)
        cls = self.cls_token.expand(B,-1,-1)
        h   = torch.cat([cls,h],dim=1)         # (B,T+1,D)
        h   = h+self.pos_embed(torch.arange(T+1,device=dev))
        h   = self.transformer(h)
        h   = self.norm(h)

        cls_out = h[:,0]                        # (B,D) global context

        if key_frame_pos is not None:
            # +1 offset because position 0 is CLS token
            kf_idx = (key_frame_pos.long()+1).clamp(1,T)   # (B,)
            kf_out = h[torch.arange(B,device=dev), kf_idx]  # (B,D)
            # Gated fusion: learn how much target frame vs global matters
            gate   = self.kf_gate(torch.cat([cls_out,kf_out],-1))  # (B,D)
            out    = gate*kf_out + (1.-gate)*cls_out
        else:
            out = cls_out

        return out   # (B, TEMPORAL_DIM)


class FactorizedClassifier(nn.Module):
    """
    Three heads sharing the same hidden representation:
      - 8-class joint head  (primary)
      - 2-class side head   (right/left team acting)
      - 4-class action head (set/spike/pass/winpoint)

    Composed logit:
      final(c) = joint(c) + 0.5*side(side_c) + 0.5*action(action_c)

    Why this helps:
      - r_spike and l_spike share action semantics → action head reinforces
      - r_pass and l_pass are the most confused pair → side head separates them
    """
    def __init__(self,in_dim,dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        # Slimmed: in→64→class (was in→128→class)
        self.joint  = nn.Sequential(
            nn.Linear(in_dim,64),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(64,N_CLASSES))
        self.side   = nn.Sequential(
            nn.Linear(in_dim,32),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(32,N_SIDES))
        self.action = nn.Sequential(
            nn.Linear(in_dim,32),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(32,N_ACTIONS))

        self.register_buffer("side_map",
            torch.tensor([0,0,0,0,1,1,1,1],dtype=torch.long))
        self.register_buffer("action_map",
            torch.tensor([0,1,2,3,0,1,2,3],dtype=torch.long))

    def forward(self,h):
        h  = self.drop(h)
        jl = self.joint(h)
        sl = self.side(h)
        al = self.action(h)
        composed = jl + 0.5*sl[:,self.side_map] + 0.5*al[:,self.action_map]
        return composed,sl,al


class TemporalClassifier(nn.Module):
    """
    Input:
      actor_tokens:  (B,T,N,D)
      key_token:     (B,D)
      team_tokens:   (B,T,2,D)
      present:       (B,T,N) bool
      key_frame_pos: (B,) int  ← target frame index in sampled sequence

    Parameters: ~180K (was ~450K)
    """
    def __init__(self,config):
        super().__init__()
        drop     = config.get("dropout_rate",0.15)
        n_heads  = config.get("n_heads",N_HEADS)
        n_layers = config.get("n_temporal_layers",3)

        # frame token = player pool (D) + 2 team tokens (2D) = 3D = 384
        # but D=128 → 128+256 = 384
        frame_in = ACTOR_DIM + ACTOR_DIM*2  # 128+256=384

        self.temporal   = TemporalTransformer(
            frame_in, n_layers, n_heads, FF_DIM, drop)

        # Fuse temporal CLS/kf output with key actor token
        self.fuse = nn.Sequential(
            nn.Linear(TEMPORAL_DIM+ACTOR_DIM, TEMPORAL_DIM),
            nn.GELU(), nn.LayerNorm(TEMPORAL_DIM))

        self.classifier = FactorizedClassifier(TEMPORAL_DIM,drop)

    def forward(self,actor_tokens,key_token,team_tokens,present,key_frame_pos=None):
        B,T,N,D = actor_tokens.shape

        # Masked mean pooling over players per frame
        pres  = present.float().unsqueeze(-1)
        wsum  = (actor_tokens*pres).sum(2)
        cnt   = pres.sum(2).clamp(min=1.)
        fp    = wsum/cnt                         # (B,T,D)

        # Team tokens per frame
        ft    = team_tokens.reshape(B,T,2*D)     # (B,T,2D)

        frame_feat = torch.cat([fp,ft],-1)       # (B,T,3D=384)

        # ── KEY FRAME FIX ────────────────────────────────────
        # Pass key_frame_pos so transformer extracts + fuses
        # the TARGET frame's representation (where label is defined).
        cls_h = self.temporal(frame_feat, key_frame_pos)  # (B,TEMPORAL_DIM)

        h  = self.fuse(torch.cat([cls_h,key_token],-1))
        jl,sl,al = self.classifier(h)

        return {"logits":jl,"side":sl,"action":al,"hidden":h}
