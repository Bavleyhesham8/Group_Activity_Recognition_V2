
import torch, torch.nn as nn, torch.nn.functional as F
from vgar.data.features import N_PLAYERS, N_KEYFRAMES
from vgar.models.actor_encoder import ACTOR_DIM

# ── Reduced dims ───────────────────────────────────────────
# Old: N_HEADS=8, FF_DIM=256, N_LAYERS=3 → very heavy
# New: N_HEADS=4, FF_DIM=128, N_LAYERS=2 → ~4x fewer params
N_HEADS  = 4
FF_DIM   = 128
N_LAYERS = 2
TEAM_DIM = ACTOR_DIM  # 128


class RelationBiasNet(nn.Module):
    """
    8 pairwise features → N_HEADS scalar biases.
    Slimmed: 8→24→N_HEADS (was 8→32→N_HEADS)
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8,24),nn.GELU(),nn.Linear(24,N_HEADS))

    def forward(self,bbox,tf,present):
        B,T,N,_ = bbox.shape
        cx = bbox[:,:,:,0]; cy = bbox[:,:,:,1]
        dx   = cx.unsqueeze(3)-cx.unsqueeze(2)
        dy   = cy.unsqueeze(3)-cy.unsqueeze(2)
        dist = (dx**2+dy**2).sqrt()+1e-6
        tf_e = tf.unsqueeze(1).expand(B,T,N)
        same = (tf_e.unsqueeze(3)==tf_e.unsqueeze(2)).float()
        opp  = 1.-same
        vx   = torch.zeros_like(cx); vy=torch.zeros_like(cy)
        vx[:,1:]=cx[:,1:]-cx[:,:-1]; vy[:,1:]=cy[:,1:]-cy[:,:-1]
        dvx  = vx.unsqueeze(3)-vx.unsqueeze(2)
        dvy  = vy.unsqueeze(3)-vy.unsqueeze(2)
        nd   = (cx-0.5).unsqueeze(3)-(cx-0.5).unsqueeze(2)
        rel  = torch.stack([dx,dy,dist,same,opp,dvx,dvy,nd],-1)
        bias = self.mlp(rel).permute(0,1,4,2,3)  # (B,T,H,N,N)
        pm   = present.float()
        pair = pm.unsqueeze(3)*pm.unsqueeze(2)
        bias = bias*pair.unsqueeze(2)
        return bias.reshape(B*T,N_HEADS,N,N)


class MHSAttention(nn.Module):
    """Efficient multi-head self-attention."""
    def __init__(self,dim,n_heads,dropout):
        super().__init__()
        self.H  = n_heads
        self.hd = dim//n_heads
        self.sc = self.hd**-0.5
        self.Wqkv = nn.Linear(dim,3*dim,bias=False)
        self.Wo   = nn.Linear(dim,dim,bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self,x,bias=None,mask=None):
        B,L,D = x.shape
        qkv   = self.Wqkv(x).reshape(B,L,3,self.H,self.hd)
        q,k,v = qkv.unbind(2)
        q=q.permute(0,2,1,3); k=k.permute(0,2,1,3); v=v.permute(0,2,1,3)
        attn  = torch.matmul(q,k.transpose(-2,-1))*self.sc
        if bias is not None: attn=attn+bias
        if mask is not None: attn=attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2),-1e9)
        attn  = self.drop(F.softmax(attn,-1))
        return self.Wo(torch.matmul(attn,v).permute(0,2,1,3).reshape(B,L,D))


class TBlock(nn.Module):
    def __init__(self,dim,n_heads,ff,drop):
        super().__init__()
        self.attn = MHSAttention(dim,n_heads,drop)
        self.ff   = nn.Sequential(
            nn.Linear(dim,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,dim))
        self.n1   = nn.LayerNorm(dim)
        self.n2   = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self,x,bias=None,mask=None):
        x = x+self.drop(self.attn(self.n1(x),bias,mask))
        x = x+self.drop(self.ff(self.n2(x)))
        return x


class PlayerInteractionModule(nn.Module):
    """
    2 layers of spatial + temporal attention.
    Parameters: ~500K (was ~1.8M with 3 layers, 8 heads, FF=256)

    Key additions vs old code:
    - Relation bias per head (team/distance/velocity aware)
    - 2 learned team tokens
    - Key actor soft attention
    - Proper positional encoding (temporal + player-id)
    """
    def __init__(self,config):
        super().__init__()
        drop = config.get("dropout_rate",0.15)

        self.rel_bias      = RelationBiasNet()
        self.spatial_layers= nn.ModuleList([
            TBlock(ACTOR_DIM,N_HEADS,FF_DIM,drop) for _ in range(N_LAYERS)])
        self.temporal_layers=nn.ModuleList([
            TBlock(ACTOR_DIM,N_HEADS,FF_DIM,drop) for _ in range(N_LAYERS)])

        self.team_tokens   = nn.Parameter(torch.randn(2,ACTOR_DIM)*0.02)
        self.key_actor_q   = nn.Linear(ACTOR_DIM,1,bias=False)
        self.temp_pe       = nn.Embedding(N_KEYFRAMES,ACTOR_DIM)
        self.player_pe     = nn.Embedding(N_PLAYERS+2,ACTOR_DIM)
        self.norm          = nn.LayerNorm(ACTOR_DIM)

    def forward(self,actor_tokens,bbox,tf,present):
        B,T,N,D = actor_tokens.shape
        dev = actor_tokens.device

        # Positional encodings
        t_pe = self.temp_pe(torch.arange(T,device=dev))    # (T,D)
        n_pe = self.player_pe(torch.arange(N,device=dev))  # (N,D)
        x    = actor_tokens+t_pe.unsqueeze(1)+n_pe.unsqueeze(0)

        # Team tokens
        tt   = self.team_tokens.unsqueeze(0).unsqueeze(0).expand(B,T,2,D)
        tt_pe= self.player_pe(torch.tensor([N,N+1],device=dev))
        tt   = tt+t_pe.unsqueeze(1)+tt_pe.unsqueeze(0)
        x_full = torch.cat([x,tt],dim=2)  # (B,T,N+2,D)

        pres_full = torch.cat([
            present,
            torch.ones(B,T,2,dtype=torch.bool,device=dev)],dim=2)

        # Relation bias padded for N+2
        rb_np = self.rel_bias(bbox,tf,present)  # (B*T,H,N,N)
        N2    = N+2
        rb    = torch.zeros(B*T,N_HEADS,N2,N2,device=dev)
        rb[:,:,:N,:N] = rb_np

        # Alternate spatial + temporal
        for sp,tp in zip(self.spatial_layers,self.temporal_layers):
            xs  = x_full.reshape(B*T,N2,D)
            ms  = pres_full.reshape(B*T,N2)
            xs  = sp(xs,bias=rb,mask=ms)
            x_full = xs.reshape(B,T,N2,D)

            xt  = x_full.permute(0,2,1,3).reshape(B*N2,T,D)
            xt  = tp(xt)
            x_full = xt.reshape(B,N2,T,D).permute(0,2,1,3)

        x_full = self.norm(x_full)
        out_tokens = x_full[:,:,:N,:]   # (B,T,N,D)
        out_teams  = x_full[:,:,N:,:]   # (B,T,2,D)

        # Key actor: soft attention over T*N
        flat  = out_tokens.reshape(B,T*N,D)
        kw    = self.key_actor_q(flat).squeeze(-1)
        pf    = present.reshape(B,T*N)
        kw    = kw.masked_fill(~pf,-1e9)
        kw    = torch.softmax(kw,-1)
        key_t = (kw.unsqueeze(-1)*flat).sum(1)  # (B,D)

        return out_tokens, key_t, out_teams
