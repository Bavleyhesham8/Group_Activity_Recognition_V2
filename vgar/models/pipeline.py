
import torch, torch.nn as nn
from vgar.models.actor_encoder      import ActorEncoder
from vgar.models.relation_transformer import PlayerInteractionModule
from vgar.models.temporal_classifier  import TemporalClassifier


class VATFormer(nn.Module):
    """
    Volleyball Actor-Team Transformer (VAT-Former) — Slim Edition

    Pipeline:
      ActorEncoder  →  PlayerInteractionModule  →  TemporalClassifier

    Key fixes vs old code:
      1. Present mask uses bbox+keypoint validity (not kp_ext[0::3])
      2. Frame sampling centers on target frame (guaranteed inclusion)
      3. key_frame_pos passed all the way to temporal transformer
      4. Bidirectional (not causal) temporal attention
      5. Factorized side/action classifier
      6. ~1.2M params (was 3.5M)
    """
    def __init__(self,config):
        super().__init__()
        self.config     = config
        self.actor_enc  = ActorEncoder(config)
        self.player_int = PlayerInteractionModule(config)
        self.classifier = TemporalClassifier(config)

    def forward(self,batch):
        feat        = batch["kp"]           # (B,T,N,232)
        kp_raw      = batch["kp_raw"]       # (B,T,N,51)
        bbox        = batch["bbox"]         # (B,T,N,4)
        tf          = batch["tf"]           # (B,N)
        present     = batch["present"]      # (B,T,N) bool
        kfp         = batch.get("key_frame_pos",None)  # (B,) or None

        actor_tokens = self.actor_enc(feat,kp_raw,present)
        enriched, key_token, team_tokens = self.player_int(
            actor_tokens,bbox,tf,present)
        out = self.classifier(
            enriched,key_token,team_tokens,present,
            key_frame_pos=kfp)
        return out
