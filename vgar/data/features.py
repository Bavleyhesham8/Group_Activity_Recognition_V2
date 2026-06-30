
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/datasets/bavleyhesham/volleyball-gar-checkpoint/volleyball_pose_data")
KP_DIR    = DATA_ROOT / "keypoints"
META_DIR  = DATA_ROOT / "metadata"
RESULTS   = Path("/kaggle/working/results")
RESULTS.mkdir(exist_ok=True)

# ── Label vocabulary ───────────────────────────────────────
GROUP_VOCAB = {
    "r_set":0,"r_spike":1,"r_pass":2,"r_winpoint":3,
    "l_set":4,"l_spike":5,"l_pass":6,"l_winpoint":7,
    "r-pass":2,"l-spike":5,"l-pass":6,
}
IDX_TO_CLASS = {
    0:"r_set",1:"r_spike",2:"r_pass",3:"r_winpoint",
    4:"l_set",5:"l_spike",6:"l_pass",7:"l_winpoint"
}
N_CLASSES = 8

# ── Factorized labels ──────────────────────────────────────
SIDE_LABEL   = {0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1}
ACTION_LABEL = {0:0,1:1,2:2,3:3,4:0,5:1,6:2,7:3}
N_SIDES   = 2
N_ACTIONS = 4
FLIP_LABEL= {0:4,1:5,2:6,3:7,4:0,5:1,6:2,7:3}

# ── Dataset constants ──────────────────────────────────────
N_PLAYERS   = 12
N_KEYFRAMES = 10   # we sample 10 from ~20 annotated frames
N_JOINTS    = 17
KP_RAW_DIM  = 51   # 17*3

# ── Skeleton ───────────────────────────────────────────────
BONES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]
N_BONES = len(BONES)  # 16

ANGLE_TRIPLETS = [
    (5,7,9),(6,8,10),
    (7,5,11),(8,6,12),
    (11,13,15),(12,14,16),
    (5,11,13),(6,12,14),
    (0,5,6),(5,0,6),
    (11,5,7),(12,6,8),
    (13,11,5),(14,12,6),
    (15,13,11),(16,14,12),
    (1,0,2),(3,1,5),(4,2,6)
]
N_ANGLES = len(ANGLE_TRIPLETS)  # 19

# ── Feature layout ─────────────────────────────────────────
# kp_global:    51  (17*3)
# kp_relative:  34  (17*2, person-normalized x,y)
# velocity:     34  (17*2)
# acceleration: 34  (17*2)
# bones:        32  (16*2)
# angles:       19
# bbox:          4  (cx,cy,w,h)
# bbox_vel:      4
# zone:          6  (2x3 grid one-hot)
# team:          2  (one-hot, zeroed if absent)
# net_signed:    1  (cx - 0.5)
# net_abs:       1
# jump_proxy:    3
# formation:     6  (team centroid x,y + spread x, per team)
# present:       1
# TOTAL:       232
FEAT_DIM = 232

FEAT_SLICES = {
    'kp_global':    (0,   51),
    'kp_relative':  (51,  85),
    'velocity':     (85,  119),
    'acceleration': (119, 153),
    'bones':        (153, 185),
    'angles':       (185, 204),
    'bbox':         (204, 208),
    'bbox_vel':     (208, 212),
    'zone':         (212, 218),
    'team':         (218, 220),
    'net_signed':   (220, 221),
    'net_abs':      (221, 222),
    'jump_proxy':   (222, 225),
    'formation':    (225, 231),
    'present':      (231, 232),
}

def normalise_label(s):
    return str(s).strip().lower().replace("-","_")
