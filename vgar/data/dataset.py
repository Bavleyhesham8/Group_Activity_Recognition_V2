
import pickle, numpy as np, torch, pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from vgar.data.features import *

# ── Official split ─────────────────────────────────────────
ALL_TRAIN = {1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54,
             0,2,8,12,17,19,24,26,27,28,30,33,46,49,51}
VAL_VIDS   = {1,3,6,7,10,13}
TRAIN_VIDS = ALL_TRAIN - VAL_VIDS
TEST_VIDS  = {4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47}


def _angle(v1, v2):
    dot  = (v1*v2).sum(-1)
    norm = (np.linalg.norm(v1,axis=-1)*np.linalg.norm(v2,axis=-1))+1e-6
    return np.clip(dot/norm,-1.,1.)


def sample_keyframes(num_frames, n_keyframes):
    """
    FIXED: Sample n_keyframes from ~20 annotated frames,
    guaranteeing the TARGET FRAME (middle, index=num_frames//2)
    is always included.

    Dataset info:
      - num_frames ≈ 20
      - target_frame = middle frame = index num_frames//2
      - label is defined at target_frame
    """
    target_idx = num_frames // 2   # always the middle frame

    if num_frames <= n_keyframes:
        # Fewer frames than needed: use all + edge-pad
        fidx = np.arange(num_frames)
        fidx = np.pad(fidx, (0, n_keyframes-num_frames), 'edge')
        # Find where target_idx is (or closest)
        kf_pos = int(np.argmin(np.abs(fidx - target_idx)))
        return fidx, kf_pos

    # Sample n_keyframes uniformly but FORCE target_idx in
    # Strategy: sample n_keyframes-1 uniformly (excluding target),
    # then insert target at the correct sorted position.
    candidates = np.delete(np.arange(num_frames), target_idx)

    # Pick n_keyframes-1 uniformly from candidates
    step = len(candidates) / (n_keyframes - 1)
    chosen_indices = [int(i*step) for i in range(n_keyframes-1)]
    chosen = candidates[chosen_indices]

    # Merge target_idx and sort
    fidx = np.sort(np.append(chosen, target_idx)).astype(int)
    # Where is the target?
    kf_pos = int(np.argmin(np.abs(fidx - target_idx)))
    return fidx, kf_pos


def compute_features(kp_raw, bbox, tf_arr, present_arr):
    """
    kp_raw:      (T, N, 17, 3)
    bbox:        (T, N, 4)   cx,cy,w,h normalized
    tf_arr:      (N,)         team 0/1
    present_arr: (T, N) bool
    Returns:     (T, N, FEAT_DIM=232)
    """
    T, N, J, _ = kp_raw.shape
    feat = np.zeros((T, N, FEAT_DIM), dtype=np.float32)

    # kp_global (0:51)
    feat[:,:,0:51] = kp_raw.reshape(T,N,51)

    # kp_relative (51:85): person-normalized
    cx = bbox[:,:,0:1]; cy = bbox[:,:,1:2]
    bh = bbox[:,:,3:4].clip(min=1e-3)
    for j in range(J):
        feat[:,:,51+j*2]   = (kp_raw[:,:,j,0]-cx[:,:,0])/bh[:,:,0]
        feat[:,:,51+j*2+1] = (kp_raw[:,:,j,1]-cy[:,:,0])/bh[:,:,0]

    # velocity (85:119)
    kp_xy = kp_raw[:,:,:,:2].reshape(T,N,J*2)
    vel   = np.zeros_like(kp_xy)
    vel[1:] = kp_xy[1:]-kp_xy[:-1]
    feat[:,:,85:119] = vel

    # acceleration (119:153)
    acc = np.zeros_like(vel)
    acc[2:] = vel[2:]-vel[1:-1]
    feat[:,:,119:153] = acc

    # bones (153:185)
    for i,(j1,j2) in enumerate(BONES):
        feat[:,:,153+i*2]   = kp_raw[:,:,j1,0]-kp_raw[:,:,j2,0]
        feat[:,:,153+i*2+1] = kp_raw[:,:,j1,1]-kp_raw[:,:,j2,1]

    # angles (185:204)
    for i,(a,b,c) in enumerate(ANGLE_TRIPLETS):
        v1 = kp_raw[:,:,a,:2]-kp_raw[:,:,b,:2]
        v2 = kp_raw[:,:,c,:2]-kp_raw[:,:,b,:2]
        feat[:,:,185+i] = _angle(v1,v2)

    # bbox (204:208)
    feat[:,:,204:208] = bbox

    # bbox_velocity (208:212)
    bvel = np.zeros_like(bbox)
    bvel[1:] = bbox[1:]-bbox[:-1]
    feat[:,:,208:212] = bvel

    # zone one-hot 2x3 (212:218)
    zx = np.clip((bbox[:,:,0]*3).astype(int),0,2)
    zy = np.clip((bbox[:,:,1]*2).astype(int),0,1)
    zi = zy*3+zx
    for t in range(T):
        for n in range(N):
            if present_arr[t,n]:
                feat[t,n,212+zi[t,n]] = 1.0

    # team one-hot (218:220) — ZERO for absent
    for n in range(N):
        for t in range(T):
            if present_arr[t,n]:
                feat[t,n,218+tf_arr[n]] = 1.0

    # net_signed (220:221)
    feat[:,:,220] = bbox[:,:,0]-0.5

    # net_abs (221:222)
    feat[:,:,221] = np.abs(bbox[:,:,0]-0.5)

    # jump_proxy (222:225)
    NOSE=0; LW=9; RW=10; LSH=5; RSH=6
    wrist_y = np.minimum(kp_raw[:,:,LW,1],kp_raw[:,:,RW,1])
    nose_y  = kp_raw[:,:,NOSE,1]
    sh_y    = (kp_raw[:,:,LSH,1]+kp_raw[:,:,RSH,1])/2.0
    feat[:,:,222] = nose_y-wrist_y
    feat[:,:,223] = sh_y-wrist_y
    feat[:,:,224] = bvel[:,:,3]

    # formation (225:231)
    for t in range(T):
        for tid in range(2):
            idx  = [n for n in range(N) if tf_arr[n]==tid and present_arr[t,n]]
            off  = tid*3
            if not idx: continue
            xs = bbox[t,idx,0]; ys = bbox[t,idx,1]
            feat[t,:,225+off]   = xs.mean()
            feat[t,:,225+off+1] = ys.mean()
            feat[t,:,225+off+2] = xs.std() if len(xs)>1 else 0.

    # present (231:232)
    feat[:,:,231] = present_arr.astype(np.float32)

    # zero absent entirely
    feat[~present_arr] = 0.
    return feat


class VolleyballDataset(Dataset):
    def __init__(self, config, split):
        assert split in ("train","val","test")
        self.split = split
        self.config = config
        self.aug = config.get("use_augmentation",True) and split=="train"
        self.rng  = np.random.default_rng(42 if split=="train" else 0)

        df   = pd.read_csv(META_DIR/"clips.csv")
        vids = {"train":TRAIN_VIDS,"val":VAL_VIDS,"test":TEST_VIDS}[split]
        df   = df[df['video_id'].astype(int).isin(vids)].reset_index(drop=True)

        self.clips  = [KP_DIR/str(r['video_id'])/f"{r['clip_id']}.pkl"
                       for _,r in df.iterrows()]
        self.labels = [GROUP_VOCAB.get(normalise_label(r['group_activity']),0)
                       for _,r in df.iterrows()]
        print(f"  [{split:5s}] {len(self.clips):,} clips")

    def __len__(self): return len(self.clips)

    def _load(self, idx):
        with open(self.clips[idx],'rb') as f:
            data = pickle.load(f)
        label = self.labels[idx]

        players = sorted(data['players'].values(), key=lambda p:p['team'])
        nf      = data['num_frames']

        # ── FIXED: centered sampling with guaranteed target frame ──
        fidx, kf_pos = sample_keyframes(nf, N_KEYFRAMES)

        kp_raw  = np.zeros((N_KEYFRAMES,N_PLAYERS,N_JOINTS,3),np.float32)
        bbox    = np.zeros((N_KEYFRAMES,N_PLAYERS,4),          np.float32)
        ks      = np.zeros((N_KEYFRAMES,N_PLAYERS),            np.float32)
        tf      = np.zeros((N_PLAYERS,),                       np.int64)
        present = np.zeros((N_KEYFRAMES,N_PLAYERS),            bool)

        for i,p in enumerate(players[:N_PLAYERS]):
            pk = p['keypoints'][fidx]
            pb = p['bboxes'][fidx]
            pc = p['bbox_conf'][fidx]

            kp_raw[:,i] = pk
            ks[:,i]     = pc
            tf[i]       = p['team']

            x1,y1,x2,y2 = pb[:,0],pb[:,1],pb[:,2],pb[:,3]
            bbox[:,i,0]  = (x1+x2)/2.
            bbox[:,i,1]  = (y1+y2)/2.
            bbox[:,i,2]  = x2-x1
            bbox[:,i,3]  = y2-y1

            # FIXED present mask: bbox has area AND keypoints non-zero
            present[:,i] = (pc>0) & ((x2-x1)>0) & ((pk[:,:,0]!=0).any(-1))

        # Normalize to [0,1]
        mx = kp_raw[:,:,:,:2].max()
        if mx>2.:
            kp_raw[:,:,:,:2] /= mx
            bbox[:,:,:2]     /= mx
            bmax = bbox[:,:,2:].max()
            if bmax>2.: bbox[:,:,2:] /= bmax

        feat    = compute_features(kp_raw, bbox, tf, present)
        kp_flat = kp_raw.reshape(N_KEYFRAMES,N_PLAYERS,51)

        return {
            "kp":           feat,
            "kp_raw":       kp_flat,
            "bbox":         bbox,
            "tf":           tf,
            "present":      present,
            "key_score":    ks,
            "label":        label,
            "side":         SIDE_LABEL[label],
            "action":       ACTION_LABEL[label],
            "key_frame_pos":kf_pos,   # index in [0..N_KEYFRAMES-1]
            "seq_idx":      idx,
        }

    def __getitem__(self, idx):
        s = self._load(idx)
        if self.aug:
            s = self._augment(s)
        return s

    def _augment(self, s):
        # 1. Horizontal flip
        if self.rng.random() < 0.5:
            s = self._hflip(s)
        # 2. Keypoint jitter
        if self.rng.random() < 0.5:
            n = self.rng.normal(0,0.008,s["kp_raw"].shape).astype(np.float32)
            n[:,:,2::3] = 0.
            s["kp_raw"] = np.clip(s["kp_raw"]+n,0,1)
        # 3. Player dropout (0-2 players)
        if self.rng.random() < 0.4:
            nd = self.rng.integers(1,3)
            di = self.rng.choice(N_PLAYERS,nd,replace=False)
            s["present"][:,di]    = False
            s["key_score"][:,di]  = 0.
        # 4. Temporal jitter — CAREFUL: do not shift key_frame_pos
        # We skip temporal roll to keep key frame alignment
        return s

    def _hflip(self, s):
        kpr  = s["kp_raw"].copy()
        bb   = s["bbox"].copy()
        tf   = s["tf"].copy()
        pr   = s["present"].copy()
        lbl  = FLIP_LABEL[s["label"]]

        kpr[:,:,0::3] = 1.-kpr[:,:,0::3]
        bb[:,:,0]     = 1.-bb[:,:,0]
        tf = np.where(tf==0,1,0).astype(tf.dtype)

        kpr4 = kpr.reshape(N_KEYFRAMES,N_PLAYERS,N_JOINTS,3)
        feat = compute_features(kpr4,bb,tf,pr)

        return {**s,"kp":feat,"kp_raw":kpr,"bbox":bb,"tf":tf,
                "label":lbl,"side":SIDE_LABEL[lbl],"action":ACTION_LABEL[lbl]}

    def get_class_weights(self):
        lbl  = np.array(self.labels)
        cnt  = np.bincount(lbl,minlength=N_CLASSES).astype(np.float32)
        cnt  = np.where(cnt==0,1.,cnt)
        beta = 0.999
        eff  = (1.-beta**cnt)/(1.-beta)
        w    = 1./eff
        w    = w/w.sum()*N_CLASSES
        return torch.from_numpy(w).float()


def collate_fn(batch):
    out = {}
    for k in ["kp","kp_raw","bbox","key_score"]:
        out[k] = torch.from_numpy(np.stack([s[k] for s in batch])).float()
    for k in ["tf","label","side","action","seq_idx","key_frame_pos"]:
        out[k] = torch.from_numpy(np.array([s[k] for s in batch])).long()
    out["present"] = torch.from_numpy(np.stack([s["present"] for s in batch]))
    return out


def build_loaders(config):
    bs = config.get("batch_size",32)
    nw = config.get("num_workers",2)
    tr = VolleyballDataset(config,"train")
    vl = VolleyballDataset(config,"val")
    te = VolleyballDataset(config,"test")
    trl = DataLoader(tr,bs,shuffle=True, num_workers=nw,pin_memory=True,
                     drop_last=True,collate_fn=collate_fn)
    vll = DataLoader(vl,bs*2,shuffle=False,num_workers=nw,pin_memory=True,
                     collate_fn=collate_fn)
    tel = DataLoader(te,bs*2,shuffle=False,num_workers=nw,pin_memory=True,
                     collate_fn=collate_fn)
    return trl,vll,tel,tr.get_class_weights()
