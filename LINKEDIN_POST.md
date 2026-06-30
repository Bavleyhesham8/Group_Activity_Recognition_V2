# LinkedIn Post — Group Activity Recognition V2

---

We just published Version 2 of our Group Activity Recognition project, and the numbers speak for themselves.

91.10% test accuracy. 96.86% top-2. ~1.2M parameters. On the Volleyball dataset benchmark.

This is not a small tweak. V2 is a ground-up redesign of everything — the architecture, the training strategy, the feature engineering, and how we think about the problem.

---

The problem: given a volleyball game clip (~20 frames), classify what the entire group of players is doing — setting, spiking, passing, or winning a point — for the left or right team. That is 8 classes, 12 players, noisy detections, fast motion, and occlusion.

The original dataset (Ibrahim et al., CVPR 2016) has 4,830 clips across 55 real match videos. We run pose estimation with YOLOv8-Pose to extract 17 keypoints per player, per frame, and build a 232-dimensional feature vector covering skeleton geometry, motion, bounding box context, team formation, and court zone.

---

What changed in V2:

V1 used a sequential stack with a causal (last-frame-only) readout, no team context, no inter-player spatial reasoning, and a single linear classifier. It worked, but there was a clear ceiling.

V2 introduces the VAT-Former — Volleyball Actor-Team Transformer — a three-stage pipeline:

Stage 1 — ActorEncoder
Each player is encoded independently using a 2-layer Skeleton GCN over joint connectivity, a Motion MLP over velocity and acceleration signals, and a Context MLP over bounding box, team, zone, formation, and net-distance features. This produces a 128-dimensional actor token per player per frame.

Stage 2 — PlayerInteractionModule
Two layers of alternating spatial and temporal attention model who is interacting with whom and how that changes over time. A Relation Bias Network injects pairwise geometric and team-membership priors into each attention head, so the model knows if two players are on the same team, how far apart they are, and how their velocities compare — without having to learn that from scratch. Two learned team tokens act as global team-level context.

Stage 3 — TemporalClassifier
A bidirectional Transformer reads the sequence of frame-level features. The key fix: the volleyball label is defined at the target (middle) frame, not the last frame. We track exactly which sampled frame is the target and use a learned gate to fuse the global CLS token with that target frame's representation. A factorized classifier then produces three outputs simultaneously — the 8-class joint prediction, a 2-class team-side head, and a 4-class activity-type head — with their logits composed at inference.

---

Training uses a 4-phase curriculum:

Phase 1 warms up the actor encoder and classifier with only those components active.
Phase 2 adds the interaction module.
Phase 3 runs full end-to-end training.
Phase 4 fine-tunes with a lower learning rate.

Total: 100 epochs, cosine annealing with warmup, AdamW, focal loss with label smoothing, and class-weighted effective number rebalancing.

---

The results:

Top-1: 91.10%
Top-2: 96.86%
Top-3: 98.58%
Macro F1: 91.07%

All 8 classes above 87.5% F1. Best class (l_set) at 93.1%.

The model has ~1.2M parameters. That is roughly 3x fewer than V1, and it outperforms it.

---

The full code, all evaluation outputs, confusion matrices, t-SNE and UMAP embeddings, per-class qualitative rankings, and sample prediction videos are available in the repository.

Code: github.com/Bavleyhesham8/Group_Activity_Recognition_V2
Dataset (preprocessed): Kaggle — Volleyball GAR Pose Dataset

---

This started as a group project and grew into something I am genuinely proud of. Building a model that actually understands collective behavior — not just individual actions — is a different kind of challenge, and V2 shows how far careful design goes.

If you are working on multi-person action recognition, sports analytics, or video understanding, feel free to reach out.

---

#ComputerVision #DeepLearning #ActionRecognition #Transformers #PyTorch #GraphNeuralNetworks #SportsAnalytics #MachineLearning #Research
