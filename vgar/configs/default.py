
DEFAULT_CONFIG = {
    "batch_size":        32,
    "num_workers":       2,
    "use_augmentation":  True,

    # Model (slim)
    "dropout_rate":      0.15,
    "n_heads":           4,
    "n_temporal_layers": 3,

    # Loss
    "focal_gamma":       2.0,
    "label_smoothing":   0.05,
    "w_side":            0.5,
    "w_action":          0.7,

    # Optimizer
    "lr":                3e-4,
    "lr_min":            1e-6,
    "weight_decay":      0.05,
    "grad_clip":         1.0,
    "warmup_epochs":     5,

    # Phases
    "phase1_epochs":     15,
    "phase2_epochs":     20,
    "phase3_epochs":     30,
    "phase4_epochs":     35,
}
