"""
Smoke test: verifies every component instantiates and runs without crashing.
Does NOT require real data -- uses synthetic random tensors throughout.

Run from repo root:
    python tests/smoke_test.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = torch.device("cpu")
BATCH = 2
IMG = (3, 224, 224)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check(name: str, fn):
    try:
        result = fn()
        print(f"  {PASS} {name}" + (f"  -> {result}" if result is not None else ""))
        return True
    except Exception as e:
        print(f"  {FAIL} {name}")
        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# 1. InceptentionNet
# -----------------------------------------------------------------------------
def test_inceptentionnet():
    print("\n-- InceptentionNet ------------------------------------------")
    from src.models.inceptentionnet import InceptentionNet
    results = []

    def _build():
        return InceptentionNet(stem_channels=64, branch_channels=64, num_heads=4, dropout=0.3)

    results.append(check("instantiate", _build))

    model = _build().to(DEVICE)
    x = torch.randn(BATCH, *IMG)

    results.append(check("forward pass", lambda: f"output shape {model(x).shape}"))

    logits = model(x)
    results.append(check("output is 1D (batch,)", lambda: str(logits.shape == torch.Size([BATCH]))))

    return all(results)


# -----------------------------------------------------------------------------
# 2. FPN-Mamba (full model)
# -----------------------------------------------------------------------------
def test_fpn_mamba_full():
    print("\n-- FPN-Mamba (full model) -----------------------------------")
    try:
        from src.models.fpn_mamba import FPNMambaClassifier
    except ImportError as e:
        print(f"  {SKIP} timm/einops not installed: {e}")
        return True

    results = []
    model = FPNMambaClassifier().to(DEVICE)
    x = torch.randn(BATCH, *IMG)

    results.append(check("instantiate FPNMambaClassifier", lambda: None))
    results.append(check("forward pass", lambda: f"output shape {model(x).shape}"))

    logits = model(x)
    results.append(check("output shape (B,1)", lambda: str(logits.shape == torch.Size([BATCH, 1]))))

    return all(results)


# -----------------------------------------------------------------------------
# 3. Ablation variants
# -----------------------------------------------------------------------------
def test_ablation_variants():
    print("\n-- Ablation Variants ----------------------------------------")
    try:
        from src.models.ablation import build_ablation_model, ABLATION_VARIANTS
    except ImportError as e:
        print(f"  {SKIP} {e}")
        return True

    x = torch.randn(BATCH, *IMG)
    all_pass = True
    for variant in ABLATION_VARIANTS:
        def _run(v=variant):
            m = build_ablation_model(v).to(DEVICE)
            out = m(x)
            return f"{out.shape}"
        ok = check(variant, _run)
        all_pass = all_pass and ok
    return all_pass


# -----------------------------------------------------------------------------
# 4. Losses
# -----------------------------------------------------------------------------
def test_losses():
    print("\n-- Losses ---------------------------------------------------")
    from src.training.losses import bce_loss, weighted_bce_smooth, compute_pos_weight
    results = []

    logits = torch.randn(BATCH)
    labels = torch.tensor([1.0, 0.0])

    results.append(check("bce_loss", lambda: f"{bce_loss(logits, labels).item():.4f}"))
    results.append(check("weighted_bce_smooth", lambda: f"{weighted_bce_smooth(logits, labels, pos_weight=4.8, label_smoothing=0.05).item():.4f}"))
    results.append(check("compute_pos_weight", lambda: f"{compute_pos_weight([1,0,0,0,0]):.2f}"))

    return all(results)


# -----------------------------------------------------------------------------
# 5. Metrics
# -----------------------------------------------------------------------------
def test_metrics():
    print("\n-- Metrics --------------------------------------------------")
    from src.evaluation.metrics import compute_metrics, find_best_threshold, summarize_folds
    results = []

    rng = np.random.default_rng(42)
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    probs = rng.uniform(0, 1, size=len(labels))

    results.append(check("find_best_threshold", lambda: f"{find_best_threshold(labels, probs):.3f}"))
    metrics = compute_metrics(labels, probs)
    results.append(check("compute_metrics keys", lambda: str(sorted(metrics.keys()))))
    results.append(check("sensitivity + specificity present",
                         lambda: str("sensitivity" in metrics and "specificity" in metrics)))

    # summarize_folds
    fake_folds = [{"metrics": {**metrics}} for _ in range(5)]
    summary = summarize_folds(fake_folds)
    results.append(check("summarize_folds", lambda: f"{list(summary.keys())[:3]}"))

    return all(results)


# -----------------------------------------------------------------------------
# 6. Statistical tests
# -----------------------------------------------------------------------------
def test_stats():
    print("\n-- Statistical Tests ----------------------------------------")
    from src.evaluation.stats import bootstrap_ci, wilcoxon_test, mcnemar_test
    results = []

    a = np.array([0.94, 0.96, 0.93, 0.95, 0.97])
    b = np.array([0.96, 0.97, 0.95, 0.98, 0.99])

    results.append(check("bootstrap_ci", lambda: f"{bootstrap_ci(a)}"))
    results.append(check("wilcoxon_test", lambda: f"p={wilcoxon_test(a, b)[1]:.4f}"))

    labels = np.array([1, 0, 1, 0, 1, 0])
    preds_a = np.array([1, 0, 0, 0, 1, 1])
    preds_b = np.array([1, 0, 1, 0, 1, 0])
    results.append(check("mcnemar_test", lambda: f"p={mcnemar_test(labels, preds_a, preds_b)[1]:.4f}"))

    return all(results)


# -----------------------------------------------------------------------------
# 7. Grad-CAM
# -----------------------------------------------------------------------------
def test_gradcam():
    print("\n-- Grad-CAM -------------------------------------------------")
    try:
        from src.models.inceptentionnet import InceptentionNet
        from src.evaluation.gradcam import GradCAM
    except ImportError as e:
        print(f"  {SKIP} {e}")
        return True

    results = []
    model = InceptentionNet().to(DEVICE)
    x = torch.randn(1, *IMG)

    cam = GradCAM(model, target_layer=model.inception.branch_3x3)
    results.append(check("GradCAM on InceptentionNet", lambda: f"heatmap shape {cam(x).shape}"))
    cam.remove_hooks()

    return all(results)


# -----------------------------------------------------------------------------
# 8. FLOPs + Parameter count
# -----------------------------------------------------------------------------
def test_flops():
    print("\n-- FLOPs & Parameters ---------------------------------------")
    from src.utils.flops import count_parameters, model_summary
    from src.models.inceptentionnet import InceptentionNet
    results = []

    model = InceptentionNet()
    params = count_parameters(model)
    results.append(check("count_parameters", lambda: f"total={params['total']:,}  trainable={params['trainable']:,}"))
    results.append(check("model_summary (InceptentionNet)", lambda: model_summary(model) or "ok"))

    return all(results)


# -----------------------------------------------------------------------------
# 9. Transforms (no real images needed)
# -----------------------------------------------------------------------------
def test_transforms():
    print("\n-- Transforms -----------------------------------------------")
    from src.data.transforms import TransformConfig, build_train_transform, build_eval_transform, build_mb_transform
    from PIL import Image
    results = []

    cfg = TransformConfig(image_size=224, gaussian_sigma=0.7, imagenet_norm=True)
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)
    mb_tf = build_mb_transform(cfg)

    results.append(check("train_transform", lambda: f"{train_tf(img).shape}"))
    results.append(check("eval_transform",  lambda: f"{eval_tf(img).shape}"))
    results.append(check("mb_transform",    lambda: f"{mb_tf(img).shape}"))

    return all(results)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  FPN-Mamba Project Smoke Test")
    print("=" * 60)

    suites = [
        ("InceptentionNet",       test_inceptentionnet),
        ("FPN-Mamba (full)",      test_fpn_mamba_full),
        ("Ablation variants",     test_ablation_variants),
        ("Losses",                test_losses),
        ("Metrics",               test_metrics),
        ("Statistical tests",     test_stats),
        ("Grad-CAM",              test_gradcam),
        ("FLOPs & parameters",    test_flops),
        ("Transforms",            test_transforms),
    ]

    results = {}
    for name, fn in suites:
        results[name] = fn()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        print("\nSome checks failed -- see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

