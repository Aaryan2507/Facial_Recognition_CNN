"""Face verification on UNSEEN identities: your ArcFace model vs pretrained FaceNet.

    python evaluate.py

Outputs: roc.png, results.json, and a printed table.
"""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

from common import load_lfw


def make_pairs(y, max_pairs=6000, seed=0):
    rng = np.random.default_rng(seed)
    by_id = {}
    for i, lab in enumerate(y):
        by_id.setdefault(lab, []).append(i)
    genuine = [p for idx in by_id.values() for p in itertools.combinations(idx, 2)]
    rng.shuffle(genuine)
    genuine = genuine[:max_pairs]
    impostor = []
    while len(impostor) < len(genuine):
        i, j = rng.integers(0, len(y), 2)
        if y[i] != y[j]:
            impostor.append((i, j))
    pairs = np.array(genuine + impostor)
    labels = np.array([1] * len(genuine) + [0] * len(impostor))
    return pairs, labels


def norm(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-10)


def metrics(emb, pairs, labels):
    emb = norm(emb)
    sims = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)
    fpr, tpr, thr = roc_curve(labels, sims)
    fnr = 1 - tpr
    eer = float(fpr[np.nanargmin(np.abs(fnr - fpr))])
    bal = (tpr + 1 - fpr) / 2
    best = int(np.argmax(bal))
    i1 = max(int(np.searchsorted(fpr, 0.01, side="right")) - 1, 1)   # last point with FAR <= 1%
    return dict(auc=float(roc_auc_score(labels, sims)), eer=eer,
                tar_at_far_1pct=float(np.interp(0.01, fpr, tpr)),
                best_accuracy=float(bal[best]),
                threshold_best_accuracy=float(np.clip(thr[best], -1.0, 1.0)),
                threshold_far_1pct=float(np.clip(thr[i1], -1.0, 1.0))), (fpr, tpr)


def main():
    d = load_lfw()
    X, y = d["Xte"], d["yte"]
    pairs, labels = make_pairs(y)
    print(f"{len(pairs)} pairs ({labels.sum()} genuine / {(1 - labels).sum()} impostor)")

    results, curves = {}, {}

    own = tf.keras.models.load_model("embedder.keras")
    results["Ours (MobileNetV2+ArcFace)"], curves["Ours (MobileNetV2+ArcFace)"] = metrics(
        own.predict(X, batch_size=64, verbose=0), pairs, labels)

    from keras_facenet import FaceNet
    fn = FaceNet()
    X160 = tf.image.resize(X, (160, 160)).numpy()
    results["Pretrained FaceNet"], curves["Pretrained FaceNet"] = metrics(
        fn.embeddings(X160), pairs, labels)

    print(f"{'model':32s} {'AUC':>6s} {'EER':>6s} {'TAR@1%FAR':>10s} {'Acc':>6s}")
    for k, m in results.items():
        print(f"{k:32s} {m['auc']:6.3f} {m['eer']:6.3f} {m['tar_at_far_1pct']:10.3f} "
              f"{m['best_accuracy']:6.3f}")

    print("\nSuggested --threshold values for demo.py (cosine similarity):")
    for k, m in results.items():
        print(f"  {k}: best-accuracy {m['threshold_best_accuracy']:.2f}, "
              f"strict (1% false accepts) {m['threshold_far_1pct']:.2f}")

    plt.figure(figsize=(5, 5))
    for k, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=f"{k} (AUC {results[k]['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.5)
    plt.xlabel("False accept rate"); plt.ylabel("True accept rate")
    plt.title("Verification on unseen LFW identities"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("roc.png", dpi=150)
    json.dump(results, open("results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
