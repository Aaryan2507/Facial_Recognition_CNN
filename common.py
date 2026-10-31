"""Shared data loading: LFW with an identity-disjoint split (the proper verification protocol)."""
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

IMG = 112


def load_lfw(min_faces=10, seed=42, test_id_frac=0.2, val_img_frac=0.1):
    """Returns dict with train/val (seen identities, relabelled 0..K-1) and
    test (identities the model has NEVER seen, original ids)."""
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces, color=True, resize=1.0,
        slice_=(slice(60, 190), slice(60, 190)),
    )
    X = (lfw.images * 255.0).astype("float32")           # sklearn scales to [0,1]
    X = tf.image.resize(X, (IMG, IMG)).numpy()
    y = lfw.target

    ids = np.unique(y)
    train_ids, test_ids = train_test_split(ids, test_size=test_id_frac, random_state=seed)
    tr_mask, te_mask = np.isin(y, train_ids), np.isin(y, test_ids)

    remap = {old: new for new, old in enumerate(sorted(train_ids))}
    Xtr, ytr = X[tr_mask], np.array([remap[v] for v in y[tr_mask]], dtype="int32")
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr, ytr, test_size=val_img_frac, stratify=ytr, random_state=seed)

    print(f"train imgs {len(Xtr)}, val imgs {len(Xval)}, classes {len(train_ids)}, "
          f"test imgs {te_mask.sum()} from {len(test_ids)} unseen identities")
    return dict(Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval,
                Xte=X[te_mask], yte=y[te_mask], n_classes=len(train_ids))
