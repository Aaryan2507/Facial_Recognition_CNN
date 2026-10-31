"""Live face recognition + face/no-face checking, with two interchangeable models.

ENROLL (guided: the app tells you which way to turn and waits before sampling)
  python demo.py enroll --name Aaryan                       # FaceNet (pretrained)
  python demo.py enroll --name Aaryan --model embedder.keras  # your own model
  python demo.py enroll --name Aaryan --manual              # press SPACE for each sample
  python demo.py enroll --images faces/                     # faces/<person>/*.jpg

RECOGNISE
  python demo.py run                                        # FaceNet
  python demo.py run --model embedder.keras --threshold 0.5   # your model
  python demo.py compare --truth Aaryan --thr-custom 0.5    # both, side by side

IS IT A FACE AT ALL?  (no enrolment needed)
  python demo.py facecheck                                  # live webcam: FACE / NO FACE
  python demo.py facecheck --images photo.jpg               # one image
  python demo.py facecheck --images some_folder --out checked/   # many images, saves annotated copies

OTHER
  python demo.py list                                       # show enrolled people

Each model keeps its own gallery (FaceNet: gallery.npz, custom: gallery_custom.npz) because their
embeddings are not interchangeable. Press q in the video window to quit.

Speed notes: detection runs in a background thread, so the video stays smooth; the labels
update a few times per second. If it is still slow, try --det-width 320 and --min-face 60.
"""
import argparse
import collections
import glob
import math
import os
import re
import sys
import threading
import time

import cv2
import numpy as np

DEFAULT_THR = 0.5
Face = collections.namedtuple("Face", "box conf")   # box = (x, y, w, h) in full-frame pixels

POSES = [
    "Look straight at the camera",
    "Turn your head a little to YOUR LEFT",
    "Turn your head a little to YOUR RIGHT",
    "Tilt your head UP a little",
    "Tilt your head DOWN a little",
    "Relax your face, then smile slightly",
]
GREEN, RED, AMBER, WHITE = (0, 200, 0), (0, 0, 255), (0, 160, 255), (255, 255, 255)


# ---------------------------------------------------------------- helpers
def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def load_gallery(path):
    if not os.path.exists(path):
        return {}
    d = np.load(path)
    return {k: d[k] for k in d.files}


def add_to_gallery(path, name, embs):
    g = load_gallery(path)
    embs = normalize(np.asarray(embs))
    name = safe(name)
    g[name] = np.vstack([g[name], embs]) if name in g else embs
    np.savez(path, **g)
    print(f"Saved {len(embs)} samples for '{name}' in {path}. Enrolled: {list(g)}")


def match(emb, gallery, threshold):
    emb = normalize(emb)
    best_name, best_sim = "Unknown", -1.0
    for name, samples in gallery.items():
        sim = float(np.max(samples @ emb))
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim < threshold:
        return "Unknown", best_sim
    return best_name, best_sim


def quiet_keras():
    """Stop Keras printing a progress bar on every call (slow and spammy on Windows)."""
    try:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception:
        pass
    try:
        import keras
        keras.config.disable_interactive_logging()
    except Exception:
        pass


def tf_fast(model):
    """Call a Keras model directly as a compiled graph. Same outputs as model.predict(), but
    without predict()'s large per-call overhead (this is what makes detection ~10x faster)."""
    import tensorflow as tf
    fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

    def call(x):
        out = fn(tf.convert_to_tensor(np.asarray(x, dtype="float32")))
        return [o.numpy() for o in out] if isinstance(out, (list, tuple)) else out.numpy()
    return call


class _FastNet:
    def __init__(self, model):
        self._call = tf_fast(model)

    def predict(self, x, *args, **kwargs):
        return self._call(x)


# --------------------------------------------------------------- detector
class Detector:
    """MTCNN face detector = the 'is this a face?' classifier. Every detection carries a
    confidence score; boxes below --min-conf (or smaller than --min-face pixels) are rejected."""

    def __init__(self, min_conf=0.95, det_width=480, min_face=40):
        from mtcnn.mtcnn import MTCNN
        self.min_conf, self.det_width, self.min_face = min_conf, det_width, min_face
        self.mtcnn = MTCNN(min_face_size=max(20, int(min_face * 0.75)))
        try:
            self.mtcnn._pnet = _FastNet(self.mtcnn._pnet)
            self.mtcnn._rnet = _FastNet(self.mtcnn._rnet)
            self.mtcnn._onet = _FastNet(self.mtcnn._onet)
        except Exception as e:                     # fall back to the slow, original path
            print(f"(note: detector speed-up unavailable: {e})")

    def warmup(self, shape):
        self.detect(np.zeros(shape, dtype=np.uint8))

    def detect(self, rgb):
        H, W = rgb.shape[:2]
        s = min(1.0, self.det_width / float(W))
        small = cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))),
                           interpolation=cv2.INTER_AREA) if s < 1.0 else rgb
        faces = []
        for d in self.mtcnn.detect_faces(small):
            if d["confidence"] < self.min_conf:
                continue
            x, y, w, h = [v / s for v in d["box"]]
            x, y = max(x, 0.0), max(y, 0.0)
            w, h = min(w, W - x), min(h, H - y)
            if min(w, h) < self.min_face:
                continue
            faces.append(Face((int(x), int(y), int(w), int(h)), float(d["confidence"])))
        return faces


# --------------------------------------------------------------- backends
class FaceNetBackend:
    """Pretrained FaceNet: 160x160 crop -> 512-d embedding."""
    label = "FaceNet"
    gallery_path = "gallery.npz"

    def __init__(self, threshold):
        from keras_facenet import FaceNet, utils
        self.fn = FaceNet()
        self._crop = utils.cropBox
        self.size = int(self.fn.metadata["image_size"])
        self.margin = int(0.1 * self.size)
        self._call = tf_fast(self.fn.model)
        self.threshold = threshold

    def embed(self, rgb, faces):
        if not faces:
            return []
        crops = [self._crop(rgb, {"box": list(f.box)}, self.margin) for f in faces]
        X = np.float32([self.fn._normalize(cv2.resize(c, (self.size, self.size))) for c in crops])
        return self._call(X)


class CustomBackend:
    """Your Colab-trained MobileNetV2+ArcFace model: 112x112 crop -> 128-d embedding."""
    label = "Custom"
    gallery_path = "gallery_custom.npz"

    def __init__(self, model_path, threshold, pad=0.2):
        if not os.path.exists(model_path):
            raise SystemExit(f"Model file '{model_path}' not found. Copy embedder.keras "
                             f"from Colab into this folder.")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        self.size = int(model.input_shape[1])
        self._call = tf_fast(model)
        self.threshold = threshold
        self.pad = pad      # extra context around the box; LFW crops include more than MTCNN boxes

    def embed(self, rgb, faces):
        if not faces:
            return []
        H, W = rgb.shape[:2]
        crops = []
        for f in faces:
            x, y, w, h = f.box
            p = int(self.pad * max(w, h))
            c = rgb[max(y - p, 0):min(y + h + p, H), max(x - p, 0):min(x + w + p, W)]
            crops.append(cv2.resize(c, (self.size, self.size)).astype("float32"))
        return self._call(np.stack(crops))        # the model rescales 0-255 internally


def analyze(rgb, detector, backends, galleries):
    """Detect once, then embed + match with every backend. Returns plain data."""
    t0 = time.perf_counter()
    faces = detector.detect(rgb)
    det_ms = (time.perf_counter() - t0) * 1000.0
    per = []
    for b, g in zip(backends, galleries):
        t1 = time.perf_counter()
        embs = b.embed(rgb, faces)
        results = [(f.box, *match(e, g, b.threshold)) for f, e in zip(faces, embs)] if g else []
        per.append({"results": results, "embs": embs,
                    "ms": det_ms + (time.perf_counter() - t1) * 1000.0})
    return {"faces": faces, "det_ms": det_ms, "per": per}


class Worker:
    """Runs analyze() on the newest frame in a background thread, so the video never waits."""

    def __init__(self, fn):
        self.fn, self._lock = fn, threading.Lock()
        self._job = self._res = self.error = None
        self._seq, self._alive = 0, True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, rgb):
        with self._lock:
            self._job = (rgb, time.time())

    def latest(self):
        if self.error:
            raise self.error
        with self._lock:
            return self._res, self._seq

    def _loop(self):
        while self._alive:
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                time.sleep(0.003)
                continue
            try:
                r = self.fn(job[0])
                r["t"] = job[1]
            except Exception as e:
                self.error, self._alive = e, False
                return
            with self._lock:
                self._res, self._seq = r, self._seq + 1

    def close(self):
        self._alive = False


class Stats:
    """Per-model counters for the live comparison (used with --truth NAME)."""

    def __init__(self, truth):
        self.truth = safe(truth) if truth else None
        self.correct = self.wrong = self.unknown = 0
        self.ms = []

    def add(self, results, ms):
        self.ms.append(ms)
        if self.truth and len(results) == 1:
            name = results[0][1]
            if name == self.truth:
                self.correct += 1
            elif name == "Unknown":
                self.unknown += 1
            else:
                self.wrong += 1

    def summary(self, label):
        ms = self.ms[3:] or self.ms          # drop warm-up calls
        line = f"{label:8s} avg {np.mean(ms):6.0f} ms per analysed frame" if ms else f"{label:8s} no data"
        if self.truth:
            n = self.correct + self.wrong + self.unknown
            if n:
                line += (f" | single-face frames {n}: correct {self.correct / n:.0%}, "
                         f"unknown {self.unknown / n:.0%}, wrong person {self.wrong / n:.0%}")
        return line


# ---------------------------------------------------------------- camera/UI
def open_cam(index, width):
    cap = None
    if sys.platform.startswith("win"):          # DirectShow opens faster and lags less than the default
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam (index {index}). Close other apps using it, "
                         f"or try --cam 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 3 / 4))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always show the newest frame
    return cap


def put(img, text, org, scale=0.6, color=WHITE, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_boxes(view, results, label, ok_color, bad_color, below=False):
    """results = [(box, name, sim)] in camera coordinates; view is the mirrored image."""
    W = view.shape[1]
    for (x, y, w, h), name, sim in results:
        x = W - x - w                              # mirror
        color = ok_color if name != "Unknown" else bad_color
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        ty = y + h + 24 if below else max(y - 8, 18)
        put(view, f"{label}: {name} {sim:.2f}", (x, ty), 0.6, color)


def need_gallery(backend, model_arg):
    g = load_gallery(backend.gallery_path)
    if not g:
        flag = f" --model {model_arg}" if model_arg and backend.label == "Custom" else ""
        raise SystemExit(f"{backend.label} gallery ({backend.gallery_path}) is empty. "
                         f"Run: python demo.py enroll --name YOURNAME{flag}")
    return g


def start_camera(detector, a, backends=()):
    cap = open_cam(a.cam, a.width)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Webcam opened but returned no frames.")
    print("Warming up the models (a few seconds the first time)...")
    detector.warmup(frame.shape)
    H, W = frame.shape[:2]
    dummy = Face((W // 4, H // 4, W // 2, H // 2), 1.0)
    for b in backends:                       # compile the embedding model before the first real frame
        b.embed(np.zeros(frame.shape, dtype=np.uint8), [dummy])
    return cap


# ------------------------------------------------------------------ enrol
def enroll_webcam(detector, backend, name, a):
    cap = start_camera(detector, a, [backend])
    worker = Worker(lambda rgb: analyze(rgb, detector, [backend], [None]))
    per_pose = max(1, math.ceil(a.samples / len(POSES)))
    need = per_pose * len(POSES)
    embs, pose, got, last_t = [], 0, 0, 0.0
    pose_start = time.time()
    done = False
    print("Guided enrolment: follow the on-screen instructions. q = finish early."
          if not a.manual else "Manual enrolment: press SPACE for each sample, q = finish.")
    while not done:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        worker.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res, _ = worker.latest()
        faces = res["faces"] if res else []
        single = res is not None and len(faces) == 1
        view = cv2.flip(frame, 1)
        W = view.shape[1]
        for f in faces:
            x, y, w, h = f.box
            cv2.rectangle(view, (W - x - w, y), (W - x, y + h), GREEN if single else RED, 2)

        take = False
        if a.manual:
            put(view, f"Press SPACE to take a sample ({len(embs)} so far). q = finish", (10, 28))
        else:
            ready_end = pose_start + a.ready
            if now < ready_end:
                put(view, POSES[pose], (10, 28), 0.7, (0, 220, 255))
                put(view, f"Get ready... {ready_end - now:.0f}", (10, 58), 0.7)
            else:
                put(view, POSES[pose], (10, 28), 0.7, GREEN)
                put(view, f"Hold still: {got}/{per_pose}", (10, 58), 0.7)
                take = (single and res["t"] >= ready_end and res["t"] - last_t >= a.interval)
        if not single:
            put(view, "No face found: face the camera / add light" if not faces
                else "One person at a time, please", (10, view.shape[0] - 40), 0.6, RED)
        bar_w = int((view.shape[1] - 20) * min(len(embs) / float(need), 1.0)) if not a.manual else 0
        cv2.rectangle(view, (10, view.shape[0] - 20), (view.shape[1] - 10, view.shape[0] - 8), (60, 60, 60), -1)
        cv2.rectangle(view, (10, view.shape[0] - 20), (10 + bar_w, view.shape[0] - 8), GREEN, -1)
        cv2.imshow(f"enroll {name}", view)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if a.manual and key == 32 and single and res["t"] > last_t:
            take = True
        if key == ord("n") and not a.manual:                  # skip this pose
            pose, got, pose_start = pose + 1, 0, time.time()
            done = pose >= len(POSES)
            continue
        if take:
            embs.append(res["per"][0]["embs"][0])
            last_t = res["t"]
            got += 1
            if not a.manual and got >= per_pose:
                pose, got, pose_start = pose + 1, 0, time.time()
                done = pose >= len(POSES)
    worker.close()
    cap.release()
    cv2.destroyAllWindows()
    if embs:
        add_to_gallery(backend.gallery_path, name, embs)
    else:
        print("No samples were taken.")


def image_files(path):
    if os.path.isdir(path):
        files = []
        for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
            files += glob.glob(os.path.join(path, f"*.{ext}")) + glob.glob(os.path.join(path, f"*.{ext.upper()}"))
        return sorted(set(files))
    return [path]


def enroll_folder(detector, backend, root):
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for f in image_files(pdir):
            img = cv2.imread(f)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect(rgb)
            if faces:
                best = max(faces, key=lambda t: t.box[2] * t.box[3])
                embs.append(backend.embed(rgb, [best])[0])
        if embs:
            add_to_gallery(backend.gallery_path, person, embs)


# --------------------------------------------------------------- live modes
def live(detector, backends, galleries, a):
    cap = start_camera(detector, a, backends)
    worker = Worker(lambda rgb: analyze(rgb, detector, backends, galleries))
    stats = [Stats(a.truth) for _ in backends]
    two = len(backends) == 2
    last_seq, t_prev, rate = -1, time.time(), 0.0
    title = "compare: FaceNet (top) vs Custom (bottom), q to quit" if two else "face recognition (q to quit)"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        worker.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res, seq = worker.latest()
        view = cv2.flip(frame, 1)
        if res:
            if seq != last_seq:
                last_seq = seq
                now = time.time()
                rate = 0.8 * rate + 0.2 / max(now - t_prev, 1e-6) if rate else 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                for s, pr in zip(stats, res["per"]):
                    s.add(pr["results"], pr["ms"])
            for k, (b, pr) in enumerate(zip(backends, res["per"])):
                draw_boxes(view, pr["results"], b.label,
                           AMBER if k == 1 else GREEN, RED, below=(k == 1))
            if not res["faces"]:
                put(view, "No face detected", (10, 55), 0.7, RED)
            thr = "  ".join(f"{b.label} thr={b.threshold}" for b in backends)
            put(view, f"{thr}   labels update ~{rate:.1f}/s", (10, 25), 0.55)
        cv2.imshow(title, view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    worker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Live summary ===")
    for b, s in zip(backends, stats):
        print(s.summary(b.label))
    if not a.truth:
        print("(Add --truth YOURNAME to also count correct / unknown / wrong-person frames.)")


def facecheck_live(detector, a):
    cap = start_camera(detector, a)
    worker = Worker(lambda rgb: analyze(rgb, detector, [], []))
    last_seq, with_face, total = -1, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        worker.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res, seq = worker.latest()
        view = cv2.flip(frame, 1)
        W = view.shape[1]
        if res:
            if seq != last_seq:
                last_seq = seq
                total += 1
                with_face += 1 if res["faces"] else 0
            for f in res["faces"]:
                x, y, w, h = f.box
                cv2.rectangle(view, (W - x - w, y), (W - x, y + h), GREEN, 2)
                put(view, f"FACE {f.conf:.2f}", (W - x - w, max(y - 8, 18)), 0.6, GREEN)
            n = len(res["faces"])
            put(view, f"FACE DETECTED ({n})" if n else "NO FACE", (10, 30), 0.9, GREEN if n else RED)
        cv2.imshow("face / no face (q to quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    worker.close()
    cap.release()
    cv2.destroyAllWindows()
    if total:
        print(f"Analysed {total} frames: face present in {with_face} ({with_face / total:.0%}).")


def facecheck_images(detector, path, out_dir):
    files = image_files(path)
    if not files:
        raise SystemExit(f"No images found at '{path}'.")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n_face = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"{os.path.basename(f)}: could not read as an image")
            continue
        faces = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        n_face += 1 if faces else 0
        verdict = (f"FACE x{len(faces)} (confidence {', '.join(f'{x.conf:.3f}' for x in faces)})"
                   if faces else "NO FACE")
        print(f"{os.path.basename(f)}: {verdict}")
        if out_dir:
            for x in faces:
                bx, by, bw, bh = x.box
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), GREEN, 2)
                put(img, f"FACE {x.conf:.2f}", (bx, max(by - 8, 18)), 0.6, GREEN)
            if not faces:
                put(img, "NO FACE", (10, 30), 0.9, RED)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), img)
    print(f"\n{n_face} of {len(files)} images contain a face.")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["enroll", "run", "compare", "facecheck", "list"])
    ap.add_argument("--model", help="path to embedder.keras to use your own model (omit for FaceNet;\n"
                                    "in compare mode defaults to embedder.keras)")
    ap.add_argument("--name")
    ap.add_argument("--images", help="image file or folder (enroll: faces/<person>/*.jpg; facecheck: any)")
    ap.add_argument("--out", help="facecheck --images: folder for annotated copies")
    ap.add_argument("--samples", type=int, default=18, help="enrol: total samples (spread over 6 poses)")
    ap.add_argument("--manual", action="store_true", help="enrol: press SPACE for each sample")
    ap.add_argument("--ready", type=float, default=3.0, help="enrol: seconds to get into each pose")
    ap.add_argument("--interval", type=float, default=0.7, help="enrol: min seconds between samples")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--thr-custom", type=float, default=None)
    ap.add_argument("--truth", help="your enrolled name; run/compare count correct/unknown/wrong")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=640, help="camera capture width")
    ap.add_argument("--det-width", type=int, default=480, help="detector input width (smaller = faster)")
    ap.add_argument("--min-face", type=int, default=40, help="ignore faces smaller than this many pixels")
    ap.add_argument("--min-conf", type=float, default=0.95, help="detector confidence needed to call it a face")
    ap.add_argument("--every", type=int, help=argparse.SUPPRESS)       # old option, now unused
    a = ap.parse_args()

    if a.mode == "list":
        for p in ("gallery.npz", "gallery_custom.npz"):
            g = load_gallery(p)
            print(p, {k: len(v) for k, v in g.items()} or "empty / missing")
        return

    quiet_keras()
    still_images = bool(a.images) and a.mode in ("enroll", "facecheck")
    detector = Detector(a.min_conf, max(a.det_width, 960) if still_images else a.det_width, a.min_face)

    if a.mode == "facecheck":
        if a.images:
            facecheck_images(detector, a.images, a.out)
        else:
            facecheck_live(detector, a)
        return

    thr_f = a.threshold if a.threshold is not None else DEFAULT_THR

    if a.mode == "compare":
        custom_path = a.model or "embedder.keras"
        thr_c = a.thr_custom if a.thr_custom is not None else DEFAULT_THR
        backends = [FaceNetBackend(thr_f), CustomBackend(custom_path, thr_c)]
        galleries = [need_gallery(backends[0], None), need_gallery(backends[1], custom_path)]
        live(detector, backends, galleries, a)
        return

    if a.model:
        thr = a.threshold if a.threshold is not None else (
            a.thr_custom if a.thr_custom is not None else DEFAULT_THR)
        backend = CustomBackend(a.model, thr)
    else:
        backend = FaceNetBackend(thr_f)

    if a.mode == "enroll":
        if a.images:
            enroll_folder(detector, backend, a.images)
        elif a.name:
            enroll_webcam(detector, backend, a.name, a)
        else:
            raise SystemExit("Give --name (webcam) or --images (folder).")
    else:
        live(detector, [backend], [need_gallery(backend, a.model)], a)


if __name__ == "__main__":
    main()
