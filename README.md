1. **Live demo (`demo.py`)**: webcam, detection (MTCNN) + pretrained FaceNet embeddings + cosine matching. Works within the first hour.
2. **Personal model (`train_arcface.py` + `evaluate.py`)**: MobileNetV2 trained with ArcFace loss on LFW, evaluated on identities it has never seen, compared against pretrained FaceNet with an ROC curve.

## Setup
1) Open a terminal and set the current directory to the directory with your files.
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## NOTE:
## If your venv is not loading in PowerShell, run the following command first
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
```
Use Google Colab (free T4 GPU) for training if you have no local GPU. Upload the `.py` files, run `!pip install mtcnn keras-facenet`, then run the training script. The webcam demo must run on your own machine.


## Pipeline 

detect face (MTCNN) -> crop/align -> CNN -> embedding vector -> cosine similarity to enrolled people -> accept if above threshold, else "Unknown".

## Description

- The test set uses **identities the model never saw in training**. That is the real face-verification setting. Testing on seen people would overstate performance.
- Pretrained FaceNet will very likely beat your model. It was trained on millions of images, yours on a few thousand. Say so. The point is that you built and evaluated the training pipeline (ArcFace margin loss, transfer learning, ROC/EER analysis) and the full working system.
- Report EER and TAR@1% FAR, not just accuracy.
- Known limitations to mention: lighting and pose sensitivity, dataset bias (LFW is mostly public figures, skewed demographics), no liveness detection (a photo can fool it), consent and privacy when enrolling people.

## Refinement ideas for later

- Train on a larger dataset (CASIA-WebFace, VGGFace2) and use it in `demo.py` in place of FaceNet.
- Add face alignment using MTCNN keypoints.
- Try a margin sweep (m = 0.3 / 0.5) and report it.
- Add liveness or blink detection.
- Convert the model to TFLite for a speed comparison.

## Switching to your own model and comparing (after Colab finishes)

Put `embedder.keras` (from Colab) in the same folder as `demo.py`.

```bash
python demo.py enroll --name Aaryan --model embedder.keras
python demo.py run --model embedder.keras --threshold 0.5
python demo.py compare --truth Aaryan --thr-custom 0.5
python demo.py list        # shows both galleries
```
## What the code does
# 1. enrol yourself with YOUR model (separate gallery file: gallery_custom.npz)
# 2. run live with your model (tune --threshold using the value evaluate.py printed)
# 3. side-by-side: FaceNet label above the box, your model below
## NOTE: When you want to exit the webcam view, press "Q"

`compare` needs both galleries (`gallery.npz` for FaceNet, `gallery_custom.npz` for yours).
When you press q it prints average ms per frame and, for frames with exactly one face,
the share that were correct / Unknown / wrong person. Stand alone in front of the camera
for a minute, then try with a second person to see wrong-person errors.

Threshold: `evaluate.py` prints suggested values for both models. Use the "best-accuracy" one first;
raise it if strangers are accepted, lower it if you get "Unknown" too often.

## Enrolment, speed and the face / no-face check

**Guided enrolment** (default): the window tells you which way to turn, counts down 3 seconds so you can
get into position, then takes 3 samples per pose (6 poses, 18 samples). The view is mirrored like a mirror.
```bash
python demo.py enroll --name Aaryan                 # guided
python demo.py enroll --name Aaryan --ready 5       # more time to get into each pose
python demo.py enroll --name Aaryan --manual        # press SPACE yourself for every sample
```
Keys: q = finish, n = skip the current pose. New samples are added to anyone already enrolled under that
name; delete `gallery.npz` (or `gallery_custom.npz`) to start fresh.

**If it is still slow:** `python demo.py run --det-width 320 --min-face 60` (smaller detector input,
ignore small faces). `--cam 1` selects another camera.

**Is it a face at all?** (no enrolment needed)
```bash
python demo.py facecheck                            # live: FACE DETECTED / NO FACE
python demo.py facecheck --images photo.jpg         # one image
python demo.py facecheck --images some_folder --out checked   # many images + annotated copies
```
`--min-conf` (default 0.95) is how sure the detector must be before it calls something a face.
