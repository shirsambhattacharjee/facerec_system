# Real-Time Face Recognition System

A complete, beginner-friendly **Computer Vision** project that performs real-time
face recognition using a webcam, and compares three classical face-recognition
algorithms: **LBPH**, **Eigenfaces**, and **Fisherfaces**.

Built entirely with **OpenCV, NumPy, Pandas, Matplotlib, and scikit-learn** —
no TensorFlow/PyTorch or deep learning frameworks required.

---

## 1. Project Goal

This project satisfies the research component:
> "Compare face-recognition methods and analyze recognition accuracy."

It does this by:
1. Collecting your own face dataset via webcam.
2. Training three different classical face recognizers on the same data.
3. Running real-time recognition with any of the three.
4. Evaluating all three on a held-out test set and producing accuracy,
   precision, recall, recognition-time, and confusion-matrix comparisons.

---

## 2. Folder Structure

```
facerec_system/
├── dataset/
│   ├── train/            # training face images, one sub-folder per person
│   └── test/              # held-out test face images (auto-split during collection)
├── models/                 # trained model files (.yml) + label_map.json
├── src/
│   ├── utils.py             # shared helpers (face detection, preprocessing, I/O)
│   ├── collect_faces.py      # Step 1: build dataset from webcam
│   ├── train_lbph.py          # Step 2a: train LBPH
│   ├── train_eigenfaces.py     # Step 2b: train Eigenfaces
│   ├── train_fisherfaces.py     # Step 2c: train Fisherfaces
│   ├── recognize_realtime.py     # Step 3: live webcam recognition
│   └── evaluate.py                # Step 4: compare all 3 methods
├── results/                  # generated charts, confusion matrices, metrics.csv
├── report/
│   └── analysis_report_template.md  # research write-up template
├── main.py                  # menu-driven entry point (run this!)
├── requirements.txt
└── README.md
```

---

## 3. How the Three Methods Differ (Quick Theory)

| Method | Core Idea | Strengths | Weaknesses |
|---|---|---|---|
| **LBPH** (Local Binary Patterns Histograms) | Encodes local texture patterns around each pixel into histograms, compares histograms between images. | Robust to lighting changes, works well with small/uneven datasets, no strict size requirement. | Can be slower on very large datasets; sensitive to noise at very small radius. |
| **Eigenfaces** (PCA) | Projects face images into a lower-dimensional space (principal components) that captures the most *overall variance* across all training faces. | Simple, fast to train, good baseline. | Variance it captures can come from lighting/pose, not just identity — less discriminative between people. |
| **Fisherfaces** (LDA) | Similar to Eigenfaces but maximizes the ratio of *between-class* variance to *within-class* variance, specifically optimizing for telling classes (people) apart. | Usually best at discriminating between specific people if enough samples/person exist. | Needs at least 2 people and several images each; can overfit with very few samples per person. |

The `evaluate.py` script quantifies these trade-offs on your own dataset rather
than relying on theory alone — that's the "research" part of the assignment.

---

## 4. Setup Instructions (Windows)

### Step 4.1 — Install Python
Make sure you have **Python 3.9–3.11** installed (from python.org). During
installation, check "Add Python to PATH".

### Step 4.2 — Open a terminal in the project folder
Extract this project, then open **Command Prompt** or **PowerShell** and `cd`
into the extracted `facerec_system` folder:
```powershell
cd path\to\facerec_system
```

### Step 4.3 — Create and activate a virtual environment (recommended)
```powershell
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` appear at the start of your prompt.

### Step 4.4 — Install dependencies
```powershell
pip install -r requirements.txt
```

> **Important:** This project uses `opencv-contrib-python` (not plain
> `opencv-python`), because the `cv2.face` module (LBPH/Eigenfaces/Fisherfaces)
> only ships in the "contrib" build. If you get an error like
> `AttributeError: module 'cv2' has no attribute 'face'`, run:
> ```powershell
> pip uninstall opencv-python opencv-contrib-python -y
> pip install opencv-contrib-python
> ```

### Step 4.5 — Allow camera access
Windows may prompt you to allow camera access for Python/Command Prompt the
first time you run a script that uses the webcam — click **Allow**.

---

## 5. How to Run

The easiest way is the menu-driven entry point:
```powershell
python main.py
```
This gives you a numbered menu for every step below. Alternatively, you can
run each script directly.

### 5.1 Collect face data (repeat once per person)
```powershell
python src\collect_faces.py --name "Alice" --samples 60
```
- A webcam window opens; a green box marks the detected face.
- Images are captured automatically every few frames.
- Press **SPACE** to pause/resume, **q** to stop early.
- Images are auto-split: 80% into `dataset/train/Alice/`, 20% into
  `dataset/test/Alice/` (change with `--train-ratio 0.75`, etc.).
- **Repeat this for every person** you want the system to recognize
  (at least 2 people, ideally 3+, for a meaningful comparison).

Tips for a good dataset:
- Capture in the lighting condition you'll test in.
- Slightly vary head angle/expression during capture.
- 50-100 images per person is a reasonable amount for this project.

### 5.2 Train the models
Train each method (or use menu option 5 to train all three at once):
```powershell
python src\train_lbph.py
python src\train_eigenfaces.py
python src\train_fisherfaces.py
```
Each script builds `models\label_map.json` (name ↔ id mapping) and saves its
trained model as a `.yml` file in `models\`.

> Fisherfaces requires **at least 2 people** in `dataset/train` — it will
> raise a clear error if you only have one.

### 5.3 Run real-time recognition
```powershell
python src\recognize_realtime.py --method lbph
python src\recognize_realtime.py --method eigen
python src\recognize_realtime.py --method fisher
```
- Detected faces are boxed green (recognized, with name + confidence score)
  or red (below-confidence match, labeled "Unknown").
- Press **q** to quit.
- If you get too many/few "Unknown" labels, tune the threshold, e.g.:
  ```powershell
  python src\recognize_realtime.py --method lbph --threshold 60
  ```
  (Lower threshold = stricter matching. See the comments in
  `recognize_realtime.py` for typical ranges per method.)

### 5.4 Evaluate and compare all three methods
```powershell
python src\evaluate.py
```
This runs every trained model against every image in `dataset/test/` and
writes to `results/`:
- `metrics.csv` — accuracy, precision, recall, avg recognition time per method
- `confusion_matrix_lbph.png`, `confusion_matrix_eigenfaces.png`,
  `confusion_matrix_fisherfaces.png`
- `comparison.png` — side-by-side bar charts comparing all three methods

Use these outputs to fill in `report/analysis_report_template.md` for your
write-up.

---

## 6. Typical End-to-End Workflow

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python main.py
# -> option 1, collect "Alice"
# -> option 1, collect "Bob"
# -> option 1, collect "Charlie"
# -> option 5, train all models
# -> option 7, evaluate & compare
# -> option 6, watch it recognize you live!
```

---

## 7. Troubleshooting

| Problem | Likely Fix |
|---|---|
| `cv2.face` not found | Install `opencv-contrib-python`, not `opencv-python` (see 4.4). |
| Webcam window doesn't open / black window | Check no other app is using the camera; try `--camera 1` if you have multiple cameras. |
| "No training images found" | Run `collect_faces.py` first for at least one person. |
| Fisherfaces training error | Needs 2+ people in `dataset/train`. |
| Poor accuracy | Collect more images per person, more consistent lighting, or lower the recognition threshold. |
| Low FPS during real-time recognition | Close other camera-using apps; ensure you're using LBPH (fastest) if speed matters most. |

---

## 8. Notes on Design Choices

- **Why Haar Cascades for detection?** They're built into OpenCV (no extra
  download), fast, and good enough for a controlled webcam setup — appropriate
  for a classical-CV assignment that isn't using deep learning.
- **Why grayscale + fixed size + histogram equalization?** Eigenfaces and
  Fisherfaces require every image to be the same size, and equalization
  reduces the impact of lighting differences across capture sessions —
  improving fairness of the method comparison.
- **Why is "confidence" actually a distance?** All three OpenCV
  `cv2.face.*Recognizer` classes return a distance-like score where *lower is
  better* (a closer match), not a probability — this is explained in
  `recognize_realtime.py` and accounted for via per-method thresholds.
