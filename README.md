# Real-Time Face Recognition System

A Computer Vision-based face recognition system developed using **Python, OpenCV, and Streamlit**. The project implements and compares three classical face recognition methods — **LBPH, Eigenfaces, and Fisherfaces** — and provides both image-upload and real-time browser webcam recognition.

**Computer Vision Assignment**
**Developed by Shirsam Bhattacharjee & Roni Ghosh**

---

## Features

* Face detection using Haar Cascade
* Face preprocessing and normalization
* Face recognition using:

  * LBPH
  * Eigenfaces
  * Fisherfaces
* Unknown-face detection using method-specific thresholds
* Image Upload Recognition
* Real-Time Browser Webcam Recognition
* Recognition distance display
* Known / Unknown status indication
* Method comparison
* Evaluation metrics
* Confusion matrices
* Distance distribution analysis
* Streamlit-based web interface
* Light/Dark-friendly interface
* Privacy-aware dataset handling

---

## Project Architecture

```text
facerec_system/
│
├── app.py
├── main.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
├── haarcascade_frontalface_default.xml
│
├── data/
│   └── haarcascade_frontalface_default.xml
│
├── models/
│   ├── lbph_model.yml
│   ├── lbph_model_meta.json
│   ├── eigenfaces_model.yml
│   ├── eigenfaces_model_meta.json
│   ├── fisherfaces_model.yml
│   ├── fisherfaces_model_meta.json
│   ├── label_map.json
│   ├── thresholds.json
│   └── .gitkeep
│
├── src/
│   ├── utils.py
│   ├── calibrate.py
│   ├── diagnose.py
│   ├── evaluate.py
│   ├── recognize_realtime.py
│   ├── train_lbph.py
│   ├── train_eigenfaces.py
│   └── train_fisherfaces.py
│
├── results/
│   ├── comparison.png
│   ├── confusion_matrix_lbph.png
│   ├── confusion_matrix_eigenfaces.png
│   ├── confusion_matrix_fisherfaces.png
│   ├── distance_hist_lbph.png
│   ├── distance_hist_eigen.png
│   ├── distance_hist_fisher.png
│   └── metrics.csv
│
└── report/
    └── analysis_report_template.md
```

> **Note:** The original face-image dataset is intentionally excluded from the GitHub repository for privacy. It is ignored through `.gitignore`.

---

## System Pipeline

```text
Input Image / Webcam
        │
        ▼
Face Detection
(Haar Cascade)
        │
        ▼
Preprocessing
Grayscale → Crop → Resize → Histogram Equalization
        │
        ▼
Face Recognition
 ┌────────┼─────────┐
 ▼        ▼         ▼
LBPH   Eigenfaces  Fisherfaces
 └────────┼─────────┘
          ▼
Distance + Threshold
          │
     ┌────┴────┐
     ▼         ▼
   Known     Unknown
```

---

## Recognition Methods

### 1. LBPH — Local Binary Patterns Histograms

LBPH describes local texture patterns around facial regions and compares the resulting histograms.

**Advantages:**

* Relatively fast
* Works well with small datasets
* Suitable for real-time recognition
* More tolerant of moderate lighting variations

---

### 2. Eigenfaces

Eigenfaces uses **Principal Component Analysis (PCA)** to represent faces in a lower-dimensional feature space.

**Advantages:**

* Reduces dimensionality
* Classical and computationally understandable
* Useful for comparing PCA-based face recognition

---

### 3. Fisherfaces

Fisherfaces uses **Linear Discriminant Analysis (LDA)** to maximize separation between different classes.

**Advantages:**

* Focuses on class discrimination
* Can perform well when differences between people need to be emphasized
* Provides a useful comparison with PCA-based Eigenfaces

---

## Face Processing Pipeline

The same preprocessing pipeline is used throughout the recognition system:

1. Detect the face using Haar Cascade.
2. Convert the image to grayscale.
3. Extract the detected face region.
4. Resize the face to **200 × 200 pixels**.
5. Apply histogram equalization.
6. Pass the processed face to the selected recognition model.

---

## Unknown Face Detection

The system does not automatically accept the closest predicted person.

For every recognition method, the predicted distance is compared against a method-specific threshold.

```text
Distance ≤ Threshold
        │
        ▼
      Known

Distance > Threshold
        │
        ▼
     Unknown
```

Threshold information is stored in:

```text
models/thresholds.json
```

The label-to-name mapping is stored in:

```text
models/label_map.json
```

---

## Web Application

The Streamlit application provides two main recognition modes.

### Upload Image

The user can upload an image and select a recognition method.

The system detects faces and displays:

* Face bounding box
* Predicted name
* Recognition distance
* Known / Unknown result

### Real-Time Webcam

The browser webcam feature uses `streamlit-webrtc`.

The webcam page provides:

* Browser camera permission
* Real-time face detection
* Real-time recognition
* Bounding boxes
* Recognition name
* Recognition distance
* Known / Unknown indication
* Start and Stop camera controls

The application processes the browser camera stream rather than attempting to access the deployment server's physical webcam.

---

## Evaluation

The project includes an evaluation module for comparing the recognition methods.

Evaluation results include:

* Accuracy
* Precision
* Recall
* Confusion Matrix
* Recognition time
* Distance distributions

Generated results are stored inside:

```text
results/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shirsambhattacharjee/facerec_system.git
cd facerec_system
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

---

## Command-Line Recognition

The project also contains the original command-line real-time recognition system.

Run:

```bash
python main.py
```

---

## Important Files

| File / Folder               | Purpose                                          |
| --------------------------- | ------------------------------------------------ |
| `app.py`                    | Streamlit web application                        |
| `main.py`                   | Command-line application                         |
| `src/utils.py`              | Shared face processing and recognition utilities |
| `src/train_lbph.py`         | LBPH training                                    |
| `src/train_eigenfaces.py`   | Eigenfaces training                              |
| `src/train_fisherfaces.py`  | Fisherfaces training                             |
| `src/evaluate.py`           | Model evaluation                                 |
| `src/calibrate.py`          | Threshold calibration                            |
| `src/diagnose.py`           | Recognition diagnostics                          |
| `src/recognize_realtime.py` | Real-time recognition pipeline                   |
| `models/`                   | Trained recognition models and metadata          |
| `results/`                  | Evaluation and analysis results                  |
| `report/`                   | Project report material                          |
| `data/`                     | Face detection resources                         |

---

## Privacy

The project contains facial-image training data.

For privacy protection:

* The original `dataset/` directory is excluded from Git using `.gitignore`.
* Personal training photographs are not included in the public project files.
* The GitHub repository is intended to remain **Private**.
* Trained model files are included because they are required by the deployed application.

---

## Technologies Used

* **Python**
* **OpenCV**
* **OpenCV-Contrib**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Streamlit**
* **Streamlit-WebRTC**
* **PyAV**

---

## Project Objective

The primary objective of this project is to study, implement, and compare classical face-recognition techniques and demonstrate their practical use through both image-based and real-time webcam recognition.

The project compares **LBPH, Eigenfaces, and Fisherfaces** using a common preprocessing and evaluation pipeline.

---

## Authors

- **Shirsam Bhattacharjee**
- **Roni Ghosh**

### Computer Vision Assignment

**Real-Time Face Recognition System**

Built with Python + OpenCV + Streamlit.
