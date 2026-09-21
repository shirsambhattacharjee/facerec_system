# Face Recognition Methods: Comparative Analysis Report

*(Fill in the bracketed sections after running `src/evaluate.py`. This
template is structured to directly satisfy the assignment's research
requirement: "Compare face-recognition methods and analyze recognition
accuracy.")*

---

## 1. Introduction

State the objective: to implement and empirically compare three classical
face-recognition algorithms — **LBPH**, **Eigenfaces**, and **Fisherfaces**
— on a self-collected webcam dataset, evaluating recognition accuracy,
precision, recall, and recognition speed.

**Dataset summary:**
- Number of people enrolled: `[N]`
- Total images collected: `[N]`
- Train / test split: `[N] train images / [N] test images (≈80/20)`
- Capture conditions: `[e.g. indoor lighting, webcam resolution, etc.]`

---

## 2. Methodology

### 2.1 Face Detection & Preprocessing
Faces are detected with an OpenCV Haar Cascade classifier
(`haarcascade_frontalface_default.xml`), cropped, resized to a fixed
200×200 grayscale image, and histogram-equalized to normalize lighting
before being passed to any recognizer. This preprocessing pipeline is
identical for all three methods, ensuring a fair comparison.

### 2.2 Algorithms Compared

**LBPH (Local Binary Patterns Histograms).** Encodes texture around each
pixel as a binary pattern relative to its neighbors, then compares
histograms of these patterns between the probe and gallery images.
Config used: radius=1, neighbors=8, grid=8×8.

**Eigenfaces (PCA).** Projects all training faces into a lower-dimensional
subspace defined by the directions of maximum variance (principal
components) and classifies new faces by nearest neighbor in that space.

**Fisherfaces (LDA).** Extends Eigenfaces by additionally maximizing the
separation *between* different people's face clusters relative to the
spread *within* each person's own images, which should make it more
discriminative when enough samples per person are available.

### 2.3 Evaluation Metrics
- **Accuracy** — fraction of correctly identified test faces.
- **Precision (macro-averaged)** — of all faces predicted as person X,
  what fraction actually were person X, averaged equally over all people.
- **Recall (macro-averaged)** — of all actual instances of person X, what
  fraction were correctly identified, averaged equally over all people.
- **Average recognition time (ms)** — mean wall-clock time per prediction,
  measuring practical real-time feasibility.
- **Confusion matrix** — per-method breakdown of which people get
  misidentified as whom.

---

## 3. Results

*(Paste values from `results/metrics.csv` here.)*

| Method | Accuracy | Precision | Recall | Avg. Recognition Time (ms) |
|---|---|---|---|---|
| LBPH | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Eigenfaces | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Fisherfaces | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Comparison chart:** see `results/comparison.png`

**Confusion matrices:**
- LBPH: `results/confusion_matrix_lbph.png`
- Eigenfaces: `results/confusion_matrix_eigenfaces.png`
- Fisherfaces: `results/confusion_matrix_fisherfaces.png`

---

## 4. Analysis & Discussion

*(Answer these based on YOUR actual results — numbers will vary by dataset.)*

1. **Which method achieved the highest accuracy, and why might that be,
   given how each method works?**
   `[ ]`

2. **Which method was fastest / slowest at recognition time, and does that
   matter for a real-time application?**
   `[ ]`

3. **Look at the confusion matrices — are any two people consistently
   confused with each other by all methods, or only by one method? What
   does that suggest about those methods' sensitivity to specific features
   (e.g. lighting vs. facial structure)?**
   `[ ]`

4. **How did dataset size / number of images per person likely affect each
   method's performance (especially Fisherfaces, which is more
   sample-sensitive)?**
   `[ ]`

5. **If you were deploying this in a real product (e.g. an attendance
   system), which method would you choose and why — considering the
   accuracy/speed trade-off you observed?**
   `[ ]`

---

## 5. Limitations

- Small, self-collected dataset — results may not generalize to larger,
  more diverse populations.
- Haar Cascade face detection can miss faces at extreme angles or poor
  lighting, which indirectly affects all downstream recognition results.
- All three methods are classical (non-deep-learning) approaches; a
  CNN-based embedding method (e.g. FaceNet) would likely outperform all
  three but was intentionally excluded per assignment scope.

---

## 6. Conclusion

*(2-4 sentences summarizing which method performed best overall in this
experiment, and the key trade-off between accuracy and speed observed.)*
`[ ]`
