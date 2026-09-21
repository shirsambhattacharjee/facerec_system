import os
import json
from datetime import datetime

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATASET_TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
DATASET_TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
THRESHOLDS_PATH = os.path.join(MODELS_DIR, "thresholds.json")
HAAR_CASCADE_PATH = os.path.join(PROJECT_ROOT, "data", "haarcascade_frontalface_default.xml")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


FACE_SIZE = (200, 200)
MAX_IMAGE_DIM = 1280            
PRECROPPED_MAX_SIDE = 400       
DETECT_PARAMS = dict(scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
PIPELINE_VERSION = 2            
METHODS = ("lbph", "eigen", "fisher")
METHOD_LABELS = {"lbph": "LBPH", "eigen": "Eigenfaces", "fisher": "Fisherfaces"}
LABEL_TO_KEY = {v: k for k, v in METHOD_LABELS.items()}
MODEL_FILES = {
    "lbph": "lbph_model.yml",
    "eigen": "eigenfaces_model.yml",
    "fisher": "fisherfaces_model.yml",
}
RESULT_STEMS = {"lbph": "lbph", "eigen": "eigenfaces", "fisher": "fisherfaces"}

LBPH_PARAMS = dict(radius=1, neighbors=8, grid_x=8, grid_y=8)
EIGEN_COMPONENTS = 60           
DEFAULT_AUGMENT_COPIES = 2      
AUGMENT_SEED = 42
DEFAULT_THRESHOLDS = {"lbph": 70.0, "eigen": 4500.0, "fisher": 400.0}
RECOGNITION_THRESHOLDS = DEFAULT_THRESHOLDS  
THRESHOLDS = DEFAULT_THRESHOLDS               
RECOGNITION_THRESHOLDS = DEFAULT_THRESHOLDS  


def ensure_dirs(*dirs):
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def require_face_module():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV Contrib is required (cv2.face is missing).\n"
            "Fix:  python -m pip uninstall opencv-python opencv-contrib-python -y\n"
            "      python -m pip install opencv-contrib-python"
        )


def decode_image(data):
    """bytes / uint8 array -> BGR image (EXIF orientation applied by OpenCV)."""
    arr = np.frombuffer(data, np.uint8) if isinstance(data, (bytes, bytearray)) else data
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def read_image(path):
    """Unicode-safe imread (BGR). Returns None if unreadable."""
    try:
        return decode_image(np.fromfile(path, dtype=np.uint8))
    except OSError:
        return None


def write_image(path, img):
    ensure_dirs(os.path.dirname(path))
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(path)


def limit_size(img, max_dim=MAX_IMAGE_DIM):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    s = max_dim / m
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def to_gray(img):
    """BGR/gray -> size-limited uint8 gray. Idempotent."""
    img = limit_size(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def read_gray(path):
    img = read_image(path)
    return None if img is None else to_gray(img)


def open_camera(index=0, width=640, height=480, warmup=10):
    """
    Open a webcam reliably. Tries DirectShow, then Media Foundation, then the default
    backend (Windows), requests a fixed resolution, and discards `warmup` frames so
    auto-exposure / auto-white-balance can settle before the first prediction.
    Always returns a VideoCapture; check .isOpened().
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            if width and height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok = False
            for _ in range(max(1, warmup)):
                ok, _frame = cap.read()
            if ok:
                return cap
        cap.release()
    return cv2.VideoCapture()        

_DETECTOR = None


def get_face_detector():
    global _DETECTOR
    if _DETECTOR is None:
        path = HAAR_CASCADE_PATH
        if not os.path.exists(path) and hasattr(cv2, "data"):
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        det = cv2.CascadeClassifier(path)
        if det.empty():
            raise IOError(f"Could not load Haar Cascade from:\n{path}")
        _DETECTOR = det
    return _DETECTOR


def detect_faces(gray, detector=None):
    """All faces as (x, y, w, h), largest first."""
    detector = detector or get_face_detector()
    boxes = detector.detectMultiScale(gray, **DETECT_PARAMS)
    boxes = [tuple(int(v) for v in b) for b in boxes]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def detect_largest_face(gray, detector=None):
    boxes = detect_faces(gray, detector)
    return boxes[0] if boxes else None


def _looks_precropped(gray):
    h, w = gray.shape[:2]
    return max(h, w) <= PRECROPPED_MAX_SIDE and 0.75 <= w / h <= 1.33


def locate_faces(gray, detector=None, allow_precropped=False):
    """
    Returns [((x, y, w, h), mode)], largest first. mode: "detected" | "whole".
    allow_precropped=True is used for dataset images (train/test) and optionally
    Streamlit: a small, squarish image is accepted as a face crop when Haar finds
    nothing or only a small sub-region of it. The webcam always uses False.
    """
    boxes = detect_faces(gray, detector)
    if allow_precropped and _looks_precropped(gray):
        h, w = gray.shape[:2]
        if not boxes or boxes[0][2] * boxes[0][3] < 0.6 * w * h:
            return [((0, 0, w, h), "whole")]
    return [(b, "detected") for b in boxes]

def crop_face(gray, box):
    x, y, w, h = box
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return gray[y0:y1, x0:x1]


def normalize_face(crop):
    """raw gray face crop -> 200x200 equalized uint8 (what every model sees)."""
    if crop is None or crop.size == 0:
        return None
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if crop.shape[1] > FACE_SIZE[0] else cv2.INTER_LINEAR
    face = cv2.resize(crop, FACE_SIZE, interpolation=interp)
    return cv2.equalizeHist(face)


def preprocess_face(gray, box):
    """gray full image + Haar box -> model input. Used by webcam, Streamlit, eval."""
    return normalize_face(crop_face(gray, box))


def preprocess_image(img):
    """Backward-compatible: an image that is ALREADY a face crop -> model input."""
    if img is None:
        return None
    return normalize_face(to_gray(img))


def preprocess_image_file(path, detector=None, allow_precropped=True):
    """image FILE -> model input (largest face), or None. Same pipeline as training."""
    gray = read_gray(path)
    if gray is None:
        return None
    regions = locate_faces(gray, detector or get_face_detector(), allow_precropped)
    return preprocess_face(gray, regions[0][0]) if regions else None


def list_image_files(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS) and os.path.isfile(os.path.join(folder, f))
    )


def list_people(dataset_dir, require_images=True):
    if not os.path.isdir(dataset_dir):
        return []
    people = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    )
    if require_images:
        people = [p for p in people if list_image_files(os.path.join(dataset_dir, p))]
    return people


def build_label_map():
    """Rebuild models/label_map.json from the folders in dataset/train."""
    people = list_people(DATASET_TRAIN_DIR)
    if not people:
        raise RuntimeError(f"No person folders with .jpg/.jpeg/.png images in {DATASET_TRAIN_DIR}")
    label_map = {name: i for i, name in enumerate(people)}
    _write_json(LABEL_MAP_PATH, label_map)
    return label_map


def load_label_map():
    """name -> int id, read from models/label_map.json (never hard-coded)."""
    data = _read_json(LABEL_MAP_PATH)
    if data is None:
        raise RuntimeError("models/label_map.json not found. Train a model first.")
    return {str(k): int(v) for k, v in data.items()}


def id_to_name_map(label_map):
    """name -> id   into   id -> name"""
    return {int(v): k for k, v in label_map.items()}


def load_split(dataset_dir, label_map, allow_unknown=False, verbose=True):
    """
    Load every valid image, find the face with the SAME detector, keep the RAW
    gray crop (normalization happens later, once, in normalize_face).
    Folders not in label_map: skipped, or label -1 ("Unknown") if allow_unknown.
    Returns (records, report).
    """
    detector = get_face_detector()
    records, report = [], {}
    for folder in list_people(dataset_dir, require_images=False):
        if folder in label_map:
            label = label_map[folder]
        elif allow_unknown:
            label = -1
        else:
            print(f"[WARN] '{folder}' in {dataset_dir} is not in the label map - skipped.")
            continue
        paths = list_image_files(os.path.join(dataset_dir, folder))
        st = dict(found=len(paths), detected=0, whole=0, skipped=0, sizes=[], brightness=[])
        for p in paths:
            gray = read_gray(p)
            if gray is None:
                st["skipped"] += 1
                print(f"[WARN] Could not read: {p}")
                continue
            regions = locate_faces(gray, detector, allow_precropped=True)
            crop = crop_face(gray, regions[0][0]) if regions else None
            if crop is None:
                st["skipped"] += 1
                print(f"[WARN] No face found, skipped: {p}")
                continue
            mode = regions[0][1]
            st[mode] += 1
            st["sizes"].append(crop.shape[1])
            st["brightness"].append(float(crop.mean()))
            records.append(dict(label=label, name=folder if label >= 0 else "Unknown",
                                folder=folder, path=p, crop=crop, mode=mode,
                                box=regions[0][0]))
        report[folder] = st
    if verbose:
        print_split_report(report, os.path.basename(dataset_dir))
    return records, report


def print_split_report(report, split_name):
    noun = "training" if split_name == "train" else split_name
    print(f"\nFound {len(report)} people:")
    total = 0
    for name, s in report.items():
        total += s["found"]
        used = s["detected"] + s["whole"]
        size = np.mean(s["sizes"]) if s["sizes"] else 0
        bright = np.mean(s["brightness"]) if s["brightness"] else 0
        print(f"{name}: {s['found']} images  (used {used}: {s['detected']} face-detected, "
              f"{s['whole']} pre-cropped, {s['skipped']} skipped | avg face {size:.0f}px, "
              f"avg brightness {bright:.0f})")
    print(f"\nTotal {noun} images: {total}")


def load_dataset(dataset_dir, label_map):
    """Backward-compatible: (faces, labels) with the unified pipeline, no augmentation."""
    records, _ = load_split(dataset_dir, label_map, verbose=False)
    return [normalize_face(r["crop"]) for r in records], [r["label"] for r in records]


def augment_crop(crop, rng):
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-7, 7), rng.uniform(0.93, 1.07))
    M[0, 2] += rng.uniform(-0.04, 0.04) * w
    M[1, 2] += rng.uniform(-0.04, 0.04) * h
    img = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255.0
    img = np.power(img, rng.uniform(0.6, 1.6))                      # gamma
    ramp = np.linspace(1 - rng.uniform(0, 0.35), 1 + rng.uniform(0, 0.35), w, dtype=np.float32)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    img = img * ramp[None, :] * rng.uniform(0.7, 1.2)               # side light + contrast
    sigma = rng.uniform(0, 1.5)
    if sigma > 0.4:
        img = cv2.GaussianBlur(img, (0, 0), sigma)                  # soft focus / screen blur
    img = img + rng.normal(0, rng.uniform(0, 0.03), img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def build_training_arrays(records, copies=DEFAULT_AUGMENT_COPIES, seed=AUGMENT_SEED):
    """
    Originals are ALWAYS included. With copies>0 every class is topped up with
    augmented samples to (largest_class * (1 + copies)) -> balanced classes.
    """
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r["crop"])
    rng = np.random.default_rng(seed)
    target = max(len(v) for v in by_label.values()) * (1 + copies)
    faces, labels = [], []
    for label, crops in sorted(by_label.items()):
        for c in crops:
            faces.append(normalize_face(c))
            labels.append(label)
        if copies > 0:
            for i in range(target - len(crops)):
                faces.append(normalize_face(augment_crop(crops[i % len(crops)], rng)))
                labels.append(label)
    return faces, np.array(labels, dtype=np.int32)


def create_recognizer(method, n_samples=None):
    require_face_module()
    if method == "lbph":
        return cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
    if method == "eigen":
        comps = EIGEN_COMPONENTS if n_samples is None else max(1, min(EIGEN_COMPONENTS, n_samples - 1))
        return cv2.face.EigenFaceRecognizer_create(num_components=comps)
    if method == "fisher":
        return cv2.face.FisherFaceRecognizer_create()
    raise ValueError(f"Unknown method: {method}")


def model_path(method):
    return os.path.join(MODELS_DIR, MODEL_FILES[method])


def meta_path(method):
    return os.path.splitext(model_path(method))[0] + "_meta.json"


def train_method(method, augment=DEFAULT_AUGMENT_COPIES):
    require_face_module()
    label_map = build_label_map()
    print(f"\n=== Training {METHOD_LABELS[method]} ===")
    print("Label map:", label_map)
    records, _ = load_split(DATASET_TRAIN_DIR, label_map, verbose=True)
    if not records:
        raise RuntimeError("No usable training images.")
    n_classes = len({r["label"] for r in records})
    if method in ("eigen", "fisher") and n_classes < 2:
        raise RuntimeError(f"{METHOD_LABELS[method]} needs at least 2 people.")

    faces, labels = build_training_arrays(records, augment)
    print(f"Training samples used: {len(faces)} "
          f"({len(records)} original + {len(faces) - len(records)} augmented)")
    rec = create_recognizer(method, n_samples=len(faces))
    rec.train(faces, labels)

    ensure_dirs(MODELS_DIR)
    path = model_path(method)
    rec.write(path)
    _write_json(meta_path(method), dict(
        method=method, pipeline_version=PIPELINE_VERSION, face_size=list(FACE_SIZE),
        label_map=label_map, original_images=len(records), training_samples=len(faces),
        augment_copies=augment, trained_at=datetime.now().isoformat(timespec="seconds")))
    clear_threshold(method)     # an old threshold belonged to the old model
    print(f"Saved model: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    return dict(method=method, path=path, samples=len(faces), originals=len(records))


def load_recognizer(method):
    """Returns (recognizer, absolute model path, meta). Refuses stale models."""
    require_face_module()
    path = model_path(method)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found:\n{path}\nTrain the {METHOD_LABELS[method]} model first.")
    meta = _read_json(meta_path(method))
    if not meta or meta.get("pipeline_version") != PIPELINE_VERSION:
        raise RuntimeError(f"{path} was not trained with the current pipeline. Retrain {METHOD_LABELS[method]}.")
    if meta.get("label_map") != load_label_map():
        raise RuntimeError(f"{path} was trained with a different label map than models/label_map.json. "
                           f"Retrain (main.py option 5 retrains all three).")
    rec = create_recognizer(method)
    rec.read(path)
    return rec, os.path.abspath(path), meta


def describe_model(path, meta):
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    return (f"{path}\n  modified {mtime} | trained {meta.get('trained_at')} | "
            f"{meta.get('original_images')} images -> {meta.get('training_samples')} samples")


# ============================================================
# PREDICTION + UNKNOWN LOGIC (identical everywhere)
# ============================================================
def recognize_face(recognizer, face, id_to_name, threshold):
    label_id, distance = recognizer.predict(face)
    label_id, distance = int(label_id), float(distance)
    closest = id_to_name.get(label_id, "?")
    known = (label_id in id_to_name) and bool(np.isfinite(distance)) and distance <= threshold
    return dict(label_id=label_id, closest=closest, name=closest if known else "Unknown",
                distance=distance, known=bool(known), threshold=float(threshold))


def analyze_image(img, recognizer, id_to_name, threshold, detector=None, allow_precropped=False):
    """
    THE recognition pipeline. Webcam frames, Streamlit uploads and check_photo.py
    all call this one function:
        BGR/gray image -> gray -> Haar faces (largest first) -> crop -> 200x200
        -> equalizeHist (once) -> recognizer.predict -> threshold -> name.
    Returns a list of dicts: box, mode, face_img + recognize_face() fields.
    """
    gray = to_gray(img)
    detector = detector or get_face_detector()
    out = []
    for box, mode in locate_faces(gray, detector, allow_precropped):
        face = preprocess_face(gray, box)
        if face is None:
            continue
        out.append(dict(box=box, mode=mode, face_img=face,
                        **recognize_face(recognizer, face, id_to_name, threshold)))
    return out


def recognize_image(gray, recognizer, id_to_name, threshold, allow_precropped=False, detector=None):
    """
    THE recognition pipeline. Webcam (recognize_realtime.py), Streamlit (app.py) and
    check_photo.py ALL call this one function, so they cannot drift apart:

        gray image -> Haar face detection (largest first) -> crop -> 200x200
        -> histogram equalization (once) -> recognizer.predict -> threshold -> name

    Returns one dict per detected face: label_id, closest, name, distance, known,
    threshold, box, mode, face_img (the exact 200x200 array the model saw).
    """
    results = []
    for box, mode in locate_faces(gray, detector, allow_precropped=allow_precropped):
        face = preprocess_face(gray, box)
        if face is None:
            continue
        res = recognize_face(recognizer, face, id_to_name, threshold)
        res.update(box=box, mode=mode, face_img=face)
        results.append(res)
    return results

def _threshold_store():
    return _read_json(THRESHOLDS_PATH) or {}


def get_threshold(method, override=None):
    if override is not None:
        return float(override)
    entry = _threshold_store().get(method)
    return float(entry["threshold"]) if entry else float(DEFAULT_THRESHOLDS[method])


def threshold_source(method):
    return "calibrated on dataset" if method in _threshold_store() else "default, UNCALIBRATED"


def save_threshold(method, value, info):
    store = _threshold_store()
    store[method] = dict(threshold=round(float(value), 4), **info)
    _write_json(THRESHOLDS_PATH, store)


def clear_threshold(method):
    store = _threshold_store()
    if store.pop(method, None) is not None:
        _write_json(THRESHOLDS_PATH, store)