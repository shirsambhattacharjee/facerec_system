import argparse
import math
import os
import time
from collections import Counter, deque

import cv2
import numpy as np

import utils

WINDOW = "Real-Time Face Recognition"
QUIT_KEYS = (ord("q"), ord("Q"), 27)        

def _text(img, text, org, color, scale=0.6, thick=2):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _bring_to_front(win):
    """Ask Windows to put the OpenCV window on top so it receives the q key."""
    try:
        prop = getattr(cv2, "WND_PROP_TOPMOST", 5)
        cv2.setWindowProperty(win, prop, 1)
    except cv2.error:
        pass


def _release_topmost(win):
    try:
        cv2.setWindowProperty(win, getattr(cv2, "WND_PROP_TOPMOST", 5), 0)
    except cv2.error:
        pass


def _window_closed(win):
    """True if the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True

class FaceTracker:
    def __init__(self, history=5, max_missed=8):
        self.history = max(1, int(history))
        self.max_missed = max_missed
        self.tracks = []
        self._next_id = 0

    def update(self, results):
        """results: list of dicts from utils.recognize_image (largest face first)."""
        used = set()
        for r in results:
            x, y, w, h = r["box"]
            cx, cy = x + w / 2.0, y + h / 2.0
            best, best_d = None, 0.75 * max(w, h)
            for t in self.tracks:
                if t["id"] in used:
                    continue
                d = math.hypot(cx - t["c"][0], cy - t["c"][1])
                if d < best_d:
                    best, best_d = t, d
            if best is None:
                best = dict(id=self._next_id, hist=deque(maxlen=self.history), missed=0, c=(cx, cy))
                self._next_id += 1
                self.tracks.append(best)
            best["c"], best["missed"] = (cx, cy), 0
            best["hist"].append((r["label_id"], r["distance"]))
            used.add(best["id"])
            r["track"] = best
        for t in self.tracks:
            if t["id"] not in used:
                t["missed"] += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]


def smooth_decision(hist, id_to_name, threshold):
    """(label_id, distance, known) from a track's recent per-frame predictions."""
    counts = Counter(l for l, _ in hist)
    top = max(counts.values())

    def med(label):
        return float(np.median([d for l, d in hist if l == label]))

    label = min((l for l, c in counts.items() if c == top), key=med)
    dist = med(label)
    return label, dist, (label in id_to_name and dist <= threshold)


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main(method="lbph", threshold=None, camera_index=0, debug=False, smooth=5,
         flip=False, use_default_threshold=False):
    if method not in utils.METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {utils.METHODS}")

    detector = utils.get_face_detector()
    recognizer, model_path, meta = utils.load_recognizer(method)
    label_map = utils.load_label_map()
    id_to_name = utils.id_to_name_map(label_map)

    if use_default_threshold:
        thr, source = float(utils.DEFAULT_THRESHOLDS[method]), "built-in default (forced)"
    elif threshold is not None:
        thr, source = float(threshold), "supplied by you"
    else:
        thr, source = utils.get_threshold(method), utils.threshold_source(method)

    print("\nReal-time recognition")
    print(f"utils.py   : {os.path.abspath(utils.__file__)}")
    print(f"Method     : {utils.METHOD_LABELS[method]}")
    print(f"Model file : {utils.describe_model(model_path, meta)}")
    print(f"Label map  : {label_map}")
    print(f"Threshold  : {thr:.2f} ({source})")
    print(f"Smoothing  : last {max(1, smooth)} frame(s) | debug: {debug} | flip: {flip}")
    print("Click the camera window, then press Q (or ESC) to quit.\n")

    cap = utils.open_camera(camera_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam (camera index {camera_index}). "
                      f"Close other apps using the camera or try --camera 1.")

    tracker = FaceTracker(history=smooth)
    frame_no, fps, prev = 0, 0.0, time.time()
    front_until = time.time() + 1.5

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        _bring_to_front(WINDOW)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read a frame from the camera.")
                break
            if flip:
                frame = cv2.flip(frame, 1)
            frame = utils.limit_size(frame)
            gray = utils.to_gray(frame)
            frame_no += 1

           
            results = utils.recognize_image(gray, recognizer, id_to_name, thr,
                                            allow_precropped=False, detector=detector)
            tracker.update(results)

            for idx, r in enumerate(results):
                x, y, w, h = r["box"]
                label, dist, known = smooth_decision(r["track"]["hist"], id_to_name, thr)
                name = id_to_name[label] if known else "Unknown"
                color = (0, 255, 0) if known else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                _text(frame, f"{name} ({dist:.1f})", (x, y - 10 if y > 30 else y + h + 25),
                      color, 0.7)
                if debug and frame_no % 10 == 0:
                    print(f"[{method}] box={r['box']} face={r['face_img'].shape[1]}x{r['face_img'].shape[0]} "
                          f"| this frame: id={r['label_id']} {r['closest']} d={r['distance']:.2f} "
                          f"| shown: id={label} {id_to_name.get(label, '?')} d={dist:.2f} "
                          f"thr={thr:.2f} -> {name}")
                if debug and idx == 0:                       # largest face
                    lines = [f"Method: {utils.METHOD_LABELS[method]}", f"Threshold: {thr:.1f}",
                             f"ID: {label}", f"Name: {name}"
                             + ("" if known else f" (closest: {id_to_name.get(label, '?')})"),
                             f"Distance: {dist:.2f}"]
                    for i, line in enumerate(lines):
                        _text(frame, line, (10, 85 + 24 * i), (0, 255, 255), 0.55, 1)
                    H, W = frame.shape[:2]
                    thumb = cv2.cvtColor(cv2.resize(r["face_img"], (120, 120)), cv2.COLOR_GRAY2BGR)
                    frame[10:130, W - 130:W - 10] = thumb   # what the model actually sees

            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            prev = now
            _text(frame, f"FPS {fps:.0f} | {utils.METHOD_LABELS[method]} | thr {thr:.1f}",
                  (10, 28), (255, 255, 0), 0.6)
            _text(frame, "Click this window, then press Q / ESC to quit", (10, 56),
                  (255, 255, 255), 0.5, 1)

            cv2.imshow(WINDOW, frame)
            if front_until and time.time() > front_until:    
                _release_topmost(WINDOW)
                front_until = 0

            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                print("Quit key pressed.")
                break
            if frame_no > 5 and _window_closed(WINDOW):
                print("Window closed.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(4):              
            cv2.waitKey(1)
        print("Camera released, windows closed.")



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-Time Face Recognition")
    ap.add_argument("--method", choices=list(utils.METHODS), default="lbph")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold (default: central calibrated threshold)")
    ap.add_argument("--use-default-threshold", action="store_true",
                    help="ignore calibration and use the built-in default "
                         "(lbph 70 / eigen 4500 / fisher 400)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--debug", action="store_true",
                    help="show Method/Threshold/ID/Name/Distance on screen + console + model input thumbnail")
    ap.add_argument("--smooth", type=int, default=5,
                    help="frames used to stabilise each face's label (1 = off)")
    ap.add_argument("--flip", action="store_true", help="mirror the camera image")
    a = ap.parse_args()
    main(method=a.method, threshold=a.threshold, camera_index=a.camera, debug=a.debug,
         smooth=a.smooth, flip=a.flip, use_default_threshold=a.use_default_threshold)