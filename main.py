import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)                     

import cv2 
import utils  

EXPECTED_UTILS = os.path.abspath(os.path.join(SRC, "utils.py"))
if os.path.normcase(os.path.abspath(utils.__file__)) != os.path.normcase(EXPECTED_UTILS):
    raise SystemExit(f"WRONG utils.py imported: {utils.__file__}\nExpected: {EXPECTED_UTILS}")


def collect_faces():
    name = input("Person name (folder name): ").strip()
    if not name:
        return
    split = (input("Split [train/test] (default train): ").strip().lower() or "train")
    if split not in ("train", "test"):
        print("Split must be train or test.")
        return
    total = int(input("Number of images (default 50): ").strip() or 50)
    out_dir = os.path.join(ROOT, "dataset", split, name)
    os.makedirs(out_dir, exist_ok=True)
    existing = [int(os.path.splitext(f)[0].split("_")[-1]) for f in os.listdir(out_dir)
                if f.lower().endswith(utils.VALID_EXTENSIONS)
                and os.path.splitext(f)[0].split("_")[-1].isdigit()]
    idx = max(existing, default=0) + 1

    detector = utils.get_face_detector()
    cap = utils.open_camera(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return
    print("Look at the camera; vary lighting/angle. Saves the RAW face crop when exactly one "
          "face is visible. Click the window and press Q to stop.")
    saved, last, win = 0, 0.0, "Collect faces - click here, press Q to stop"
    try:
        while saved < total:
            ok, frame = cap.read()
            if not ok:
                break
            frame = utils.limit_size(frame)
            gray = utils.to_gray(frame)
            boxes = utils.detect_faces(gray, detector)
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(boxes) == 1 and time.time() - last > 0.3:
                crop = utils.crop_face(gray, boxes[0])
                if crop is not None:
                    utils.write_image(os.path.join(out_dir, f"{name}_{idx:04d}.png"), crop)
                    idx += 1
                    saved += 1
                    last = time.time()
            cv2.putText(frame, f"{name}: {saved}/{total}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)
    print(f"Saved {saved} images to {out_dir}. Retrain ALL models (option 5) afterwards.")


def train_all():
    import train_lbph
    import train_eigenfaces
    import train_fisherfaces
    for mod in (train_lbph, train_eigenfaces, train_fisherfaces):
        mod.run()
    print("\nAll three models retrained from the current dataset.")


def realtime():
    import recognize_realtime
    method = input("Method [lbph/eigen/fisher] (default lbph): ").strip().lower() or "lbph"
    if method not in utils.METHODS:
        print("Unknown method.")
        return
    raw = input(f"Threshold override (Enter = {utils.get_threshold(method):.1f}, "
                f"{utils.threshold_source(method)}): ").strip()
    try:
        threshold = float(raw) if raw else None
    except ValueError:
        print("Not a number - using the central threshold.")
        threshold = None
    debug = input("Debug overlay? [y/N]: ").strip().lower() == "y"
    recognize_realtime.main(method=method, threshold=threshold, camera_index=0, debug=debug)


def evaluate_all():
    import evaluate
    evaluate.main()


def menu():
    actions = {
        "1": ("Collect face images", collect_faces),
        "2": ("Train LBPH", lambda: __import__("train_lbph").run()),
        "3": ("Train Eigenfaces", lambda: __import__("train_eigenfaces").run()),
        "4": ("Train Fisherfaces", lambda: __import__("train_fisherfaces").run()),
        "5": ("Train ALL models", train_all),
        "6": ("Run real-time recognition", realtime),
        "7": ("Evaluate & compare all methods", evaluate_all),
    }
    while True:
        print("\n===== Real-Time Face Recognition System =====")
        for k, (label, _) in actions.items():
            print(f"{k}. {label}")
        print("8. Exit")
        choice = input("Select: ").strip()
        if choice == "8":
            break
        if choice in actions:
            try:
                actions[choice][1]()
            except Exception as e:
                print(f"\n[ERROR] {e}")
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    print(f"utils.py in use: {os.path.abspath(utils.__file__)}")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train-all":
        train_all()
    elif cmd == "evaluate":
        evaluate_all()
    else:
        menu()