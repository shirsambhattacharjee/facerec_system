import argparse
import os

import numpy as np

import utils


def audit():
    label_map = utils.load_label_map() if os.path.exists(utils.LABEL_MAP_PATH) else utils.build_label_map()
    print("Label map:", label_map)
    utils.load_split(utils.DATASET_TRAIN_DIR, label_map, verbose=True)
    utils.load_split(utils.DATASET_TEST_DIR, label_map, allow_unknown=True, verbose=True)
    print("\nModels:")
    for m in utils.METHODS:
        p = utils.model_path(m)
        if not os.path.exists(p):
            print(f"  {m:<7} MISSING {p}")
            continue
        try:
            _, path, meta = utils.load_recognizer(m)
            print(f"  {m:<7} OK   {utils.describe_model(path, meta)}")
        except Exception as e:
            print(f"  {m:<7} STALE/INVALID: {e}")
        print(f"          threshold {utils.get_threshold(m):.2f} ({utils.threshold_source(m)})")


def selfcheck():
    """Training images pushed through the inference path MUST give distance ~0."""
    label_map = utils.load_label_map()
    id_to_name = utils.id_to_name_map(label_map)
    records, _ = utils.load_split(utils.DATASET_TRAIN_DIR, label_map, verbose=False)
    for m in utils.METHODS:
        rec, path, _ = utils.load_recognizer(m)
        d, ok = [], 0
        for r in records:
            res = utils.recognize_face(rec, utils.normalize_face(r["crop"]), id_to_name, 1e12)
            d.append(res["distance"])
            ok += res["label_id"] == r["label"]
        flag = "OK" if max(d) < 1e-2 else "PIPELINE MISMATCH (train vs inference differ!)"
        print(f"{utils.METHOD_LABELS[m]:<12} correct {ok}/{len(records)}  "
              f"distance mean {np.mean(d):.4f} max {max(d):.4f}  -> {flag}")


def check_image(path, methods):
    gray = utils.read_gray(path)
    if gray is None:
        raise SystemExit(f"Cannot read {path}")
    label_map = utils.load_label_map()
    id_to_name = utils.id_to_name_map(label_map)
    regions = utils.locate_faces(gray, utils.get_face_detector(), allow_precropped=True)
    print(f"Image {gray.shape[1]}x{gray.shape[0]}, faces found: {len(regions)}")
    for i, (box, mode) in enumerate(regions, 1):
        face = utils.preprocess_face(gray, box)
        out = os.path.join(utils.RESULTS_DIR, f"debug_face_{i}.png")
        utils.write_image(out, face)
        print(f"\nFace {i}: box={box} mode={mode} processed={face.shape[1]}x{face.shape[0]} "
              f"mean={face.mean():.0f} std={face.std():.0f}  (saved {out})")
        for m in methods:
            rec, mpath, _ = utils.load_recognizer(m)
            thr = utils.get_threshold(m)
            r = utils.recognize_face(rec, face, id_to_name, thr)
            print(f"  {utils.METHOD_LABELS[m]:<12} id={r['label_id']} closest={r['closest']:<8} "
                  f"dist={r['distance']:.2f} thr={thr:.2f} -> {r['name']}\n"
                  f"               model: {mpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--method", choices=list(utils.METHODS) + ["all"], default="all")
    a = ap.parse_args()
    if a.audit:
        audit()
    if a.selfcheck:
        selfcheck()
    if a.image:
        check_image(a.image, utils.METHODS if a.method == "all" else [a.method])
    if not (a.audit or a.selfcheck or a.image):
        ap.print_help()