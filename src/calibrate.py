import argparse
import os

import numpy as np

import utils

DEFAULT_MAX_FAR = 0.05


def _predict_all(recognizer, records):
    out = []
    for r in records:
        lab, dist = recognizer.predict(utils.normalize_face(r["crop"]))
        out.append((int(lab), float(dist)))
    return out


def collect_distances(method, augment):
    label_map = utils.load_label_map()
    train, _ = utils.load_split(utils.DATASET_TRAIN_DIR, label_map, verbose=False)
    test, _ = utils.load_split(utils.DATASET_TEST_DIR, label_map, allow_unknown=True, verbose=False)
    if not test:
        raise RuntimeError("dataset/test has no usable images - cannot calibrate.")

    full, _, _ = utils.load_recognizer(method)
    correct, wrong, impostor, per_class = [], [], [], {}
    for r, (lab, d) in zip(test, _predict_all(full, test)):
        if r["label"] < 0:
            impostor.append(d)
            continue
        pc = per_class.setdefault(r["name"], dict(n=0, correct=[], wrong=[]))
        pc["n"] += 1
        if lab == r["label"]:
            correct.append(d)
            pc["correct"].append(d)
        else:
            wrong.append(d)
            pc["wrong"].append(d)
    n_genuine = sum(1 for r in test if r["label"] >= 0)

    for person in sorted({r["label"] for r in train}):
        test_p = [r for r in test if r["label"] == person]
        rest = [r for r in train if r["label"] != person]
        if not test_p or len({r["label"] for r in rest}) < 2:
            continue
        faces, labels = utils.build_training_arrays(rest, augment)
        rec = utils.create_recognizer(method, n_samples=len(faces))
        rec.train(faces, labels)
        impostor += [d for _, d in _predict_all(rec, test_p)]

    return dict(correct=correct, wrong=wrong, impostor=impostor,
                n_genuine=n_genuine, per_class=per_class)


def choose_threshold(correct, negatives, n_genuine, max_far):
    correct = np.sort(np.asarray(correct, float))
    neg = np.sort(np.asarray(negatives, float))
    if len(correct) == 0:
        return None
    if len(neg) == 0:
        return dict(threshold=float(correct.max() * 1.15), tpr=len(correct) / n_genuine,
                    far=None, mode="no negatives available: max correct distance x 1.15")
    cands = np.unique(np.concatenate([correct, neg]))
    rows = []
    for t in cands:
        tpr = np.searchsorted(correct, t, side="right") / max(n_genuine, 1)
        far = np.searchsorted(neg, t, side="right") / len(neg)
        rows.append((float(t), float(tpr), float(far)))
    ok = [r for r in rows if r[2] <= max_far]
    if ok and max(r[1] for r in ok) > 0:
        best = max(r[1] for r in ok)
        t_hi = max(r[0] for r in ok if r[1] >= best - 1e-12)
        higher = cands[cands > t_hi]
        nxt = float(higher[0]) if len(higher) else t_hi * 1.1
        thr, mode = (t_hi + nxt) / 2, f"max true-accept with false-accept <= {max_far:.0%}"
    else:
        thr = max(rows, key=lambda r: r[1] - r[2])[0]
        mode = "Youden fallback (correct/unknown distances overlap too much for the FAR limit)"
    tpr = float(np.searchsorted(correct, thr, side="right") / max(n_genuine, 1))
    far = float(np.searchsorted(neg, thr, side="right") / len(neg))
    return dict(threshold=float(thr), tpr=tpr, far=far, mode=mode)


def _stats(a):
    if not len(a):
        return "n=0"
    a = np.asarray(a)
    return (f"n={len(a)} min={a.min():.1f} median={np.median(a):.1f} "
            f"p95={np.percentile(a, 95):.1f} max={a.max():.1f}")


def _plot(method, data, thr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in (("correct", "correct match", "tab:green"),
                              ("wrong", "wrong known person", "tab:orange"),
                              ("impostor", "unknown person", "tab:red")):
        if data[key]:
            ax.hist(data[key], bins=30, alpha=0.6, label=label, color=color)
    ax.axvline(thr, color="k", ls="--", label=f"threshold {thr:.1f}")
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.set_title(f"{utils.METHOD_LABELS[method]} distance distributions")
    ax.legend()
    utils.ensure_dirs(utils.RESULTS_DIR)
    fig.savefig(os.path.join(utils.RESULTS_DIR, f"distance_hist_{method}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def calibrate_and_save(method, augment=utils.DEFAULT_AUGMENT_COPIES, max_far=DEFAULT_MAX_FAR):
    print(f"\n--- Calibrating {utils.METHOD_LABELS[method]} threshold ---")
    try:
        data = collect_distances(method, augment)
    except Exception as e:
        print(f"[WARN] Calibration failed: {e}\n       Using default threshold "
              f"{utils.DEFAULT_THRESHOLDS[method]} (UNCALIBRATED).")
        return None
    negatives = data["wrong"] + data["impostor"]
    print(f"correct match   : {_stats(data['correct'])}")
    print(f"wrong known     : {_stats(data['wrong'])}")
    print(f"unknown persons : {_stats(data['impostor'])}")
    for name, pc in sorted(data["per_class"].items()):
        print(f"  {name}: {len(pc['correct'])}/{pc['n']} correct"
              + (f", median correct distance {np.median(pc['correct']):.1f}" if pc["correct"] else ""))
    res = choose_threshold(data["correct"], negatives, data["n_genuine"], max_far)
    if res is None:
        print("[WARN] No correct predictions on the test set - cannot calibrate. Using default.")
        return None
    utils.save_threshold(method, res["threshold"], dict(
        mode=res["mode"], true_accept_rate=round(res["tpr"], 4),
        false_accept_rate=None if res["far"] is None else round(res["far"], 4),
        n_correct=len(data["correct"]), n_wrong=len(data["wrong"]), n_unknown=len(data["impostor"]),
        max_far=max_far))
    _plot(method, data, res["threshold"])
    far = "n/a" if res["far"] is None else f"{res['far']:.1%}"
    print(f"=> {method} threshold = {res['threshold']:.2f}   "
          f"(true-accept {res['tpr']:.1%}, false-accept {far}; {res['mode']})")
    if res["tpr"] < 0.5:
        print("[WARN] Fewer than half of correct test matches pass. Classes are not well separated "
              "for this method; add more varied training images rather than raising the threshold.")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(utils.METHODS) + ["all"], default="all")
    ap.add_argument("--augment", type=int, default=utils.DEFAULT_AUGMENT_COPIES)
    ap.add_argument("--max-far", type=float, default=DEFAULT_MAX_FAR)
    a = ap.parse_args()
    for m in (utils.METHODS if a.method == "all" else [a.method]):
        calibrate_and_save(m, a.augment, a.max_far)
