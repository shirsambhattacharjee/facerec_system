import argparse
import csv
import os
import time

import numpy as np

import utils


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def macro_precision_recall(cm):
    prec, rec = [], []
    for i in range(len(cm)):
        row, col = cm[i, :].sum(), cm[:, i].sum()
        if row == 0 and col == 0:
            continue                                   
        prec.append(cm[i, i] / col if col else 0.0)
        rec.append(cm[i, i] / row if row else 0.0)
    return float(np.mean(prec)), float(np.mean(rec))


def plot_confusion(cm, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_method(method, test, label_map):
    rec, path, meta = utils.load_recognizer(method)
    id_to_name = utils.id_to_name_map(label_map)
    thr = utils.get_threshold(method)
    print(f"\n{utils.METHOD_LABELS[method]}: model {path}\n  threshold {thr:.2f} ({utils.threshold_source(method)})")

    y_true, y_pred, closest_ok, times = [], [], [], []
    for r in test:
        t0 = time.perf_counter()
        face = utils.normalize_face(r["crop"])
        res = utils.recognize_face(rec, face, id_to_name, thr)
        times.append((time.perf_counter() - t0) * 1000)
        y_true.append(r["name"] if r["label"] >= 0 else "Unknown")
        y_pred.append(res["name"])
        if r["label"] >= 0:
            closest_ok.append(res["closest"] == r["name"])

    labels = sorted(label_map, key=label_map.get) + ["Unknown"]
    cm = confusion(y_true, y_pred, labels)
    precision, recall = macro_precision_recall(cm)
    accuracy = float(np.trace(cm) / max(cm.sum(), 1))
    utils.ensure_dirs(utils.RESULTS_DIR)
    plot_confusion(cm, labels, f"{utils.METHOD_LABELS[method]} confusion matrix",
                   os.path.join(utils.RESULTS_DIR, f"confusion_matrix_{utils.RESULT_STEMS[method]}.png"))

    print("  confusion matrix (rows=true, cols=pred):", labels)
    print(cm)
    return dict(
        Method=utils.METHOD_LABELS[method], Threshold=round(thr, 3),
        Accuracy=round(accuracy, 4), Precision=round(precision, 4), Recall=round(recall, 4),
        Closest_Class_Accuracy=round(float(np.mean(closest_ok)) if closest_ok else 0.0, 4),
        Unknown_Rate=round(y_pred.count("Unknown") / max(len(y_pred), 1), 4),
        Avg_Time_ms=round(float(np.mean(times)), 3), Test_Images=len(test))


def main(methods=utils.METHODS):
    label_map = utils.load_label_map()
    test, _ = utils.load_split(utils.DATASET_TEST_DIR, label_map, allow_unknown=True, verbose=True)
    if not test:
        raise RuntimeError("No usable images in dataset/test.")
    rows = [evaluate_method(m, test, label_map) for m in methods]

    utils.ensure_dirs(utils.RESULTS_DIR)
    csv_path = os.path.join(utils.RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(rows))
        for k, key in enumerate(("Accuracy", "Precision", "Recall")):
            ax.bar(x + (k - 1) * 0.25, [r[key] for r in rows], 0.25, label=key)
        ax.set_xticks(x)
        ax.set_xticklabels([r["Method"] for r in rows])
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Method comparison (dataset/test)")
        fig.savefig(os.path.join(utils.RESULTS_DIR, "comparison.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass

    print("\n" + "=" * 78)
    print(f"{'Method':<12}{'Thr':>9}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'Closest':>9}{'Unk%':>7}{'ms':>8}")
    for r in rows:
        print(f"{r['Method']:<12}{r['Threshold']:>9.2f}{r['Accuracy']:>8.3f}{r['Precision']:>8.3f}"
              f"{r['Recall']:>8.3f}{r['Closest_Class_Accuracy']:>9.3f}{r['Unknown_Rate']:>7.2f}{r['Avg_Time_ms']:>8.2f}")
    print(f"\nSaved {csv_path} and confusion matrices in {utils.RESULTS_DIR}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(utils.METHODS) + ["all"], default="all")
    a = ap.parse_args()
    main(utils.METHODS if a.method == "all" else [a.method])
