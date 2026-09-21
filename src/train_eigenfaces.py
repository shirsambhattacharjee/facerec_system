import argparse

import utils
import facerec_system.src.calibrate as calibrate


def run(augment=utils.DEFAULT_AUGMENT_COPIES, do_calibrate=True, max_far=calibrate.DEFAULT_MAX_FAR):
    info = utils.train_method("eigen", augment)
    if do_calibrate:
        calibrate.calibrate_and_save("eigen", augment, max_far)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Eigenfaces")
    ap.add_argument("--augment", type=int, default=utils.DEFAULT_AUGMENT_COPIES)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--max-far", type=float, default=calibrate.DEFAULT_MAX_FAR)
    a = ap.parse_args()
    run(a.augment, not a.no_calibrate, a.max_far)
