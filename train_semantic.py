"""Training entrypoint for yolo26s-sem on the converted Vistas dataset.

Defaults match the plan doc's recommended config for a single RTX 5060 Ti
(16GB): imgsz=640 scalar (training imgsz must be scalar -- a tuple is
silently collapsed to max(imgsz) by check_imgsz(..., max_dim=1), verified
against ultralytics 8.4.167's engine/trainer.py), batch=16, amp on.

cls_pw defaults to 1.0, NOT the Ultralytics default of 0.0. At 0.0, class
weighting is skipped entirely (confirmed by reading
DetectionTrainer.set_class_weights: `if self.args.cls_pw == 0.0: return`)
and curb/curb-cut -- a fraction of a percent of pixels each -- would get
essentially no gradient signal. Since detecting those classes is the entire
reason for this training run, 1.0 is the correct default here even though
it's not Ultralytics' own default.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

from gpu_guard import assert_min_vram, assert_torch_cuda

# The four tensors expected to stay at fresh init when warm-starting from a
# 19- or 150-class checkpoint onto a 124-class head: classifier.1.{weight,bias}
# and aux_head.1.{weight,bias} (verified: SemanticSegmentationTrainer.get_model
# builds the model at the DATASET's nc first, then loads weights through
# intersect_dicts, which keeps only shape-matching keys).
EXPECTED_TRANSFER_DEFICIT = 4


class _TransferCapture(logging.Handler):
    """Captures Ultralytics' "Transferred X/Y items from pretrained weights"
    LOGGER.info line (nn/tasks.py) so we can assert the deficit is exactly
    EXPECTED_TRANSFER_DEFICIT -- the cheapest possible check that the right
    checkpoint got loaded onto the right architecture."""

    _PATTERN = re.compile(r"Transferred (\d+)/(\d+) items")

    def __init__(self) -> None:
        super().__init__()
        self.transferred: int | None = None
        self.total: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        m = self._PATTERN.search(record.getMessage())
        if m:
            self.transferred, self.total = int(m.group(1)), int(m.group(2))


def _check_transfer(capture: _TransferCapture, strict: bool) -> None:
    if capture.total is None:
        msg = ("[train_semantic] could not find the 'Transferred X/Y items' log line -- "
               "skipped the head-swap sanity check.")
        print(msg, file=sys.stderr)
        return

    deficit = capture.total - capture.transferred
    print(f"[train_semantic] pretrained transfer: {capture.transferred}/{capture.total} "
          f"(deficit {deficit}, expected {EXPECTED_TRANSFER_DEFICIT})", file=sys.stderr)
    if deficit != EXPECTED_TRANSFER_DEFICIT:
        msg = (f"[train_semantic] deficit is {deficit}, not {EXPECTED_TRANSFER_DEFICIT} -- "
               "this usually means the wrong checkpoint was loaded, or the model "
               "architecture doesn't match what this plan verified. Check --model "
               "and the dataset YAML's nc.")
        if strict:
            sys.exit(msg)
        print(f"WARNING: {msg}", file=sys.stderr)


def _make_early_abort_callback(capture: _TransferCapture, strict: bool):
    """Registered on 'on_pretrain_routine_end', which fires right after
    _setup_train (model build + weight load + dataloaders) and BEFORE the
    epoch loop starts (_do_train). This is what makes --strict-transfer-check
    actually useful on the full 100-epoch run and not just the smoke test --
    without this hook, the check could only run after .train() returns, i.e.
    6-9 hours too late."""

    def _callback(trainer) -> None:
        _check_transfer(capture, strict)

    return _callback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to vistas-v2.0.yaml (or vistas8.yaml for the smoke run)")
    ap.add_argument("--model", default="yolo26s-sem-ade20k.pt",
                     help="warm-start checkpoint; yolo26s-sem.pt (Cityscapes-19) is the A/B alternative")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="scalar only -- see module docstring")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cls-pw", type=float, default=1.0)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--optimizer", default="auto",
                     help="ultralytics optimizer name. 'auto' picks lr0=0.002*5/(4+nc) "
                          "(~7.8e-5 at nc=124) -- tuned for the full run's thousands of "
                          "iterations, too conservative to overfit a tiny --limit smoke "
                          "set in any reasonable epoch budget. Pass e.g. AdamW with "
                          "--lr0 for smoke runs.")
    ap.add_argument("--lr0", type=float, default=None,
                     help="initial LR; only takes effect with a non-'auto' --optimizer "
                          "(ultralytics ignores lr0 when optimizer='auto')")
    ap.add_argument("--project", default="/runs")
    ap.add_argument("--name", default="vistas124-s-640")
    ap.add_argument("--save-period", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-vram-gb", type=float, default=8.0)
    ap.add_argument("--strict-transfer-check", action="store_true",
                     help="exit instead of warning if the pretrained-transfer deficit is unexpected")
    args = ap.parse_args()

    assert_torch_cuda()
    assert_min_vram(args.min_vram_gb)

    from ultralytics import YOLO

    capture = _TransferCapture()
    logging.getLogger("ultralytics").addHandler(capture)

    model = YOLO(args.model)
    model.add_callback("on_pretrain_routine_end", _make_early_abort_callback(capture, args.strict_transfer_check))
    train_kwargs = dict(
        data=args.data,
        task="semantic",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        amp=args.amp,
        workers=args.workers,
        cls_pw=args.cls_pw,
        close_mosaic=args.close_mosaic,
        cos_lr=True,
        optimizer=args.optimizer,
        device=0,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        plots=True,
        seed=args.seed,
        deterministic=False,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    results = model.train(**train_kwargs)

    print(f"[train_semantic] done. results: {results}", file=sys.stderr)


if __name__ == "__main__":
    main()
