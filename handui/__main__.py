"""Command line: python -m handui [run|record|train|labels|forget]"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m handui", description="Hand-tracking gesture UI")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="start the gesture UI (default)")
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--width", type=int, default=1280)
    p_run.add_argument("--height", type=int, default=720)

    p_rec = sub.add_parser("record", help="record samples for a custom gesture")
    p_rec.add_argument("label", help="gesture name, e.g. thumbs_up (use 'none' for background poses)")
    p_rec.add_argument("--seconds", type=float, default=8.0)
    p_rec.add_argument("--camera", type=int, default=0)

    p_train = sub.add_parser("train", help="train the custom gesture classifier")
    p_train.add_argument("--augment", type=int, default=3, help="augmented copies per sample")

    sub.add_parser("labels", help="list recorded gesture labels")

    p_forget = sub.add_parser("forget", help="delete all samples for a label")
    p_forget.add_argument("label")

    args = parser.parse_args()

    if args.command in (None, "run"):
        from .app import run
        run(getattr(args, "camera", 0), getattr(args, "width", 1280), getattr(args, "height", 720))
    elif args.command == "record":
        from .record import record
        record(args.label, args.seconds, args.camera)
    elif args.command == "train":
        from .classifier import train
        try:
            report = train(augment=args.augment)
        except ValueError as e:
            parser.exit(1, f"{e}\n")
        for label, n in sorted(report["counts"].items()):
            print(f"  {label:<20} {n:>5} samples")
        if report["holdout_accuracy"] is not None:
            print(f"Held-out accuracy: {report['holdout_accuracy']:.1%}")
        print(f"Saved model to {report['model_path']}")
    elif args.command == "labels":
        from .classifier import label_counts
        counts = label_counts()
        if not counts:
            print("No samples yet. Try: python -m handui record thumbs_up")
        for label, n in sorted(counts.items()):
            print(f"  {label:<20} {n:>5} samples")
    elif args.command == "forget":
        from .classifier import remove_label
        print(f"Removed {remove_label(args.label)} samples for '{args.label}'.")


if __name__ == "__main__":
    main()
