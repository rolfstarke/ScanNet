import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ScanNet visualization toolkit")
    parser.add_argument("--visualize", action="store_true",
                        help="visualize a ScanNet scene")
    parser.add_argument("--predict", action="store_true",
                        help="run model predictions on a scene")
    parser.add_argument("--models", type=str,
                        help="comma-separated models to predict with (e.g. mosaic3d,openins3d)")
    parser.add_argument("--classes", type=str, default=None,
                        help="comma-separated classes to predict (default: the benchmark's "
                             "official class list)")
    parser.add_argument("--gpu", type=int, nargs="*", default=None,
                        help="GPU indices to use (default: all available GPUs)")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["ScanNet20", "ScanNet200"],
                        help="benchmark backend (default: spellbook/settings.yaml)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="run id for output isolation (default: auto-generated)")
    parser.add_argument("--engine", nargs="+", default=None,
                        choices=["zed", "metashape", "rtabmap", "isaac", "open3d",
                                 "bundlefusion"],
                        help="reconstruct mode: SVO2 -> ScanNet-native scan per "
                             "(scene, engine); frames extracted once, tasks parallel over --gpu")
    parser.add_argument("--replace", action="store_true",
                        help="re-extract frames even if a complete set exists")
    parser.add_argument("--scene", nargs="+", required=True,
                        help="scene numbers (e.g., 0568_00 0304_00 or 9004 9009)")
    args = parser.parse_args()

    from benchmark import load_settings, resolve_benchmark
    benchmark = args.benchmark or load_settings()["default"]
    spec = resolve_benchmark(benchmark)

    if args.visualize:
        if len(args.scene) != 1:
            parser.error("--visualize takes exactly one --scene")
        from utils.visualize import visualize
        visualize(f"scene{args.scene[0]}", benchmark=benchmark, run_id=args.run_id)
    elif args.predict:
        if not args.models:
            parser.error("--predict requires --models")
        run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
        classes = args.classes.split(",") if args.classes else None
        n_classes = len(classes) if classes else len(spec.class_labels)
        print(f"[INFO] benchmark={spec.name} run_id={run_id} classes={n_classes}")
        from predict.runner import predict
        predict([f"scene{s}" for s in args.scene], args.models.split(","),
                classes, args.gpu, benchmark, run_id)
    elif args.engine:
        from reconstruct.batch import run_batch
        ok, failed = run_batch([int(s) for s in args.scene], args.engine,
                               args.gpu, args.replace)
        sys.exit(1 if failed else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
