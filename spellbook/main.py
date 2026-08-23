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
    parser.add_argument("--gpu-check", action="store_true",
                        help="GPU distribution smoke: launch native runtimes of every "
                             "prediction/reconstruction method under the settings GPU "
                             "pool (1-4; physical GPU 0 user-reserved), prove the "
                             "assignment, hold briefly, exit. Produces no results.")
    parser.add_argument("--models", type=str,
                        help="comma-separated models to predict with (e.g. mosaic3d,openins3d)")
    parser.add_argument("--classes", type=str, default=None,
                        help="comma-separated classes to predict (default: the benchmark's "
                             "official class list)")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["ScanNet20", "ScanNet200"],
                        help="benchmark backend (default: spellbook/settings.yaml)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="run id for output isolation (default: auto-generated)")
    parser.add_argument("--engine", nargs="+", default=None,
                        choices=["zed", "metashape", "rtabmap", "isaac", "open3d",
                                 "bundlefusion"],
                        help="reconstruct mode: SVO2 -> ScanNet-native scan per "
                             "(scene, engine); frames extracted once, tasks automatically "
                             "share the settings gpu_pool")
    parser.add_argument("--extract-frames", action="store_true",
                        help="main-checkout multi-GPU SVO frame extraction into the "
                             "shared pool (never from engine worktrees)")
    parser.add_argument("--replace", action="store_true",
                        help="re-extract frames even if a complete set exists")
    parser.add_argument("--scene", nargs="+",
                        help="scene numbers (e.g., 0568_00 0304_00 or 9004 9009)")
    args = parser.parse_args()

    from benchmark import load_settings, resolve_benchmark
    benchmark = args.benchmark or load_settings()["default"]
    spec = resolve_benchmark(benchmark)

    if args.gpu_check:
        if args.visualize or args.predict or args.extract_frames:
            parser.error("--gpu-check is exclusive with --visualize/--predict/--extract-frames")
        if args.scene:
            parser.error("--gpu-check does not use --scene")
        import sys as _sys
        _sys.argv = ["gpu-check"]
        if args.models:
            _sys.argv += ["--models", args.models]
        if args.engine:
            _sys.argv += ["--engine", *args.engine]
        from gpu_check import main as gpu_check_main
        gpu_check_main()
    elif args.extract_frames:
        if args.visualize or args.predict or args.engine:
            parser.error("--extract-frames is exclusive with --visualize/--predict/--engine")
        if not args.scene:
            parser.error("--extract-frames requires --scene")
        from reconstruct.extract import extract_scenes
        extract_scenes([int(s) for s in args.scene], replace=args.replace)
    elif args.visualize:
        if len(args.scene or []) != 1:
            parser.error("--visualize takes exactly one --scene")
        from utils.visualize import visualize
        visualize(f"scene{args.scene[0]}", benchmark=benchmark, run_id=args.run_id)
    elif args.predict:
        if not args.models:
            parser.error("--predict requires --models")
        if not args.scene:
            parser.error("--predict requires --scene")
        run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
        classes = args.classes.split(",") if args.classes else None
        n_classes = len(classes) if classes else len(spec.class_labels)
        print(f"[INFO] benchmark={spec.name} run_id={run_id} classes={n_classes}")
        from predict.runner import predict
        predict([f"scene{s}" for s in args.scene], args.models.split(","),
                classes, benchmark, run_id)
    elif args.engine:
        if not args.scene:
            parser.error("--engine requires --scene")
        from reconstruct.batch import run_batch
        ok, failed = run_batch([int(s) for s in args.scene], args.engine,
                               args.replace)
        sys.exit(1 if failed else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
