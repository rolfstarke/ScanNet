import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ScanNet visualization toolkit")
    parser.add_argument("--visualize", action="store_true",
                        help="visualize a ScanNet scene")
    parser.add_argument("--predict", action="store_true",
                        help="run model predictions on a scene")
    parser.add_argument("--models", type=str,
                        help="comma-separated models to predict with (e.g. mosaic3d,openins3d)")
    parser.add_argument("--classes", type=str, default="chair,table",
                        help="comma-separated classes to predict (default: chair,table)")
    parser.add_argument("--gpu", type=int, nargs="*", default=None,
                        help="GPU indices to use (default: all available GPUs)")
    parser.add_argument("--label_set", type=str, default="scannet18",
                        choices=["scannet18", "scannet200"],
                        help="label vocabulary for prediction output ids (default: scannet18)")
    parser.add_argument("--scene", nargs="+", required=True,
                        help="scene numbers (e.g., 0568_00 0304_00)")
    args = parser.parse_args()

    if args.visualize:
        if len(args.scene) != 1:
            parser.error("--visualize takes exactly one --scene")
        from utils.visualize import visualize
        visualize(f"scene{args.scene[0]}")
    elif args.predict:
        if not args.models:
            parser.error("--predict requires --models")
        from predict.runner import predict
        predict([f"scene{s}" for s in args.scene], args.models.split(","),
                args.classes.split(","), args.gpu, args.label_set)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
