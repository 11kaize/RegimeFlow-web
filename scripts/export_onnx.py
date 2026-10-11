"""Compatibility entrypoint for the corrected, strict CPU ONNX exporter."""
import argparse
from pathlib import Path
from export_onnx_v2 import ROOT, export

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'ckpt/RegimeFlow/seed53_best.ckpt')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.export or args.test:
        export(args.checkpoint, args.output_root)
    else:
        parser.print_help()
