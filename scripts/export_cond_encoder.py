"""Export the condition encoder using strict checkpoint/EMA loading."""
import argparse
from pathlib import Path
from export_onnx_v2 import ROOT, export

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'ckpt/RegimeFlow/seed53_best.ckpt')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    export(args.checkpoint, args.output_root, component='condition')
