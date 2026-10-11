"""Compare the corrected ONNX engine with independent upstream CPU semantics."""
import argparse
from pathlib import Path
from validate_inference import validate_reference

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reexport', type=Path)
    validate_reference(parser.parse_args().reexport)
