"""Estimate multiply-add operations for Keras models.

TensorFlow's graph profiler reports floating-point operations. For dense and
convolutional networks this is conventionally converted to multiply-adds as
MACs ~= FLOPs / 2. The estimate is for one spectrum, using batch size 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported so Keras can deserialize models that reference these objects.
from model_definitions.model_definitions import WarmupCosineDecay  # noqa: F401


def _input_dim_from_npz(path: Path) -> int:
    data = np.load(path, allow_pickle=True)
    return int(data["X_test_features"].shape[1])


def _format_number(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f}G"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def estimate_model_flops(model_path: Path, input_dim: int) -> tuple[int, int, int]:
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

    @tf.function
    def forward_pass(x: tf.Tensor):
        return model(x, training=False)

    concrete = forward_pass.get_concrete_function(
        tf.TensorSpec([1, input_dim], tf.float32, name="stellar_input")
    )
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        profile = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            cmd="op",
            options=opts,
        )

    flops = int(profile.total_float_ops) if profile is not None else 0
    macs = int(round(flops / 2.0))
    params = int(model.count_params())

    # Keep a reference so TensorFlow does not release objects before profiling
    # finishes in eager execution.
    _ = frozen_func
    return params, flops, macs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate FLOPs and MACs for saved Keras models.")
    parser.add_argument("models", nargs="+", type=Path, help="One or more .keras model paths.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz",
        help="Compact NPZ used only to infer the spectrum input dimension.",
    )
    parser.add_argument("--input-dim", type=int, default=None, help="Override input dimension.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_dim = int(args.input_dim) if args.input_dim is not None else _input_dim_from_npz(args.data_path)

    print(f"Input shape per inference: (1, {input_dim})")
    print("MACs are estimated as TensorFlow profiler FLOPs / 2.")
    print()
    print(f"{'Model':<46} {'Params':>12} {'FLOPs':>12} {'MACs':>12}")
    print("-" * 86)

    rows = []
    for model_path in args.models:
        params, flops, macs = estimate_model_flops(model_path, input_dim)
        rows.append(
            {
                "model": str(model_path),
                "input_dim": input_dim,
                "params": params,
                "flops": flops,
                "macs": macs,
            }
        )
        print(
            f"{model_path.stem:<46} "
            f"{_format_number(params):>12} "
            f"{_format_number(flops):>12} "
            f"{_format_number(macs):>12}"
        )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
