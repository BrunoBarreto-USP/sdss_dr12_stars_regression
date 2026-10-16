"""Profile the PyTorch SpecT topology with TensorFlow's FLOP profiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetuning.spectral_transformer_keras import build_spectral_transformer_keras
from finetuning.checkpoint_io import load_checkpoint


def _checkpoint_architecture(path: Path) -> dict[str, int | float]:
    checkpoint = load_checkpoint(path)
    args = checkpoint["args"]
    return {
        "patch_size": args["patch_size"],
        "d_model": args["d_model"],
        "num_layers": args["layers"],
        "num_heads": args["heads"],
        "mlp_ratio": args["mlp_ratio"],
        "dropout": args["dropout"],
    }


def profile_model(model: tf.keras.Model, input_dim: int) -> int:
    @tf.function
    def forward_pass(features: tf.Tensor):
        return model(features, training=False)

    concrete = forward_pass.get_concrete_function(tf.TensorSpec([1, input_dim], tf.float32))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        profile = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=tf.compat.v1.RunMetadata(),
            cmd="op",
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),
        )
    _ = frozen_func
    return int(profile.total_float_ops) if profile is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch SpecT best_model.pt checkpoint.")
    args = parser.parse_args()

    architecture = _checkpoint_architecture(args.checkpoint)
    model = build_spectral_transformer_keras(input_dim=4000, **architecture)
    params = model.count_params()
    flops = profile_model(model, input_dim=4000)
    print("TensorFlow profiler, batch 1, input shape (1, 4000)")
    print("Architecture loaded from:", args.checkpoint)
    print(f"Keras-equivalent parameters: {params:,} ({params / 1e6:.6f} M)")
    print(f"FLOPs per spectrum: {flops:,} ({flops / 1e6:.6f} M)")
    print(f"MACs per spectrum: {flops / 2:,.0f} ({flops / 2e6:.6f} M)")


if __name__ == "__main__":
    main()
