"""Keras implementation of the PyTorch SpectralTransformerRegressor topology.

This module mirrors the architecture for framework-consistent complexity
profiling. It does not import or convert trained PyTorch weights because FLOP
counts depend on tensor shapes and operations, not parameter values.
"""

from __future__ import annotations

import tensorflow as tf


def build_spectral_transformer_keras(
    *,
    input_dim: int = 4000,
    patch_size: int = 16,
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    output_dim: int = 3,
) -> tf.keras.Model:
    """Build the pre-norm 1-D ViT used by the PyTorch SpecT implementation."""
    if input_dim % patch_size:
        raise ValueError("input_dim must be divisible by patch_size.")
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads.")

    num_patches = input_dim // patch_size
    inputs = tf.keras.Input(shape=(input_dim,), name="stellar_input")
    x = tf.keras.layers.Reshape((input_dim, 1), name="add_channel")(inputs)
    x = tf.keras.layers.Conv1D(
        filters=d_model,
        kernel_size=patch_size,
        strides=patch_size,
        name="patch_embed",
    )(x)

    cls_ids = tf.keras.layers.Lambda(
        lambda tokens: tf.zeros((tf.shape(tokens)[0], 1), dtype=tf.int32),
        name="cls_token_ids",
    )(x)
    cls_token = tf.keras.layers.Embedding(1, d_model, name="cls_token")(cls_ids)
    x = tf.keras.layers.Concatenate(axis=1, name="prepend_cls_token")([cls_token, x])
    position_ids = tf.keras.layers.Lambda(
        lambda tokens: tf.expand_dims(tf.range(tf.shape(tokens)[1]), axis=0),
        name="position_ids",
    )(x)
    position_embeddings = tf.keras.layers.Embedding(
        num_patches + 1,
        d_model,
        name="pos_embed",
    )(position_ids)
    x = tf.keras.layers.Add(name="add_position_embeddings")([x, position_embeddings])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="input_norm")(x)

    feedforward_dim = int(d_model * mlp_ratio)
    for index in range(num_layers):
        residual = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"encoder_{index}_norm1")(x)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"encoder_{index}_self_attention",
        )(x, x)
        attention = tf.keras.layers.Dropout(dropout, name=f"encoder_{index}_attention_dropout")(attention)
        x = tf.keras.layers.Add(name=f"encoder_{index}_attention_add")([residual, attention])

        residual = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"encoder_{index}_norm2")(x)
        x = tf.keras.layers.Dense(feedforward_dim, activation="gelu", name=f"encoder_{index}_linear1")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"encoder_{index}_ffn_dropout1")(x)
        x = tf.keras.layers.Dense(d_model, name=f"encoder_{index}_linear2")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"encoder_{index}_ffn_dropout2")(x)
        x = tf.keras.layers.Add(name=f"encoder_{index}_ffn_add")([residual, x])

    x = tf.keras.layers.Lambda(lambda tokens: tokens[:, 0], name="select_cls_token")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="head_norm")(x)
    x = tf.keras.layers.Dense(d_model, activation="gelu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(output_dim, dtype="float32", name="stellar_outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="spectral_transformer_regressor")
