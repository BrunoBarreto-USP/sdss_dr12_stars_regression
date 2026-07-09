"""Model architecture and Keras Tuner builders."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import keras_tuner as kt
import tensorflow as tf
from keras import layers
from keras_tuner.src.engine import tuner_utils


DEFAULT_OUTPUT_NAMES = ("teff_output", "feh_output", "logg_output")
DEFAULT_ARCHITECTURE = "baseline"
DEFAULT_TUNER_PROJECT_NAME = "stellar_multitask_branch_templates"
FIXED_TASK_HEAD_TEMPLATE = (32, 16)
TASK_HEAD_TEMPLATE_CHOICES: dict[str, tuple[int, ...]] = {
    "(16,)":        (16,),
    "(32,)":        (32,),
    "(48,)":        (48,),
    "(48, 32)":     (48, 32),
    "(48, 16)":     (48, 16),
    "(32, 16)":     (32, 16),
    "(48, 32, 16)": (48, 32, 16),
}
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_COSINE_ALPHA = 0.05
WARMUP_FRACTION = 0.05
REFERENCE_BATCH_SIZE = 256


def get_output_names(num_outputs: int) -> list[str]:
    if num_outputs == len(DEFAULT_OUTPUT_NAMES):
        return list(DEFAULT_OUTPUT_NAMES)
    return [f"output_{idx}" for idx in range(num_outputs)]



def _build_huber_loss(delta: float = DEFAULT_HUBER_DELTA) -> tf.keras.losses.Huber:
    return tf.keras.losses.Huber(delta=delta)


@tf.keras.utils.register_keras_serializable(package="stellar")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, initial_learning_rate, total_steps, warmup_steps, alpha=0.01):
        super().__init__()
        self.initial_learning_rate = float(initial_learning_rate)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(alpha)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        peak_lr = tf.cast(self.initial_learning_rate, tf.float32)
        alpha = tf.cast(self.alpha, tf.float32)

        warmup_lr = peak_lr * (step / tf.maximum(warmup, 1.0))

        decay_steps = tf.maximum(total - warmup, 1.0)
        decay_progress = tf.minimum((step - warmup) / decay_steps, 1.0)
        cosine_lr = alpha * peak_lr + 0.5 * (1.0 - alpha) * peak_lr * (
            1.0 + tf.cos(tf.constant(3.141592653589793) * decay_progress)
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }


@tf.keras.utils.register_keras_serializable(package="stellar")
class PerCycleWarmupCosineRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with restarts, where every cycle starts with a linear warm-up.

    The LR at the boundary between cycles is continuous: the warm-up of cycle
    n+1 starts exactly where the cosine floor of cycle n ends, so there is no
    abrupt spike.

    Within each cycle the schedule is:
      warm-up phase (local_frac < warmup_fraction):
          LR = cycle_peak * (alpha + (1-alpha) * local_frac / warmup_fraction)
      cosine phase (local_frac >= warmup_fraction):
          LR = alpha*cycle_peak + 0.5*(1-alpha)*cycle_peak*(1+cos(π*cosine_frac))
    where cycle_peak = initial_lr * m_mul^n for cycle n.
    """

    def __init__(
        self,
        initial_lr: float,
        first_decay_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.01,
        warmup_fraction: float = 0.05,
    ):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.first_decay_steps = int(first_decay_steps)
        self.t_mul = float(t_mul)
        self.m_mul = float(m_mul)
        self.alpha = float(alpha)
        self.warmup_fraction = float(warmup_fraction)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        T = tf.cast(self.first_decay_steps, tf.float32)
        t_mul = tf.cast(self.t_mul, tf.float32)
        m_mul = tf.cast(self.m_mul, tf.float32)
        alpha = tf.cast(self.alpha, tf.float32)
        warmup_frac = tf.cast(self.warmup_fraction, tf.float32)
        initial_lr = tf.cast(self.initial_lr, tf.float32)

        completed = step / tf.maximum(T, 1.0)

        # Cycle index n and local fraction within the cycle.
        # For t_mul != 1: cumulative steps at start of cycle n = T*(t_mul^n-1)/(t_mul-1).
        # Solving for n: n = floor(log(completed*(t_mul-1)+1) / log(t_mul)).
        i = tf.floor(
            tf.math.log(tf.maximum(completed * (t_mul - 1.0) + 1.0, 1e-8))
            / tf.math.log(tf.maximum(t_mul, 1.0 + 1e-8))
        )
        t_mul_pow_i = tf.pow(t_mul, i)
        cycle_start = (t_mul_pow_i - 1.0) / (t_mul - 1.0)
        local_frac = tf.minimum((completed - cycle_start) / t_mul_pow_i, 1.0)

        cycle_peak = initial_lr * tf.pow(m_mul, i)

        # Warm-up: linear from alpha*cycle_peak up to cycle_peak.
        warmup_lr = cycle_peak * (
            alpha + (1.0 - alpha) * local_frac / tf.maximum(warmup_frac, 1e-8)
        )

        # Cosine decay: from cycle_peak down to alpha*cycle_peak.
        cosine_frac = tf.minimum(
            (local_frac - warmup_frac) / tf.maximum(1.0 - warmup_frac, 1e-8), 1.0
        )
        cosine_lr = alpha * cycle_peak + 0.5 * (1.0 - alpha) * cycle_peak * (
            1.0 + tf.cos(tf.constant(3.141592653589793) * cosine_frac)
        )

        return tf.where(local_frac < warmup_frac, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
            "alpha": self.alpha,
            "warmup_fraction": self.warmup_fraction,
        }


def resnet_block(x, units: int, dropout_rate: float = 0.0):
    """Dense residual block with pre-norm.

    Pre-norm ordering (LN before Dense) improves gradient flow for deeper
    networks.  The Add is left bare so the identity shortcut is preserved
    and gradients flow unimpeded back through the residual connections.
    """
    shortcut = x

    # Pre-norm residual path
    x = layers.LayerNormalization()(x)
    x = layers.Dense(units)(x)
    x = layers.Activation("gelu")(x)

    x = layers.LayerNormalization()(x)
    x = layers.Dense(units)(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    # Dimension-matching shortcut projection.
    # LN before the Dense ensures the shortcut arrives at the Add at the same
    # scale as the residual path (which passed through two LN+Dense+GELU rounds).
    if shortcut.shape[-1] != units:
        shortcut = layers.LayerNormalization()(shortcut)
        shortcut = layers.Dense(units)(shortcut)

    x = layers.Add()([x, shortcut])

    return x


def _normalize_loss_weights(
    loss_weights: Mapping[str, float] | Sequence[float] | None,
    output_names: Sequence[str],
) -> dict[str, float] | None:
    if loss_weights is None:
        return None
    if isinstance(loss_weights, Mapping):
        return {name: float(loss_weights[name]) for name in output_names}

    weights = list(loss_weights)
    if len(weights) != len(output_names):
        raise ValueError(
            "loss_weights must have the same length as the model outputs.",
        )
    return {name: float(weight) for name, weight in zip(output_names, weights)}


def _apply_dense_stack(
    x,
    units_stack: Sequence[int],
    *,
    dropout_rate: float,
    name_prefix: str,
):
    units_list = list(units_stack)
    last_idx = len(units_list) - 1
    for layer_idx, units in enumerate(units_list):
        x = layers.Dense(
            units,
            activation="gelu",
            name=f"{name_prefix}_dense_{layer_idx}",
        )(x)
        if dropout_rate > 0 and layer_idx < last_idx:
            x = layers.Dropout(
                dropout_rate,
                name=f"{name_prefix}_dropout_{layer_idx}",
            )(x)
    return x


def _build_input_stem(inputs, *, hp=None, dropout_rate: float = 0.0):
    """Dense stem for rest-frame aligned spectral processing."""

    # Dense projection
    if hp is None:
        units = 320
    else:
        units = hp.Choice("initial_units", [64, 96, 128], default=96)

    x = layers.Dense(units,name="input_stem_dense",)(inputs)
    x = layers.LayerNormalization(name="input_stem_ln")(x)
    x = layers.Activation("gelu", name="input_stem_activation")(x)

    stem_dropout = dropout_rate if hp is None else hp.Float(
        "initial_dropout",
        0.0,
        0.5,
        step=0.1,
        default=0.1,
    )
    if stem_dropout > 0:
        x = layers.Dropout(stem_dropout, name="input_stem_dropout")(x)

    return x, units


def _build_baseline_shared_trunk(inputs, *, hp=None):
    stem, _ = _build_input_stem(inputs, hp=hp, dropout_rate=0.0)

    if hp is None:
        num_blocks = 2
        block_units = 48
        trunk_dropout_rate = 0.2
    else:
        num_blocks = hp.Int("shared_blocks", min_value=1, max_value=2, default=1)
        block_units = hp.Choice("block_units", [32, 48, 64], default=48)
        trunk_dropout_rate = hp.Float("trunk_dropout", 0.0, 0.4, step=0.1, default=0.2)

    x = stem
    for _ in range(num_blocks):
        x = resnet_block(x, block_units, dropout_rate=trunk_dropout_rate)

    # Pre-Norm networks require a final normalization before downstream linear/sigmoid heads
    x = layers.LayerNormalization(name="trunk_final_ln")(x)

    return x, block_units


def _build_task_output(
    task_input,
    feature_dim: int,
    output_name: str,
    task_idx: int,
    *,
    hp=None,
    dropout_rate: float = 0.0,
):
    head = task_input

    # Head depth and width are tunable: 1-3 layer templates.
    if hp is None:
        head_template = FIXED_TASK_HEAD_TEMPLATE
        head_dropout = dropout_rate
    else:
        template_key = hp.Choice(
            f"{output_name}_head_template",
            list(TASK_HEAD_TEMPLATE_CHOICES.keys()),
            default="(32, 16)",
        )
        head_template = TASK_HEAD_TEMPLATE_CHOICES[template_key]
        head_dropout = hp.Float(f"{output_name}_head_dropout", 0.0, 0.4, step=0.1, default=0.1)

    head = _apply_dense_stack(
        head,
        head_template,
        dropout_rate=head_dropout,
        name_prefix=f"{output_name}_head",
    )
    return layers.Dense(1, activation="linear", name=output_name, dtype="float32")(head)


def _build_optimizer(*, hp=None, total_steps=None, warm_restart_steps=None,
                     warmup_fraction: float = WARMUP_FRACTION,
                     cosine_alpha: float = DEFAULT_COSINE_ALPHA,
                     use_lr_schedule: bool = True,
                     batch_size: int = REFERENCE_BATCH_SIZE):
    import math
    lr_scale = math.sqrt(batch_size / REFERENCE_BATCH_SIZE)

    if hp is None:
        learning_rate = 1e-3 * lr_scale
        weight_decay = 1e-4
    else:
        base_lr = hp.Float("learning_rate", 1e-4, 5e-3, sampling="log", default=1e-3)
        learning_rate = base_lr * lr_scale
        weight_decay = hp.Float("weight_decay", 1e-5, 1e-2, sampling="log", default=1e-4)

    peak_lr = learning_rate  # scalar at this point

    if use_lr_schedule and warm_restart_steps:
        learning_rate = PerCycleWarmupCosineRestarts(
            peak_lr,
            first_decay_steps=warm_restart_steps,
            t_mul=2.0,
            m_mul=1.0,
            alpha=cosine_alpha,
            warmup_fraction=warmup_fraction,
        )

    elif use_lr_schedule and total_steps:
        warmup_steps = int(total_steps * warmup_fraction)
        if warmup_steps > 0:
            learning_rate = WarmupCosineDecay(
                peak_lr,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                alpha=cosine_alpha,
            )
        else:
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                peak_lr,
                total_steps,
                alpha=cosine_alpha,
            )

    return tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        global_clipnorm=5.0,
    )


def _compile_multitask_model(
    model: tf.keras.Model,
    *,
    output_names: Sequence[str],
    optimizer,
    loss_weights: Mapping[str, float] | Sequence[float] | None,
    jit_compile: bool = False,
):
    normalized_loss_weights = _normalize_loss_weights(loss_weights, output_names)
    loss = {name: _build_huber_loss() for name in output_names}
    metrics = {name: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for name in output_names}
    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=normalized_loss_weights,
        metrics=metrics,
        jit_compile=jit_compile,
    )
    return model


def compile_huber_finetune(
    model: tf.keras.Model,
    *,
    output_names: Sequence[str],
    loss_weights: Mapping[str, float] | Sequence[float] | None,
    learning_rate: float = 1e-4,
    jit_compile: bool = False,
):
    """Recompile *model* for a short Huber fine-tune stage."""
    normalized_loss_weights = _normalize_loss_weights(loss_weights, output_names)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.0,
        global_clipnorm=5.0,
    )
    loss = {name: _build_huber_loss() for name in output_names}
    metrics = {name: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for name in output_names}
    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=normalized_loss_weights,
        metrics=metrics,
        jit_compile=jit_compile,
    )
    return model


def compile_mae_finetune(
    model: tf.keras.Model,
    *,
    output_names: Sequence[str],
    loss_weights: Mapping[str, float] | Sequence[float] | None,
    learning_rate: float = 1e-4,
    jit_compile: bool = False,
):
    """Backward-compatible alias for ``compile_huber_finetune``."""
    return compile_huber_finetune(
        model,
        output_names=output_names,
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        jit_compile=jit_compile,
    )


class LightBayesianOptimization(kt.BayesianOptimization):
    """Bayesian tuner that skips per-trial weight saving (only HPs are tracked)."""

    def run_trial(self, trial, *args, **kwargs):
        original_callbacks = kwargs.pop("callbacks", [])

        histories = []
        for execution in range(self.executions_per_trial):
            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            copied_kwargs["callbacks"] = callbacks
            obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
            histories.append(obj_value)
        return histories


def build_model(
    input_shape,
    num_outputs: int = 3,
    dropout_rate: float = 0.3,
    hp=None,
    loss_weights: Mapping[str, float] | Sequence[float] | None = None,
    total_steps=None,
    warm_restart_steps=None,
    cosine_alpha: float = DEFAULT_COSINE_ALPHA,
    use_lr_schedule: bool = True,
    jit_compile: bool = False,
    batch_size: int = REFERENCE_BATCH_SIZE,
    **_ignored,
):
    if isinstance(input_shape, (tuple, list)):
        input_shape_tuple = tuple(input_shape)
    else:
        input_shape_tuple = (input_shape,)

    output_names = get_output_names(num_outputs)
    inputs = layers.Input(shape=input_shape_tuple)

    shared, feature_dim = _build_baseline_shared_trunk(inputs, hp=hp)

    outputs = [
        _build_task_output(
            shared,
            feature_dim,
            output_name,
            task_idx,
            hp=hp,
            dropout_rate=dropout_rate,
        )
        for task_idx, output_name in enumerate(output_names)
    ]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stellar_model")
    optimizer = _build_optimizer(
        hp=hp,
        total_steps=total_steps,
        warm_restart_steps=warm_restart_steps,
        cosine_alpha=cosine_alpha,
        use_lr_schedule=use_lr_schedule,
        batch_size=batch_size,
    )
    return _compile_multitask_model(
        model,
        output_names=output_names,
        optimizer=optimizer,
        loss_weights=loss_weights,
        jit_compile=jit_compile,
    )


def build_export_model(model: tf.keras.Model) -> tf.keras.Model:
    """Create a plain Functional model for serialization and inference."""
    base_model = getattr(model, "base_model", model)
    return tf.keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.outputs,
        name=base_model.name,
    )


def build_bayesian_tuner(
    input_shape,
    num_outputs: int = 3,
    max_trials: int = 7,
    executions_per_trial: int = 1,
    directory: str = "keras_tuner",
    project_name: str | None = None,
    loss_weights: Mapping[str, float] | Sequence[float] | None = None,
    total_steps=None,
    warm_restart_steps=None,
    cosine_alpha: float = DEFAULT_COSINE_ALPHA,
    use_lr_schedule: bool = True,
    objective_name: str = "val_loss",
    jit_compile: bool = False,
    batch_size: int = REFERENCE_BATCH_SIZE,
    **_ignored,
):
    if project_name is None:
        project_name = DEFAULT_TUNER_PROJECT_NAME

    def model_builder(hp):
        return build_model(
            input_shape=input_shape,
            num_outputs=num_outputs,
            hp=hp,
            loss_weights=loss_weights,
            total_steps=total_steps,
            warm_restart_steps=warm_restart_steps,
            cosine_alpha=cosine_alpha,
            use_lr_schedule=use_lr_schedule,
            jit_compile=jit_compile,
            batch_size=batch_size,
        )

    objective = kt.Objective(objective_name, direction="min")
    return LightBayesianOptimization(
        hypermodel=model_builder,
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project_name,
        overwrite=False,
    )
