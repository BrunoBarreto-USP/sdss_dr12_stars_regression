"""Multitask target helpers, augmentation, and PCA-neighbour mixup tools."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def prepare_multitask_dicts(
    y_train_targets: np.ndarray,
    y_val_targets: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Prepare float32 dict labels for multitask training."""
    y_train_dict = {
        "teff_output": np.asarray(y_train_targets[:, 0], dtype=np.float32),
        "feh_output": np.asarray(y_train_targets[:, 1], dtype=np.float32),
        "logg_output": np.asarray(y_train_targets[:, 2], dtype=np.float32),
    }
    y_val_dict = {
        "teff_output": np.asarray(y_val_targets[:, 0], dtype=np.float32),
        "feh_output": np.asarray(y_val_targets[:, 1], dtype=np.float32),
        "logg_output": np.asarray(y_val_targets[:, 2], dtype=np.float32),
    }
    return y_train_dict, y_val_dict


def augment_training_data(
    X_train: np.ndarray,
    y_train_dict: dict[str, np.ndarray],
    *,
    aug_factor: int = 3,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Offline dataset augmentation by additive Gaussian noise.

    Produces a dataset of size ``aug_factor * N`` by stacking the original
    samples with ``aug_factor - 1`` noisy copies.  Targets are duplicated
    unchanged — noise is applied to features only.

    Parameters
    ----------
    X_train:
        Feature array of shape ``(N, n_features)``.
    y_train_dict:
        Dict mapping output names to 1-D target arrays of length ``N``.
    aug_factor:
        Total size multiplier.  ``aug_factor=3`` yields the original plus
        two noisy copies, for a dataset 3× larger.
    noise_level:
        Noise magnitude as a fraction of each sample's own standard
        deviation (e.g. ``0.03`` = 3 %).
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    X_aug : np.ndarray, shape ``(aug_factor * N, n_features)``
    y_aug_dict : dict with same keys, each value shape ``(aug_factor * N,)``
    """
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    n = len(X_train)

    per_sample_std = X_train.std(axis=1, keepdims=True)              # (N, 1)
    X_parts = [X_train]
    for _ in range(aug_factor - 1):
        # Draw a different noise fraction per copy, uniformly in [0.01, noise_level]
        level = rng.uniform(0.01, noise_level)
        noise = rng.standard_normal(X_train.shape).astype(np.float32)
        X_parts.append(X_train + level * per_sample_std * noise)

    X_aug = np.concatenate(X_parts, axis=0)                          # (aug_factor*N, F)

    y_aug_dict = {
        key: np.tile(arr.astype(np.float32), aug_factor)
        for key, arr in y_train_dict.items()
    }

    # Shuffle so original and augmented samples are interleaved
    idx = rng.permutation(aug_factor * n)
    X_aug = X_aug[idx]
    y_aug_dict = {key: arr[idx] for key, arr in y_aug_dict.items()}

    print(
        f"Augmentation: {n} -> {aug_factor * n} samples "
        f"(aug_factor={aug_factor}, noise_level=U[0.01, {noise_level}])"
    )
    return X_aug, y_aug_dict


def save_pre_model_dataset(
    output_path: str,
    *,
    X_train_features: np.ndarray,
    y_train_targets: np.ndarray,
    X_val_features: np.ndarray,
    y_val_targets: np.ndarray,
    X_test_features: np.ndarray,
    y_test_targets: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    target_cols: np.ndarray,
    target_names: np.ndarray,
) -> None:
    """Save the final processed dataset right before model definition.

    Uses compressed NPZ to reduce disk usage.
    """
    np.savez_compressed(
        output_path,
        X_train_features=X_train_features,
        y_train_targets=y_train_targets,
        X_val_features=X_val_features,
        y_val_targets=y_val_targets,
        X_test_features=X_test_features,
        y_test_targets=y_test_targets,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        target_cols=target_cols,
        target_names=target_names,
    )
    print(f"Saved compressed pre-model dataset to {output_path}")


def pca_neighbor_mixup(
    X_train: np.ndarray,
    *,
    sample_idx: int,
    y_train_targets: np.ndarray | None = None,
    target_scalers: list | tuple | None = None,
    target_names: list[str] | tuple[str, ...] | None = None,
    n_neighbors: int = 10,
    n_components: int = 2,
    n_direction_samples: int = 720,
    random_state: int = 42,
    plot: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Generate a synthetic spectrum from PCA-neighbour geometry.

    Steps
    -----
    1. Fit PCA on the training set and project all spectra into the first
       ``n_components`` PCs.
    2. Find the ``n_neighbors`` nearest neighbours of ``sample_idx`` in that PCA space.
    3. Define a search radius as the mean distance from the anchor point to those neighbours.
    4. Search directions on that radius and select the point that maximizes the
       mean distance to the anchor's neighbours.
    5. Build a synthetic spectrum as the inverse-distance-weighted average of the
       neighbour spectra, where distances are measured from the synthetic PCA point
       to each neighbour in the same PCA space.

    Notes
    -----
    The geometric calculations use all ``n_components`` dimensions.  The plotting
    helper still visualizes only the first two PCA dimensions.

    Returns a dict with all intermediate values so the caller can inspect or reuse
    the generated spectrum.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2-D array of shape (n_samples, n_features).")
    if not (0 <= sample_idx < len(X_train)):
        raise IndexError("sample_idx is out of bounds.")
    if n_components < 2:
        raise ValueError("n_components must be >= 2 so the geometry can still be visualized on the first two PCs.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    if y_train_targets is not None:
        y_train_targets = np.asarray(y_train_targets, dtype=np.float32)
        if y_train_targets.ndim != 2 or len(y_train_targets) != len(X_train):
            raise ValueError("y_train_targets must have shape (n_samples, n_targets) and match X_train.")
        if target_scalers is not None and len(target_scalers) != y_train_targets.shape[1]:
            raise ValueError("target_scalers must have one entry per target column.")
        if target_names is not None and len(target_names) != y_train_targets.shape[1]:
            raise ValueError("target_names must have one entry per target column.")

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_train)

    n_query = min(n_neighbors + 1, len(X_train))
    nbrs = NearestNeighbors(n_neighbors=n_query, metric="euclidean")
    nbrs.fit(X_pca)

    anchor_point = X_pca[sample_idx]
    distances_all, indices_all = nbrs.kneighbors(anchor_point.reshape(1, -1))
    distances_all = distances_all[0]
    indices_all = indices_all[0]

    # Drop the anchor itself when present.
    if indices_all[0] == sample_idx:
        neighbor_indices = indices_all[1:]
        neighbor_distances = distances_all[1:]
    else:
        neighbor_indices = indices_all[:n_neighbors]
        neighbor_distances = distances_all[:n_neighbors]

    if len(neighbor_indices) == 0:
        raise ValueError("Need at least one neighbour distinct from the anchor sample.")

    radius = float(np.mean(neighbor_distances))
    neighbor_points = X_pca[neighbor_indices]

    angles = np.linspace(0.0, 2.0 * np.pi, n_direction_samples, endpoint=False)
    candidate_offsets = np.zeros((n_direction_samples, n_components), dtype=np.float32)
    candidate_offsets[:, 0] = np.cos(angles) * radius
    candidate_offsets[:, 1] = np.sin(angles) * radius
    candidate_points = anchor_point + candidate_offsets

    # Maximize the mean distance from the synthetic PCA point to the anchor neighbours.
    candidate_to_neighbors = np.linalg.norm(
        candidate_points[:, None, :] - neighbor_points[None, :, :],
        axis=2,
    )
    direction_scores = candidate_to_neighbors.mean(axis=1)
    best_direction_idx = int(np.argmax(direction_scores))
    synthetic_point = candidate_points[best_direction_idx]
    synthetic_to_neighbors = candidate_to_neighbors[best_direction_idx]

    eps = 1e-8
    weights = 1.0 / np.maximum(synthetic_to_neighbors, eps)
    weights /= weights.sum()
    generated_spectrum = np.sum(X_train[neighbor_indices] * weights[:, None], axis=0).astype(np.float32)

    anchor_params = None
    neighbor_params = None
    generated_params = None
    if y_train_targets is not None:
        unscaled_targets = y_train_targets.astype(np.float32).copy()
        if target_scalers is not None:
            for col, scaler in enumerate(target_scalers):
                unscaled_targets[:, col] = scaler.inverse_transform(y_train_targets[:, [col]]).ravel()
        anchor_params = unscaled_targets[sample_idx].astype(np.float32)
        neighbor_params = unscaled_targets[neighbor_indices].astype(np.float32)
        generated_params = np.sum(neighbor_params * weights[:, None], axis=0).astype(np.float32)
        if target_names is None:
            target_names = tuple(f"target_{i}" for i in range(unscaled_targets.shape[1]))

    result: dict[str, np.ndarray | float | int] = {
        "sample_idx": int(sample_idx),
        "anchor_point": anchor_point.astype(np.float32),
        "neighbor_indices": neighbor_indices.astype(np.int32),
        "neighbor_distances": neighbor_distances.astype(np.float32),
        "neighbor_points": neighbor_points.astype(np.float32),
        "radius": radius,
        "angles": angles.astype(np.float32),
        "candidate_points": candidate_points.astype(np.float32),
        "direction_scores": direction_scores.astype(np.float32),
        "best_direction_idx": best_direction_idx,
        "synthetic_point": synthetic_point.astype(np.float32),
        "synthetic_to_neighbors": synthetic_to_neighbors.astype(np.float32),
        "weights": weights.astype(np.float32),
        "generated_spectrum": generated_spectrum,
        "anchor_spectrum": X_train[sample_idx].astype(np.float32),
        "neighbor_spectra": X_train[neighbor_indices].astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "X_pca": X_pca.astype(np.float32),
        "n_components": int(n_components),
    }
    if anchor_params is not None and neighbor_params is not None and generated_params is not None:
        result["anchor_params"] = anchor_params
        result["neighbor_params"] = neighbor_params
        result["generated_params"] = generated_params
        result["target_names"] = np.asarray(target_names, dtype=object)

    if plot:
        plot_pca_neighbor_mixup(result)

    return result


def plot_pca_neighbor_mixup(
    result: dict[str, np.ndarray | float | int],
    *,
    n_spectra_to_overlay: int | None = None,
) -> None:
    """Plot only the generated 4000-D spectrum against anchor and neighbours."""
    neighbor_indices = np.asarray(result["neighbor_indices"], dtype=np.int32)
    weights = np.asarray(result["weights"], dtype=np.float32)
    generated_spectrum = np.asarray(result["generated_spectrum"], dtype=np.float32)
    anchor_spectrum = np.asarray(result["anchor_spectrum"], dtype=np.float32)
    neighbor_spectra = np.asarray(result["neighbor_spectra"], dtype=np.float32)
    sample_idx = int(result["sample_idx"])

    if n_spectra_to_overlay is None:
        n_spectra_to_overlay = len(neighbor_indices)
    n_spectra_to_overlay = min(n_spectra_to_overlay, len(neighbor_indices))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    feature_axis = np.arange(anchor_spectrum.shape[0])
    for i in range(n_spectra_to_overlay):
        idx = neighbor_indices[i]
        ax.plot(
            feature_axis,
            neighbor_spectra[i],
            lw=0.9,
            alpha=0.55,
            label=f"Neighbour #{idx} (w={weights[i]:.3f})",
        )
    ax.plot(feature_axis, anchor_spectrum, lw=2.0, color="tab:blue", label=f"Anchor #{sample_idx}")
    ax.plot(feature_axis, generated_spectrum, lw=2.0, color="tab:red", label="Generated spectrum")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Flux")
    ax.set_title("Generated spectrum vs anchor and neighbours")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    if "anchor_params" in result and "neighbor_params" in result and "generated_params" in result:
        target_names = np.asarray(result["target_names"], dtype=object)
        anchor_params = np.asarray(result["anchor_params"], dtype=np.float32)
        neighbor_params = np.asarray(result["neighbor_params"], dtype=np.float32)
        generated_params = np.asarray(result["generated_params"], dtype=np.float32)

        print("Unscaled parameters:")
        print(
            "  Anchor "
            f"#{sample_idx}: "
            + ", ".join(f"{name}={value:.4f}" for name, value in zip(target_names, anchor_params))
        )
        for i in range(n_spectra_to_overlay):
            idx = neighbor_indices[i]
            print(
                "  Neighbour "
                f"#{idx} (w={weights[i]:.4f}): "
                + ", ".join(f"{name}={value:.4f}" for name, value in zip(target_names, neighbor_params[i]))
            )
        print(
            "  Generated (weighted avg): "
            + ", ".join(f"{name}={value:.4f}" for name, value in zip(target_names, generated_params))
        )
