from __future__ import annotations

import importlib
import json
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .nts_adapter import load_sequences, make_lag_windows


OUTCOME_FEATURES = {
    "transactions",
    "quantity",
    "spend",
    "other_discount",
    "coupon_discount",
    "unique_items",
    "unique_categories",
    "redemptions_reported",
    "is_click", "is_like", "long_view",
}
ASSIGNMENT_FEATURES = {"offers_started", "unique_coupons_started", "is_rand"}


def load_official_implementation(source: str | Path) -> Any:
    """Import the pinned official NTS-NOTEARS implementation without vendoring it."""
    module_dir = Path(source).resolve() / "notears"
    required = module_dir / "nts_notears.py"
    if not required.exists():
        raise FileNotFoundError(
            f"official NTS-NOTEARS source not found at {required}; see scripts/get_nts_notears.py"
        )
    module_text = str(module_dir)
    if module_text not in sys.path:
        sys.path.insert(0, module_text)

    # The paper code imports plotting helpers (and therefore python-igraph) at
    # module load time, although training does not use them. Keep that optional
    # visualization dependency out of the core fitting path.
    previous_utils = sys.modules.get("utils")
    sys.modules["utils"] = types.ModuleType("utils")
    try:
        return importlib.import_module("nts_notears")
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils


def default_prior_knowledge(features: list[str]) -> list[dict[str, object]]:
    """Forbid same-week outcomes from causing campaign-start assignment variables."""
    return [
        {
            "from_node": source,
            "from_lag": 0,
            "to_node": target,
            "lower_bound": 0,
            "upper_bound": 0,
        }
        for source in features
        if source in OUTCOME_FEATURES
        for target in features
        if target in ASSIGNMENT_FEATURES
    ]


def forward_windows(model: Any, windows: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run the official CNN on independent [batch, feature, lag+target] windows."""
    if windows.ndim != 3 or windows.shape[1] != model.dims[0]:
        raise ValueError("windows must have shape [batch, feature, lag+target]")
    if windows.shape[2] != model.kernal_size:
        raise ValueError("window width must equal model kernel size")

    hidden = model.conv1d_pos.to(device)(windows) - model.conv1d_neg.to(device)(windows)
    hidden = hidden.squeeze(dim=2).view(-1, model.dims[0], model.dims[1])
    for layer in model.fc2.to(device):
        hidden = layer(torch.sigmoid(hidden))
    return hidden.squeeze(dim=2)


def _train_windows(
    official: Any,
    model: Any,
    windows: np.ndarray,
    *,
    lambda1: float,
    lambda2: float,
    threshold: float,
    max_iter: int,
    h_tol: float,
    rho_max: float,
) -> tuple[np.ndarray, list[float]]:
    """Multi-sequence adaptation of the official augmented-Lagrangian loop."""
    device = torch.device("cpu")
    tensor = torch.from_numpy(windows).to(device=device, dtype=torch.float64)
    target = tensor[:, :, -1]
    rho, alpha, h = 1.0, 0.0, np.inf
    history: list[float] = []

    for _ in range(max_iter):
        optimizer = official.LBFGSBScipy(model.parameters())
        optimizer.assign_bounds(model)
        h_new = np.inf
        while rho < rho_max:
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                estimate = forward_windows(model, tensor, device)
                loss = 0.5 / len(target) * torch.sum((estimate - target) ** 2)
                h_value = model.h_func()
                objective = (
                    loss
                    + 0.5 * rho * h_value * h_value
                    + alpha * h_value
                    + 0.5 * lambda2 * model.l2_reg()
                    + lambda1 * model.fc1_l1_reg()
                )
                objective.backward()
                return objective

            optimizer.step(closure)
            with torch.no_grad():
                h_new = float(model.h_func().item())
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        h = h_new
        history.append(h)
        if h <= h_tol or rho >= rho_max:
            break

    if h > h_tol:
        warnings.warn(f"NTS-NOTEARS did not reach h_tol: h={h:.3e}, h_tol={h_tol:.3e}")

    simultaneous, lagged = model.fc1_to_adj()
    simultaneous[np.abs(simultaneous) < threshold] = 0
    lagged[np.abs(lagged) < threshold] = 0
    d = len(simultaneous)
    full = np.zeros((d * model.kernal_size, d * model.kernal_size))
    full[:, -d:] = np.vstack((lagged, simultaneous))
    return full, history


def adjacency_to_edges(
    adjacency: np.ndarray, features: list[str], max_lag: int
) -> list[dict[str, object]]:
    d = len(features)
    expected = d * (max_lag + 1)
    if adjacency.shape != (expected, expected):
        raise ValueError(f"adjacency must have shape {(expected, expected)}")
    edges = []
    for block in range(max_lag + 1):
        lag = max_lag - block
        for source in range(d):
            for target in range(d):
                weight = float(adjacency[block * d + source, max_lag * d + target])
                if weight != 0:
                    edges.append(
                        {
                            "source": features[source],
                            "target": features[target],
                            "lag": lag,
                            "weight": weight,
                        }
                    )
    return edges


def instantaneous_is_dag(adjacency: np.ndarray, feature_count: int) -> bool:
    graph = adjacency[-feature_count:, -feature_count:] != 0
    indegree = graph.sum(axis=0).astype(int)
    queue = list(np.flatnonzero(indegree == 0))
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == feature_count


def normalization_parameters(values: np.ndarray, preserve_binary: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    constant = scale == 0
    scale[constant] = 1.0
    if preserve_binary:
        binary = np.isin(values, [0.0, 1.0]).all(axis=0)
        mean[binary] = 0.0
        scale[binary] = 1.0
    return mean, scale, constant


def bootstrap_user_indices(user_ids: np.ndarray, user_count: int, seed: int) -> np.ndarray:
    if user_ids.ndim != 1 or not len(user_ids) or user_count < 1:
        raise ValueError('user_ids must be a non-empty vector and user_count positive')
    unique, inverse = np.unique(user_ids, return_inverse=True)
    selected = np.random.default_rng(seed).choice(len(unique), size=user_count, replace=True)
    return np.concatenate([np.flatnonzero(inverse == user) for user in selected])


def fit_nts_notears(
    sequence_file: str | Path,
    output: str | Path,
    official_source: str | Path,
    *,
    max_lag: int = 2,
    hidden: int = 10,
    lambda1: float = 0.005,
    lambda2: float = 0.01,
    threshold: float = 0.3,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    max_sequences: int | None = None,
    drop_features: list[str] | None = None,
    seed: int = 2026,
    preserve_binary: bool = False,
    forbid_instantaneous: bool = False,
    bootstrap_users: int | None = None,
) -> dict[str, object]:
    sequences, features = load_sequences(sequence_file)
    if drop_features:
        unknown = set(drop_features) - set(features)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        keep = [index for index, name in enumerate(features) if name not in drop_features]
        if not keep:
            raise ValueError("cannot drop every feature")
        sequences = [sequence[:, keep] for sequence in sequences]
        features = [features[index] for index in keep]
    if max_sequences is not None and bootstrap_users is not None:
        raise ValueError('cannot combine episode sampling with user bootstrap')
    if bootstrap_users is not None:
        with np.load(sequence_file, allow_pickle=False) as data:
            if 'user_ids' not in data or len(data['user_ids']) != len(sequences):
                raise ValueError('user_ids must match sequence count')
            selected = bootstrap_user_indices(data['user_ids'], bootstrap_users, seed)
        sequences = [sequences[index] for index in selected]
    elif max_sequences is not None:
        if max_sequences < 1:
            raise ValueError("max_sequences must be positive")
        selected = np.random.default_rng(seed).choice(
            len(sequences), size=min(max_sequences, len(sequences)), replace=False
        )
        sequences = [sequences[index] for index in selected]
    values = np.concatenate(sequences)
    mean, scale, constant = normalization_parameters(values, preserve_binary)
    normalized = [(sequence - mean) / scale for sequence in sequences]
    windows, sequence_ids = make_lag_windows(normalized, max_lag)

    np.random.seed(seed)
    torch.manual_seed(seed)
    official = load_official_implementation(official_source)
    prior = default_prior_knowledge(features)
    if forbid_instantaneous:
        prior += [{'from_node': source, 'from_lag': 0, 'to_node': target,
                   'lower_bound': 0, 'upper_bound': 0}
                  for source in features for target in features if source != target]
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = official.NTS_NOTEARS(
            dims=[len(features), hidden, 1],
            bias=True,
            number_of_lags=max_lag,
            prior_knowledge=prior,
            variable_names_no_time=features,
        ).double()
        adjacency, history = _train_windows(
            official,
            model,
            windows,
            lambda1=lambda1,
            lambda2=lambda2,
            threshold=threshold,
            max_iter=max_iter,
            h_tol=h_tol,
            rho_max=rho_max,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "nts_dag.npz",
        adjacency=adjacency,
        features=np.asarray(features),
        mean=mean,
        scale=scale,
    )
    result: dict[str, object] = {
        "method": "NTS-NOTEARS multi-sequence window adaptation",
        "official_source_commit": "e1ba34b784f9eec3d2139904589d2fd1ed4ea7ad",
        "sequence_count": len(sequences),
        "window_count": len(windows),
        "max_lag": max_lag,
        "seed": seed,
        "sequence_sampling": ("user bootstrap with replacement" if bootstrap_users is not None
                              else "seeded episode sampling without replacement" if max_sequences is not None else "all"),
        "bootstrap_users": bootstrap_users,
        "hyperparameters": {"hidden": hidden, "lambda1": lambda1, "lambda2": lambda2, "threshold": threshold, "max_iter": max_iter, "h_tol": h_tol},
        "features": features,
        "dropped_features": drop_features or [],
        "preserve_binary": preserve_binary,
        "forbid_instantaneous": forbid_instantaneous,
        "constant_features": [name for name, flag in zip(features, constant) if flag],
        "prior_knowledge": prior,
        "acyclicity_history": history,
        "converged": bool(history and history[-1] <= h_tol),
        "instantaneous_is_dag": instantaneous_is_dag(adjacency, len(features)),
        "edges": adjacency_to_edges(adjacency, features, max_lag),
        "warning": "Observational structure is a causal hypothesis, not an identified intervention effect.",
    }
    (output / "nts_dag.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
