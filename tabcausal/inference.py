"""High-level public inference API for TabCausal."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import TabCausalModel
from .preprocessing import normalize_batch


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_state_dict(checkpoint: Any, *, prefer_ema: bool = True) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        if prefer_ema and isinstance(checkpoint.get("ema"), Mapping):
            return dict(checkpoint["ema"])
        if isinstance(checkpoint.get("state_dict"), Mapping):
            return dict(checkpoint["state_dict"])
        if isinstance(checkpoint.get("model_state_dict"), Mapping):
            return dict(checkpoint["model_state_dict"])
    if isinstance(checkpoint, Mapping) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return dict(checkpoint)
    raise ValueError("Could not find a tensor state_dict in the checkpoint.")


def _canonicalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "_orig_mod.", "raw_model.", "model.")
    clean: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        clean[new_key] = value
    return clean


def _infer_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    public_keys = {
        "embed_dim",
        "out_dim",
        "col_num_blocks",
        "col_nhead",
        "key_size",
        "widening_factor",
        "dropout",
        "cosine_temp_init",
        "logit_bias_init",
        "input_dim",
    }
    explicit = checkpoint.get("model_config") if isinstance(checkpoint, Mapping) else None
    if isinstance(explicit, Mapping):
        return {key: explicit[key] for key in public_keys if key in explicit}

    hp = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, Mapping) else {}
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}

    def get(key: str, default: Any) -> Any:
        value = _get_attr(hp, key, None)
        if value is not None:
            return value
        value = _get_attr(cfg, key, None)
        if value is not None:
            return value
        return default

    return {
        "embed_dim": int(get("embed_dim", 128)),
        "out_dim": get("out_dim", None),
        "col_num_blocks": int(get("col_num_blocks", 8)),
        "col_nhead": int(get("col_nhead", 8)),
        "key_size": int(get("key_size", 32)),
        "widening_factor": int(get("widening_factor", 4)),
        "dropout": float(get("dropout", 0.1)),
        "cosine_temp_init": float(get("cosine_temp_init", 2.0)),
        "logit_bias_init": float(get("logit_bias_init", -1.0)),
        "input_dim": int(get("input_dim", 2)),
    }


def _load_inference_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    prefer_ema: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    checkpoint_mapping = checkpoint if isinstance(checkpoint, Mapping) else {}
    return {
        "model_config": _infer_model_config(checkpoint_mapping),
        "state_dict": _canonicalize_state_dict(_extract_state_dict(checkpoint, prefer_ema=prefer_ema)),
    }


def _extract_logits(output: Any) -> torch.Tensor:
    """Return graph logits from the model output."""

    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        output = output[-1]
    if not torch.is_tensor(output):
        raise TypeError(f"Model output must be a tensor, tuple, or list; got {type(output)!r}.")
    return output


class TabCausalPredictor:
    """Load a TabCausal checkpoint and predict directed adjacency matrices."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        normalize: bool = True,
        strict: bool = False,
        prefer_ema: bool = True,
        use_amp: bool = True,
        max_observations: int | None = None,
        observation_seed: int = 0,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.normalize = normalize
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.max_observations = max_observations
        self.observation_seed = int(observation_seed)

        checkpoint = _load_inference_checkpoint(checkpoint_path, map_location="cpu", prefer_ema=prefer_ema)
        self.model_config = dict(checkpoint["model_config"])
        self.model = TabCausalModel(**self.model_config)
        missing, unexpected = self.model.load_state_dict(checkpoint["state_dict"], strict=strict)
        self.load_report = {"missing": list(missing), "unexpected": list(unexpected)}
        self.model.to(self.device)
        if dtype is not None:
            self.model.to(dtype=dtype)
        self.model.eval()

    def _limit_observations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optionally reduce very large tables before the axial encoder.

        The encoder attends over observations on alternating layers, so memory
        can grow quickly with very large ``N``.  This release therefore exposes
        an explicit, deterministic cap.  When ``max_observations`` is ``None``
        no rows are removed.
        """

        limit = self.max_observations
        if limit is None or limit <= 0 or tensor.shape[1] <= limit:
            return tensor
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.observation_seed)
        idx = torch.randperm(tensor.shape[1], generator=generator)[:limit]
        idx, _ = torch.sort(idx)
        return tensor[:, idx]

    @torch.inference_mode()
    def predict_logits(self, x: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        """Predict raw directed-edge logits.

        Parameters
        ----------
        x:
            Array shaped ``[batch, observations, variables, 2]`` or
            ``[observations, variables, 2]``.
        batch_size:
            Optional micro-batch size for memory-efficient evaluation.
        """

        tensor = torch.as_tensor(x, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[-1] != 2:
            raise ValueError(f"Expected [batch, observations, variables, 2], got {tuple(tensor.shape)}")

        tensor = self._limit_observations(tensor)
        if self.normalize:
            tensor = normalize_batch(tensor)
        batch_size = batch_size or int(tensor.shape[0])

        outputs: list[torch.Tensor] = []
        for start in range(0, tensor.shape[0], batch_size):
            xb = tensor[start : start + batch_size].to(self.device)
            if self.dtype is not None:
                xb = xb.to(dtype=self.dtype)
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                logits = _extract_logits(self.model(xb))
            diag = torch.arange(logits.shape[-1], device=logits.device)
            logits[:, diag, diag] = -20.0
            outputs.append(logits.detach().float().cpu())
        return torch.cat(outputs, dim=0).numpy()

    def predict_proba(self, x: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        """Predict directed-edge probabilities."""

        logits = self.predict_logits(x, batch_size=batch_size)
        return 1.0 / (1.0 + np.exp(-logits))

    @torch.inference_mode()
    def predict_embeddings(self, x: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        """Return final-layer node embeddings shaped ``[batch, variables, dim]``."""

        tensor = torch.as_tensor(x, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[-1] != 2:
            raise ValueError(f"Expected [batch, observations, variables, 2], got {tuple(tensor.shape)}")

        tensor = self._limit_observations(tensor)
        if self.normalize:
            tensor = normalize_batch(tensor)
        batch_size = batch_size or int(tensor.shape[0])

        outputs: list[torch.Tensor] = []
        for start in range(0, tensor.shape[0], batch_size):
            xb = tensor[start : start + batch_size].to(self.device)
            if self.dtype is not None:
                xb = xb.to(dtype=self.dtype)
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                embeddings = self.model.extract_feature_embeddings(xb)
            outputs.append(embeddings.detach().float().cpu())
        return torch.cat(outputs, dim=0).numpy()

    def predict_adjacency(
        self,
        x: np.ndarray | torch.Tensor,
        *,
        threshold: float = 0.5,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Predict binary directed adjacency matrices."""

        probs = self.predict_proba(x, batch_size=batch_size)
        adj = (probs >= threshold).astype(np.int64)
        diag = np.arange(adj.shape[-1])
        adj[:, diag, diag] = 0
        return adj
