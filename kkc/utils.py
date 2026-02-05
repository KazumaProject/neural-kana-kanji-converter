from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def save_checkpoint(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))
