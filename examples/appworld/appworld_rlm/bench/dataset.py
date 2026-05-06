from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATASET_FILES = {
    "train": "train.txt",
    "validation": "train.txt",
    "dev": "dev.txt",
    "test_normal": "test_normal.txt",
    "test_challenge": "test_challenge.txt",
}


@dataclass(frozen=True)
class AppWorldExample:
    task_id: str
    dataset: str
    instruction: str = ""

    @property
    def group_id(self) -> str:
        return self.task_id.rsplit("_", 1)[0]


def load_dataset(dataset: str, data_root: str | Path = "data") -> list[AppWorldExample]:
    if dataset not in DATASET_FILES:
        valid = ", ".join(sorted(DATASET_FILES))
        raise ValueError(f"unknown AppWorld dataset {dataset!r}; expected one of {valid}")
    root = Path(data_root)
    if dataset == "validation":
        _train, validation = load_train_validation(root)
        return validation
    path = root / "datasets" / DATASET_FILES[dataset]
    if not path.is_file():
        raise FileNotFoundError(
            f"AppWorld split file not found at {path}. Run AppWorld data download or pass --data-root."
        )
    return [
        AppWorldExample(
            task_id=task_id,
            dataset=dataset,
            instruction=_load_instruction(root, task_id),
        )
        for task_id in _read_split_task_ids(path)
    ]


def _read_split_task_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load_instruction(data_root: Path, task_id: str) -> str:
    specs_path = data_root / "tasks" / task_id / "specs.json"
    if not specs_path.is_file():
        return ""
    specs = json.loads(specs_path.read_text())
    instruction = specs.get("instruction", "")
    if not isinstance(instruction, str):
        instruction = ""
    return _format_instruction_with_supervisor_context(instruction, specs)


def _format_instruction_with_supervisor_context(instruction: str, specs: dict) -> str:
    supervisor = specs.get("supervisor") or specs.get("main_user") or {}
    if not isinstance(supervisor, dict):
        return instruction
    first_name = supervisor.get("first_name")
    last_name = supervisor.get("last_name")
    email = supervisor.get("email")
    phone_number = supervisor.get("phone_number")
    if not all(isinstance(value, str) and value for value in (first_name, last_name, email, phone_number)):
        return instruction
    return (
        "I am your supervisor. "
        f"My name is: {first_name} {last_name}. "
        f"My personal email is {email} and phone number is {phone_number}.\n\n"
        "# Real Task Instruction\n"
        f"{instruction}"
    )


def load_train_validation(
    data_root: str | Path = "data",
    val_ratio: float = 0.20,
    seed: int = 13,
) -> tuple[list[AppWorldExample], list[AppWorldExample]]:
    train_pool = load_dataset("train", data_root)
    return split_train_validation(train_pool, val_ratio=val_ratio, seed=seed)


def split_train_validation(
    examples: list[AppWorldExample],
    val_ratio: float = 0.20,
    seed: int = 13,
) -> tuple[list[AppWorldExample], list[AppWorldExample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    groups = sorted({example.group_id for example in examples})
    if len(groups) < 2:
        raise ValueError("need at least two task groups for train/validation split")
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_size = max(1, int(round(len(groups) * val_ratio)))
    val_groups = set(groups[:val_size])
    train = [example for example in examples if example.group_id not in val_groups]
    validation = [example for example in examples if example.group_id in val_groups]
    if not train or not validation:
        raise ValueError("train/validation split produced an empty side")
    return train, validation
