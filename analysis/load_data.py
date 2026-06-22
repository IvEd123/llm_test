"""Load math/others hidden-state samples.

Source confirmed from hui_gpt2_1.ipynb (cells 20-24) and gpt2_walker.py:
each samples/<label>_<i>.txt is np.savetxt(...) of `_h` = block.mlp.act(block.mlp.c_fc(ln_2(x)))
for model.transformer.h[0] (layer 0), i.e. GPT-2 block-0 MLP *post-GELU* activations.
Shape per file: (num_tokens, 2816) where 2816 = n_inner = 4 * n_embd (n_embd=704, 11 layers, 11 heads,
model = ./weiser/101M-0.4, a GPT-2-architecture LM). Rows = token positions, columns = MLP neurons.

samples/samples.txt holds "|prompt| - label_i; token_count" metadata lines (one is a duplicate: other_3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@dataclass
class Sample:
    name: str          # e.g. "math_0"
    label: str         # "math" | "others"
    prompt: str | None
    token_count: int | None
    matrix: np.ndarray  # (tokens, 2816)


def _parse_metadata(meta_path: Path) -> dict[str, tuple[str, int]]:
    """Parse samples.txt -> {sample_name: (prompt, token_count)}."""
    meta = {}
    if not meta_path.exists():
        return meta
    pattern = re.compile(r"\|(.*)\|\s*-\s*([\w]+);\s*(\d+)")
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        prompt, name, count = m.groups()
        meta[name] = (prompt.strip(), int(count))
    return meta


def load_samples(samples_dir: Path = SAMPLES_DIR) -> list[Sample]:
    meta_path = samples_dir / "samples.txt"
    if not meta_path.exists():
        meta_path = samples_dir / f"{samples_dir.name}.txt"
    meta = _parse_metadata(meta_path)
    samples = []
    for path in sorted(samples_dir.glob("*.txt")):
        if path.name in ("samples.txt", f"{samples_dir.name}.txt"):
            continue
        name = path.stem  # e.g. "math_0"
        label_raw = name.rsplit("_", 1)[0]
        label = "math" if label_raw == "math" else "others"

        matrix = np.loadtxt(path)
        if matrix.ndim == 1:  # single-token edge case
            matrix = matrix[None, :]

        prompt, token_count = meta.get(name, (None, None))
        samples.append(Sample(name=name, label=label, prompt=prompt,
                               token_count=token_count, matrix=matrix))
    return samples


def validate_samples(samples: list[Sample]) -> dict:
    """Sanity-check loaded samples; returns a summary dict and raises on hard errors."""
    widths = {s.matrix.shape[1] for s in samples}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent feature width across samples: {widths}")
    width = widths.pop()

    for s in samples:
        if not np.isfinite(s.matrix).all():
            raise ValueError(f"Non-finite values found in {s.name}")
        if s.token_count is not None and s.matrix.shape[0] != s.token_count:
            raise ValueError(
                f"{s.name}: matrix has {s.matrix.shape[0]} rows but "
                f"metadata says token_count={s.token_count}"
            )

    labels = [s.label for s in samples]
    return {
        "n_samples": len(samples),
        "width": width,
        "class_balance": {lbl: labels.count(lbl) for lbl in set(labels)},
        "token_counts": {s.name: s.matrix.shape[0] for s in samples},
    }


if __name__ == "__main__":
    samples = load_samples()
    summary = validate_samples(samples)
    print(summary)
