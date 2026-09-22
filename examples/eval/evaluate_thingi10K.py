"""Thingi10K evaluation driver.

Uses the shared pipeline from eval_pipeline.py with a Thingi10K-specific
reader that streams STL bytes out of the tar.gz archive.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import igl
import numpy as np

from eval_pipeline import (
    MeshSource,
    add_common_args,
    finalize_args,
    run_eval,
)


# =============================================================================
# STL parsing
# =============================================================================

def parse_stl_via_igl(data: bytes) -> np.ndarray:
    """Parse STL bytes via libigl (uses a temp file). Returns (N, 3, 3)."""
    fd, path = tempfile.mkstemp(suffix=".stl")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        v, f_idx = igl.read_triangle_mesh(path)
        if v.size == 0 or f_idx.size == 0:
            raise ValueError("igl returned empty mesh")
        if f_idx.shape[1] != 3:
            raise ValueError(f"igl returned non-triangular faces {f_idx.shape}")
        return v[f_idx].astype(np.float32)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# =============================================================================
# Source
# =============================================================================

class Thingi10KSource(MeshSource):
    """Stream STLs out of a Thingi10K-style .tar.gz archive."""

    def __init__(self, archive: str):
        self.archive = archive

    @property
    def parse_in_producer(self) -> bool:
        # igl STL parsing is cheap and the tar is a stream, so keeping
        # the parse in the producer lets the GPU workers stay saturated.
        return True

    def enumerate(
        self, name_filter: Optional[str]
    ) -> Iterator[Tuple[str, Any]]:
        import tarfile
        with tarfile.open(self.archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if not member.name.endswith(".stl"):
                    continue
                if name_filter and name_filter not in member.name:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                yield member.name, f.read()

    def load(self, name: str, payload: Any) -> np.ndarray:
        return parse_stl_via_igl(payload)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--archive", type=str, required=True,
            help="Path to thingi10K.tar.gz.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Thingi10KSource":
        return cls(args.archive)


# =============================================================================
# Paper comparison
# =============================================================================

def compute_paper_metrics(results_path: Path) -> dict:
    """Recompute exactly the three metrics the FWN paper reports."""
    import json
    misclass_counts = []
    rms_values = []
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            if "error" in rec:
                continue
            misclass_counts.append(rec["misclass_count"])
            rms_values.append(rec["rms_abs"])
    counts = np.array(misclass_counts)
    rms = np.array(rms_values)
    return {
        "n_shapes": len(counts),
        "mean_misclass_count": float(counts.mean()),
        "median_misclass_count": float(np.median(counts)),
        "frac_zero_misclass": float((counts == 0).mean()),
        "mean_rms": float(rms.mean()),
        "median_rms": float(np.median(rms)),
    }


def print_paper_comparison(results_path: Path) -> None:
    paper = compute_paper_metrics(results_path)
    print("\n" + "=" * 72)
    print("  Comparison to FWN paper (Thingi10K, 100³ voxel grid, β=2)")
    print("=" * 72)
    print(f"  {'metric':<40} | {'ours':>12} | {'paper':>12}")
    print("  " + "-" * 68)
    print(f"  {'Average RMS error':<40} | {paper['mean_rms']:>12.3e} | {'8e-3':>12}")
    print(
        f"  {'Average misclassifications':<40} | "
        f"{paper['mean_misclass_count']:>12.1f} | {'363':>12}"
    )
    print(
        f"  {'Median misclassifications':<40} | "
        f"{paper['median_misclass_count']:>12.1f} | {'0':>12}"
    )
    print(
        f"  {'Fraction of shapes with 0 misclass':<40} | "
        f"{100 * paper['frac_zero_misclass']:>11.1f}% | {'>50%':>12}"
    )
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate winder's fast forward engine on Thingi10K."
    )
    Thingi10KSource.add_cli_args(parser)
    add_common_args(parser)
    args = parser.parse_args()

    finalize_args(args)
    source = Thingi10KSource.from_args(args)

    results_path, _ = run_eval(source, args)
    print_paper_comparison(results_path)


if __name__ == "__main__":
    main()

