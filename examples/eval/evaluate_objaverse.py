"""Objaverse evaluation driver.

Uses the shared pipeline from eval_pipeline.py with an Objaverse-specific
reader that walks the pre-extracted GLB directory tree.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import trimesh

from eval_pipeline import (
    MeshSource,
    add_common_args,
    finalize_args,
    run_eval,
)


# Silence trimesh's extremely chatty loader
logging.getLogger("trimesh").setLevel(logging.ERROR)


# =============================================================================
# GLB loading
# =============================================================================


def load_glb_triangles(path: Path) -> np.ndarray:
    """Load a GLB and return (N, 3, 3) float32 triangles in world space.

    Handles both single-mesh files and scene graphs. Scene transforms are
    applied so every triangle lands in a common world frame. Returns only
    triangular geometry; polygonal faces are triangulated by trimesh.
    """
    loaded = trimesh.load(str(path), force="scene", skip_materials=True)

    if isinstance(loaded, trimesh.Scene):
        parts = []
        for geom in loaded.dump():
            if not hasattr(geom, "faces") or not hasattr(geom, "vertices"):
                continue
            if len(geom.faces) == 0 or len(geom.vertices) == 0:
                continue
            parts.append(geom.vertices[geom.faces])
        if not parts:
            raise ValueError("Scene contains no triangular geometry")
        tris = np.concatenate(parts, axis=0)
    else:
        if len(loaded.faces) == 0:
            raise ValueError("Empty mesh")
        tris = loaded.vertices[loaded.faces]

    return tris.astype(np.float32)


# =============================================================================
# Source
# =============================================================================


class ObjaverseSource(MeshSource):
    """Walk a directory tree of pre-extracted GLB files."""

    def __init__(self, root: str):
        self.root = Path(root)

    @property
    def parse_in_producer(self) -> bool:
        # trimesh.load is CPU-bound (parsing, scene graph resolution,
        # transform application). Push it to the workers so parsing
        # parallelizes.
        return False

    def enumerate(self, name_filter: str | None) -> Iterator[tuple[str, Any]]:
        paths = []
        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                if not f.endswith(".glb"):
                    continue
                full = Path(dirpath) / f
                rel = str(full.relative_to(self.root))
                if name_filter and name_filter not in rel:
                    continue
                paths.append(rel)
        paths.sort()
        for rel in paths:
            yield rel, rel

    def load(self, name: str, payload: Any) -> np.ndarray:
        return load_glb_triangles(self.root / payload)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--root",
            type=str,
            required=True,
            help="Root directory containing GLB files (walked recursively).",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ObjaverseSource":
        return cls(args.root)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate winder's fast forward engine on Objaverse."
    )
    ObjaverseSource.add_cli_args(parser)
    add_common_args(parser)
    # Objaverse wants more workers by default
    parser.set_defaults(workers=4, queue_size=16)
    args = parser.parse_args()

    finalize_args(args)
    source = ObjaverseSource.from_args(args)

    run_eval(source, args)


if __name__ == "__main__":
    main()
