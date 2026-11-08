"""Retarget the SuperMimic LAFAN1 clip set to BSRL and export 50 Hz NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import joblib
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.convert_lafan_to_npz import (
  motion_to_arrays,
  normalize_quaternions,
  resample_motion,
  save_npz,
)
from src.tasks.supermimic.config.morphology import robot_layout, tracked_bodies
from src.tasks.supermimic.config.robot_cfg import get_bsrl_robot_cfg

BSRL_JOINT_CONVENTION = "bsrl_xyz_positive_v1"
BSRL_GROUNDING_CONVENTION = "bsrl_per_frame_collision_v1"


def file_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def retarget(source: Path, output: Path, gmr_dir: Path) -> None:
  sys.path.insert(0, str(gmr_dir))
  sys.path.insert(0, str(gmr_dir / "scripts"))
  from bvh_to_robot_dataset import retarget_file

  retarget_file(
    str(source), str(output), "bsrl", motion_fps=30.0,
    ground_offset=0.0, verbose=False,
  )


def convert(cache: Path, destination: Path, model, joints, bodies) -> int:
  source = joblib.load(cache)
  if source.get("joint_coordinate_convention") != BSRL_JOINT_CONVENTION:
    raise ValueError(f"Unexpected BSRL joint convention in {cache}")
  if source.get("grounding_convention") != BSRL_GROUNDING_CONVENTION:
    raise ValueError(f"Unexpected BSRL grounding convention in {cache}")
  if Path(source.get("source_motion", "")).stem != destination.stem:
    raise ValueError(f"Cache source does not match destination clip: {cache}")
  fps = float(source["fps"])
  q_xyzw = normalize_quaternions(np.asarray(source["root_rot"], dtype=np.float64))
  raw = np.concatenate(
    (
      np.asarray(source["root_pos"], dtype=np.float64),
      q_xyzw,
      np.asarray(source["dof_pos"], dtype=np.float64),
    ),
    axis=-1,
  )
  if raw.shape[1] != 19:
    raise ValueError(f"Expected 7 root + 12 joint values in {cache}, got {raw.shape}")
  motion = resample_motion(raw, fps, 50.0, quaternion_order="xyzw")
  source_joints = tuple(str(name) for name in source["dof_names"])
  arrays = motion_to_arrays(
    model, joints, bodies, motion, 50.0, source_joint_names=source_joints
  )
  joint_ids = np.asarray([model.joint(name).id for name in joints])
  limits = model.jnt_range[joint_ids]
  lower_error = limits[:, 0] - arrays["joint_pos"].min(axis=0)
  upper_error = arrays["joint_pos"].max(axis=0) - limits[:, 1]
  if max(float(lower_error.max()), float(upper_error.max())) > 1.0e-5:
    raise ValueError(f"Retargeted joint positions exceed BSRL limits in {cache}")
  arrays.update(
    {
      "source_file": np.asarray(str(Path(source["source_motion"]).name)),
      "retarget_cache": np.asarray(str(cache.resolve())),
      "input_fps": np.array([fps], dtype=np.float32),
      "input_quaternion_order": np.asarray("xyzw"),
      "joint_coordinate_convention": np.asarray(source["joint_coordinate_convention"]),
      "grounding_convention": np.asarray(source["grounding_convention"]),
      "schema_version": np.array([1], dtype=np.int32),
    }
  )
  save_npz(destination, arrays)
  return len(motion)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source-dir", type=Path,
    default=Path("/root/mjw/GMR/motion_data/LAFAN1_source/lafan1"),
  )
  parser.add_argument("--gmr-dir", type=Path, default=Path("/root/mjw/GMR"))
  parser.add_argument(
    "--g1-manifest", type=Path,
    default=ROOT / "src/assets/motions/g1/lafan1_npz/manifest.json",
  )
  parser.add_argument("--output-dir", type=Path, default=robot_layout("bsrl_12dof").manifest.parent)
  parser.add_argument("--cache-dir", type=Path)
  parser.add_argument("--clips", nargs="*", default=())
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()
  args.cache_dir = args.cache_dir or args.output_dir.parent / "lafan1_gmr_cache"

  selected = {
    Path(entry["file"]).stem: entry
    for entry in json.loads(args.g1_manifest.read_text())["clips"]
  }
  if args.clips:
    unknown = set(args.clips) - selected.keys()
    if unknown:
      parser.error(f"Unknown clips: {', '.join(sorted(unknown))}")
    selected = {name: selected[name] for name in args.clips}

  from mjlab.entity import Entity

  entity = Entity(get_bsrl_robot_cfg())
  model = entity.spec.compile()
  joints = tuple(entity.joint_names)
  bodies = tracked_bodies("bsrl_12dof")
  with tempfile.NamedTemporaryFile(suffix=".mjb") as stream:
    mujoco.mj_saveModel(model, stream.name, None)
    model_hash = file_sha256(Path(stream.name))

  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.cache_dir.mkdir(parents=True, exist_ok=True)
  entries = []
  for index, (name, original) in enumerate(selected.items(), 1):
    source = args.source_dir / f"{name}.bvh"
    cache = args.cache_dir / f"{name}.pkl"
    destination = args.output_dir / f"{name}.npz"
    if not source.is_file():
      raise FileNotFoundError(source)
    if not cache.exists():
      retarget(source, cache, args.gmr_dir)
    if destination.exists() and not args.overwrite:
      raise FileExistsError(f"{destination}; pass --overwrite")
    frames = convert(cache, destination, model, joints, bodies)
    entries.append(
      {
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "file": destination.name,
        "input_frames": int(original["input_frames"]),
        "frames": frames,
        "duration_seconds": (frames - 1) / 50.0,
      }
    )
    print(f"[{index}/{len(selected)}] {name}: {frames} frames", flush=True)

  manifest = {
    "schema_version": 1,
    "input_fps": 30.0,
    "fps": 50.0,
    "input_quaternion_order": "xyzw",
    "quaternion_order": "wxyz",
    "robot": "bsrl_12dof",
    "model_source": str(robot_layout("bsrl_12dof").xml.resolve()),
    "model_sha256": model_hash,
    "mujoco_version": mujoco.__version__,
    "joint_names": joints,
    "body_names": bodies,
    "retargeter": "GMR bvh_lafan1_to_bsrl",
    "joint_coordinate_convention": BSRL_JOINT_CONVENTION,
    "grounding_convention": BSRL_GROUNDING_CONVENTION,
    "clips": entries,
  }
  manifest_path = args.output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"Saved {len(entries)} clips to {manifest_path}")


if __name__ == "__main__":
  main()
