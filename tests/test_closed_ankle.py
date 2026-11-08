"""Explicit BSRL closed-chain geometry and semantic-interface regressions."""

import unittest

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity

from src.tasks.supermimic.config.robot_cfg import get_bsrl_closed_robot_cfg
from src.tasks.supermimic.models.closed_ankle import ClosedAnkleGeometry
from src.tasks.supermimic.models.kinematics import robot_metadata


class ClosedAnkleTests(unittest.TestCase):
  def test_model_has_four_closures_and_semantic_metadata(self):
    entity = Entity(get_bsrl_closed_robot_cfg())
    model = entity.compile()
    self.assertEqual(model.neq, 4)
    self.assertEqual(model.nu, 16)  # 12 semantic targets plus four passive zeros.
    self.assertEqual(len(entity.joint_names), 24)
    self.assertEqual(len(robot_metadata("bsrl_closed_ankle")["joint_names"]), 12)

  def test_reset_coordinates_close_all_rods(self):
    entity = Entity(get_bsrl_closed_robot_cfg())
    model, data = entity.compile(), None
    data = mujoco.MjData(model)
    geometry = ClosedAnkleGeometry("cpu", torch.float64)
    ankle = torch.tensor([[[0.2, 0.1], [-0.15, -0.08]]], dtype=torch.float64)
    motor, _, rod, _, valid = geometry.reset_coordinates(
      ankle, torch.zeros_like(ankle)
    )
    self.assertTrue(bool(valid.all()))
    for side_id, side in enumerate(("right", "left")):
      for axis_id, axis in enumerate(("pitch", "roll")):
        data.joint(f"joint_{side}_ankle_{axis}").qpos[0] = ankle[0, side_id, axis_id]
      for branch in range(2):
        data.joint(f"motor_{side}_{branch + 1}").qpos[0] = motor[0, side_id, branch]
        for axis_id, axis in enumerate("xy"):
          data.joint(f"rod_ball_{side}_{branch + 1}_{axis}").qpos[0] = rod[
            0, side_id, branch, axis_id
          ]
    mujoco.mj_forward(model, data)
    gaps = []
    for side in ("right", "left"):
      for branch in (1, 2):
        gaps.append(
          np.linalg.norm(
            data.site(f"rod_end_{side}_{branch}").xpos
            - data.site(f"foot_end_{side}_{branch}").xpos
          )
        )
    self.assertLess(max(gaps), 1.0e-8)


if __name__ == "__main__":
  unittest.main()
