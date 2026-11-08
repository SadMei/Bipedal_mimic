"""Explicit BSRL parallel-ankle geometry and mjlab actuator support.

The policy continues to command semantic ankle pitch/roll coordinates.  The
simulator contains the physical crank motors and rod closure constraints; this
module maps joint-space PD effort to the two motors on each ankle.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.spec import create_motor_actuator


BSRL_SEMANTIC_JOINTS = (
  "joint_right_hip_yaw",
  "joint_right_hip_roll",
  "joint_right_hip_pitch",
  "joint_right_knee_pitch",
  "joint_right_ankle_pitch",
  "joint_right_ankle_roll",
  "joint_left_hip_yaw",
  "joint_left_hip_roll",
  "joint_left_hip_pitch",
  "joint_left_knee_pitch",
  "joint_left_ankle_pitch",
  "joint_left_ankle_roll",
)

# CAD-zero geometry from the validated Bipedal_AMP explicit ankle.  Side order
# is right, left and branch order is long, short.
MOUNTS = (
  ((-0.06702747, -0.022, -0.01179703), (-0.067027465, 0.022, -0.01179702)),
  ((-0.06702746, 0.022, -0.011897015), (-0.06702747, -0.022, -0.011897015)),
)
PIVOTS = (
  ((-0.01001815, -0.022, 0.24500082), (-0.01003136, 0.022, 0.165000755)),
  ((-0.01001815, 0.022, 0.24490254), (-0.01003136, -0.022, 0.164902485)),
)
CRANK_LENGTHS = ((0.0679998645, 0.06799984), (0.068000159, 0.0680001411))
ROD_LENGTHS = ((0.2452000145, 0.1653000002), (0.2451999947, 0.1652999899))


def _rotation(q: torch.Tensor) -> torch.Tensor:
  """Foot rotation ``Ry(pitch) Rx(roll)``."""
  pitch, roll = q.unbind(-1)
  cp, sp, cr, sr = pitch.cos(), pitch.sin(), roll.cos(), roll.sin()
  zero = torch.zeros_like(pitch)
  return torch.stack(
    (cp, sp * sr, sp * cr, zero, cr, -sr, -sp, cp * sr, cp * cr), -1
  ).reshape(*q.shape[:-1], 3, 3)


def _point_z(direction: torch.Tensor) -> torch.Tensor:
  z = torch.nn.functional.normalize(direction, dim=-1)
  reference = torch.zeros_like(z)
  reference[..., 1] = 1
  x = torch.nn.functional.normalize(torch.linalg.cross(reference, z), dim=-1)
  return torch.stack((x, torch.linalg.cross(z, x), z), -1)


def _matrix_to_xyz(matrix: torch.Tensor) -> torch.Tensor:
  """Angles for MuJoCo's co-located Rx, Ry, Rz hinge sequence."""
  sy = matrix[..., 0, 2].clamp(-1.0, 1.0)
  y = torch.asin(sy)
  x = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
  z = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
  return torch.stack((x, y, z), -1)


class ClosedAnkleGeometry:
  def __init__(self, device: str, dtype: torch.dtype = torch.float32):
    self.mounts = torch.tensor(MOUNTS, device=device, dtype=dtype)
    self.pivots = torch.tensor(PIVOTS, device=device, dtype=dtype)
    self.crank_lengths = torch.tensor(CRANK_LENGTHS, device=device, dtype=dtype)
    self.rod_lengths = torch.tensor(ROD_LENGTHS, device=device, dtype=dtype)
    zeros = torch.zeros((2, 2), device=device, dtype=dtype)
    self.alpha0 = self._raw_angles(torch.zeros((2, 2), device=device, dtype=dtype))[0]
    self.ball_zero_rot = self.crank_rotation(zeros).transpose(-1, -2) @ _point_z(
      self.mounts - self.crank_end(zeros)
    )

  def foot_ends(self, q: torch.Tensor) -> torch.Tensor:
    return (_rotation(q).unsqueeze(-3) @ self.mounts.unsqueeze(-1)).squeeze(-1)

  def _raw_angles(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = self.foot_ends(q) - self.pivots
    radius = torch.linalg.vector_norm(delta[..., (0, 2)], dim=-1)
    cosine = (
      self.rod_lengths.square()
      - delta.square().sum(-1)
      - self.crank_lengths.square()
    ) / (2 * self.crank_lengths * radius.clamp_min(1.0e-12))
    valid = (cosine.abs() <= 1 + 2.0e-6) & torch.isfinite(cosine)
    theta = torch.atan2(delta[..., 2], delta[..., 0]) + cosine.clamp(-1, 1).acos()
    return theta, valid.all(-1)

  def inverse_kinematics(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta, valid = self._raw_angles(q)
    motor = torch.remainder(theta - self.alpha0 + torch.pi, 2 * torch.pi) - torch.pi
    return motor, valid

  def crank_end(self, motor: torch.Tensor) -> torch.Tensor:
    theta = self.alpha0 + motor
    return self.pivots + torch.stack(
      (
        -self.crank_lengths * theta.cos(),
        torch.zeros_like(theta),
        -self.crank_lengths * theta.sin(),
      ),
      -1,
    )

  def crank_rotation(self, motor: torch.Tensor) -> torch.Tensor:
    angle = self.alpha0 + motor
    cosine, sine = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack(
      (-cosine, zero, -sine, zero, -one, zero, -sine, zero, cosine), -1
    ).reshape(*angle.shape, 3, 3)

  def jacobian(self, q: torch.Tensor, motor: torch.Tensor) -> torch.Tensor:
    ends = self.foot_ends(q)
    residual = ends - self.crank_end(motor)
    theta = self.alpha0 + motor
    dcrank = torch.stack(
      (
        self.crank_lengths * theta.sin(),
        torch.zeros_like(theta),
        -self.crank_lengths * theta.cos(),
      ),
      -1,
    )
    denominator = (residual * dcrank).sum(-1)
    pitch_axis = torch.zeros_like(ends)
    pitch_axis[..., 1] = 1
    pitch = q[..., 0]
    roll_axis = torch.stack(
      (pitch.cos(), torch.zeros_like(pitch), -pitch.sin()), -1
    ).unsqueeze(-2).expand_as(ends)
    numerator = torch.stack(
      (
        (residual * torch.linalg.cross(pitch_axis, ends)).sum(-1),
        (residual * torch.linalg.cross(roll_axis, ends)).sum(-1),
      ),
      -1,
    )
    safe = torch.where(
      denominator >= 0,
      denominator.clamp_min(1.0e-9),
      denominator.clamp_max(-1.0e-9),
    )
    return numerator / safe.unsqueeze(-1)

  def rod_angles(self, q: torch.Tensor, motor: torch.Tensor) -> torch.Tensor:
    rod = _point_z(self.foot_ends(q) - self.crank_end(motor))
    relative = (self.crank_rotation(motor) @ self.ball_zero_rot).transpose(-1, -2) @ rod
    return _matrix_to_xyz(relative)

  def reset_coordinates(
    self, q: torch.Tensor, dq: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return motor/rod positions and velocities satisfying loop closure."""
    motor, valid = self.inverse_kinematics(q)
    jacobian = self.jacobian(q, motor)
    motor_velocity = (jacobian @ dq.unsqueeze(-1)).squeeze(-1)
    rod = self.rod_angles(q, motor)
    epsilon = 1.0e-4
    rod_next = self.rod_angles(q + epsilon * dq, motor + epsilon * motor_velocity)
    delta = torch.remainder(rod_next - rod + torch.pi, 2 * torch.pi) - torch.pi
    # The final Rz factor rotates about the rod itself and leaves its endpoint
    # unchanged.  The scalarized universal joint therefore stores only Rx/Ry.
    return motor, motor_velocity, rod[..., :2], (delta / epsilon)[..., :2], valid


def _motor_torque(jacobian: torch.Tensor, joint_torque: torch.Tensor):
  a, b = jacobian[..., 0, 0], jacobian[..., 0, 1]
  c, d = jacobian[..., 1, 0], jacobian[..., 1, 1]
  determinant = a * d - b * c
  safe = torch.where(
    determinant >= 0,
    determinant.clamp_min(1.0e-6),
    determinant.clamp_max(-1.0e-6),
  )
  pitch, roll = joint_torque.unbind(-1)
  torque = torch.stack((d * pitch - c * roll, -b * pitch + a * roll), -1)
  valid = torch.isfinite(jacobian).all(dim=(-1, -2)) & (determinant.abs() > 1.0e-5)
  return torch.where(valid[..., None], torque / safe[..., None], 0.0), valid


@dataclass(kw_only=True)
class ClosedAnkleActuatorCfg(IdealPdActuatorCfg):
  motor_speed: float = 26.1799387799
  motor_power: float = 1250.0

  def build(self, entity: Entity, target_ids: list[int], target_names: list[str]):
    return ClosedAnkleActuator(self, entity, target_ids, target_names)


class ClosedAnkleActuator(IdealPdActuator[ClosedAnkleActuatorCfg]):
  def __init__(self, cfg, entity, target_ids, target_names):
    super().__init__(cfg, entity, target_ids, target_names)
    self._q_ids = [
      [target_names.index(f"joint_{side}_ankle_{axis}") for axis in ("pitch", "roll")]
      for side in ("right", "left")
    ]
    self._motor_ids = [
      [target_names.index(f"motor_{side}_{branch}") for branch in ("1", "2")]
      for side in ("right", "left")
    ]
    self.geometry: ClosedAnkleGeometry | None = None
    self.jacobian_valid: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    for name in target_names:
      actuator = create_motor_actuator(
        spec,
        name,
        effort_limit=self.cfg.effort_limit,
        armature=0.01 if "ankle" in name else 0.0,
        frictionloss=self.cfg.frictionloss,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    self.geometry = ClosedAnkleGeometry(device)
    motor_ids = torch.tensor(sum(self._motor_ids, []), device=device)
    assert self.stiffness is not None and self.damping is not None
    self.stiffness[:, motor_ids] = 0
    self.damping[:, motor_ids] = 0
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.geometry is not None
    assert self.stiffness is not None and self.damping is not None
    q_ids = torch.tensor(self._q_ids, device=cmd.pos.device)
    motor_ids = torch.tensor(self._motor_ids, device=cmd.pos.device)
    q, dq = cmd.pos[:, q_ids], cmd.vel[:, q_ids]
    requested_joint = self.stiffness[:, q_ids] * (cmd.position_target[:, q_ids] - q)
    requested_joint += self.damping[:, q_ids] * (cmd.velocity_target[:, q_ids] - dq)
    requested_joint += cmd.effort_target[:, q_ids]
    jacobian = self.geometry.jacobian(q, cmd.pos[:, motor_ids])
    requested_motor, self.jacobian_valid = _motor_torque(jacobian, requested_joint)
    velocity = cmd.vel[:, motor_ids]
    magnitude = torch.minimum(
      torch.full_like(velocity, self.cfg.effort_limit),
      torch.full_like(velocity, self.cfg.motor_power) / velocity.abs().clamp_min(1.0e-6),
    )
    upper = torch.where(velocity >= self.cfg.motor_speed, 0.0, magnitude)
    lower = torch.where(velocity <= -self.cfg.motor_speed, 0.0, -magnitude)
    output = torch.zeros_like(cmd.pos)
    output[:, motor_ids] = torch.clamp(requested_motor, min=lower, max=upper)
    return output


@dataclass(kw_only=True)
class SemanticJointPositionActionCfg(JointPositionActionCfg):
  joint_names: tuple[str, ...] = BSRL_SEMANTIC_JOINTS

  def build(self, env):
    return SemanticJointPositionAction(self, env)


class SemanticJointPositionAction(JointPositionAction):
  def _find_targets(self, cfg):
    return self._entity.find_joints(cfg.joint_names, preserve_order=False)
