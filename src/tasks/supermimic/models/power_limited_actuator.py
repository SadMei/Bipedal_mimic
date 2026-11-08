"""BSRL joint-side PD motor envelope used by the mjlab environment."""

from dataclasses import dataclass

import torch
from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg
from mjlab.entity import Entity


@dataclass(kw_only=True)
class PowerLimitedPdActuatorCfg(IdealPdActuatorCfg):
  """Explicit PD with independent torque, speed, and mechanical-power limits."""

  velocity_limit: float
  peak_power: float

  def __post_init__(self):
    super().__post_init__()
    if self.velocity_limit <= 0 or self.peak_power <= 0:
      raise ValueError("Motor velocity and peak power must be positive")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> "PowerLimitedPdActuator":
    return PowerLimitedPdActuator(self, entity, target_ids, target_names)


class PowerLimitedPdActuator(IdealPdActuator[PowerLimitedPdActuatorCfg]):
  """Match Bipedal_AMP's BSRL gearbox-output saturation semantics."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None and self.damping is not None
    effort = self.stiffness * (cmd.position_target - cmd.pos)
    effort += self.damping * (cmd.velocity_target - cmd.vel)
    effort += cmd.effort_target
    assert self.force_limit is not None
    speed = cmd.vel.abs()
    magnitude = torch.minimum(
      self.force_limit,
      torch.full_like(speed, self.cfg.peak_power) / speed.clamp_min(1.0e-6),
    )
    lower, upper = -magnitude, magnitude
    # Outside the rated speed, retain braking torque but forbid acceleration
    # farther outside the envelope.
    upper = torch.where(cmd.vel >= self.cfg.velocity_limit, 0.0, upper)
    lower = torch.where(cmd.vel <= -self.cfg.velocity_limit, 0.0, lower)
    return torch.clamp(effort, min=lower, max=upper)
