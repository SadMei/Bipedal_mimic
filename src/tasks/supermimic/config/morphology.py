"""Robot variants and tensor layouts shared by simulation, data and learning."""

from dataclasses import dataclass
from pathlib import Path

ROBOTS = ("g1_29dof", "g1_23dof", "bsrl_12dof", "bsrl_closed_ankle")
ASSETS = Path(__file__).resolve().parents[3] / "assets"


@dataclass(frozen=True)
class RobotLayout:
  num_joints: int = 29

  def __post_init__(self):
    if self.num_joints not in (12, 23, 29):
      raise ValueError("Only BSRL 12DoF and G1 23/29DoF are supported")

  @property
  def robot(self):
    if self.num_joints == 12:
      return "bsrl_12dof"
    return f"g1_{self.num_joints}dof"

  @property
  def root_body(self):
    return "base_link" if self.num_joints == 12 else "pelvis"

  @property
  def motion_anchor_body(self):
    return "base_link" if self.num_joints == 12 else "torso_link"

  @property
  def num_tracked_bodies(self):
    return 7 if self.num_joints == 12 else 14

  @property
  def q(self):
    return slice(7, 7 + self.num_joints)

  @property
  def v(self):
    return slice(7 + self.num_joints, 10 + self.num_joints)

  @property
  def w(self):
    return slice(10 + self.num_joints, 13 + self.num_joints)

  @property
  def qd(self):
    return slice(13 + self.num_joints, self.state_dim)

  @property
  def state_dim(self):
    return 13 + 2 * self.num_joints

  @property
  def observation_dim(self):
    return 6 + 3 * self.num_joints

  @property
  def context_dim(self):
    return 4 + self.num_joints

  @property
  def delta_groups(self):
    return (3, 3, self.num_joints, 3, 3, self.num_joints)

  @property
  def reference_dim(self):
    return 10 * (15 + 2 * self.num_joints)

  @property
  def full_body_reference_dim(self):
    return 10 * (15 * self.num_tracked_bodies + 2 * self.num_joints)

  @property
  def xml(self):
    if self.num_joints == 12:
      return ASSETS / "robots/bsrl/xml/bsrl_scene.xml"
    return ASSETS / f"robots/g1/{self.robot}_rev_1_0.xml"

  @property
  def manifest(self):
    if self.num_joints == 12:
      return ASSETS / "motions/bsrl/lafan1_npz/manifest.json"
    suffix = "_23dof" if self.num_joints == 23 else ""
    return ASSETS / f"motions/g1/lafan1_npz{suffix}/manifest.json"


def robot_layout(robot="g1_29dof"):
  if robot not in ROBOTS:
    raise ValueError(f"Unknown robot {robot!r}; expected one of {ROBOTS}")
  return RobotLayout(
    {
      "bsrl_12dof": 12,
      "bsrl_closed_ankle": 12,
      "g1_23dof": 23,
      "g1_29dof": 29,
    }[robot]
  )


def state_layout(state):
  layout = RobotLayout((state.shape[-1] - 13) // 2)
  if state.shape[-1] != layout.state_dim:
    raise ValueError("Physical state width does not match a supported G1 variant")
  return layout


def tracked_bodies(robot="g1_29dof"):
  from .task_cfg import TRACKED_BODIES

  layout = robot_layout(robot)
  if robot in ("bsrl_12dof", "bsrl_closed_ankle"):
    return (
      "base_link",
      "link_left_hip_roll",
      "link_left_shank",
      "link_left_ankle_roll",
      "link_right_hip_roll",
      "link_right_shank",
      "link_right_ankle_roll",
    )
  if robot == "g1_23dof":
    return tuple(
      name.replace("wrist_yaw_link", "wrist_roll_rubber_hand")
      for name in TRACKED_BODIES
    )
  bodies = TRACKED_BODIES
  if len(bodies) != layout.num_tracked_bodies:
    raise ValueError("G1 tracked-body layout is inconsistent")
  return bodies


def motion_anchor_body(robot="g1_29dof"):
  return robot_layout(robot).motion_anchor_body


def motion_anchor_index(robot="g1_29dof"):
  return tracked_bodies(robot).index(motion_anchor_body(robot))
