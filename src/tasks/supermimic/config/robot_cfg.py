"""G1 physical configuration shared by simulation and model fingerprints."""

import re
from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco
from mjlab.actuator import DelayedActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from .morphology import robot_layout
from ..models.power_limited_actuator import PowerLimitedPdActuatorCfg
from ..models.closed_ankle import ClosedAnkleActuatorCfg

G1_XML = Path(__file__).resolve().parents[3] / "assets/robots/g1/g1_29dof_rev_1_0.xml"
BSRL_XML = Path(__file__).resolve().parents[3] / "assets/robots/bsrl/xml/bsrl_scene.xml"
BSRL_CLOSED_XML = (
  Path(__file__).resolve().parents[3]
  / "assets/robots/bsrl/xml/bsrl_closed_ankles.xml"
)
G1_FOOT_COLLISION_EXPR = r"^(left|right)_foot[1-4]_collision$"
BSRL_FOOT_COLLISION_EXPR = r"^(left|right)_foot[1-9]_collision$"
# Backward-compatible G1 default used by existing imports.
FOOT_COLLISION_EXPR = G1_FOOT_COLLISION_EXPR

BSRL_ACTION_SCALE = {".*": 0.25}
BSRL_SMALL_MOTOR_PEAK_TORQUE = 112.0
BSRL_SMALL_MOTOR_PEAK_SPEED = 250.0 * 2.0 * 3.141592653589793 / 60.0
BSRL_SMALL_MOTOR_PEAK_POWER = 1250.0
BSRL_LARGE_MOTOR_PEAK_TORQUE = 216.0
BSRL_LARGE_MOTOR_PEAK_SPEED = 144.5 * 2.0 * 3.141592653589793 / 60.0
BSRL_LARGE_MOTOR_PEAK_POWER = 1050.0


def foot_collision_expr(robot: str = "g1_29dof") -> str:
  return (
    BSRL_FOOT_COLLISION_EXPR
    if robot in ("bsrl_12dof", "bsrl_closed_ankle")
    else G1_FOOT_COLLISION_EXPR
  )


def get_supermimic_spec(robot: str = "g1_29dof") -> mujoco.MjSpec:
  """Adapt the vendored RoboJuDo-Plus asset to the mjlab training scene."""
  if robot == "bsrl_12dof":
    return get_bsrl_spec()
  if robot == "bsrl_closed_ankle":
    return get_bsrl_closed_spec()
  xml = G1_XML if robot == "g1_29dof" else robot_layout(robot).xml
  spec = mujoco.MjSpec.from_file(str(xml))
  update_assets(spec.assets, xml.parent / "meshes", spec.meshdir)
  # The scene supplies its own terrain; Entity supplies the training PD motors.
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  # Source geoms are unnamed. Name them without changing their geometry or
  # enabling visual-only meshes, so collision filters and friction DR can select
  # the original collision shapes consistently.
  for body in spec.bodies:
    foot_index = 0
    for index, geom in enumerate(body.geoms):
      if geom.contype or geom.conaffinity:
        if body.name in ("left_ankle_roll_link", "right_ankle_roll_link"):
          foot_index += 1
          side = body.name.split("_", 1)[0]
          geom.name = f"{side}_foot{foot_index}_collision"
        else:
          geom.name = f"{body.name}_{index}_collision"
      else:
        geom.name = f"{body.name}_{index}_visual"
  return spec


def get_bsrl_spec() -> mujoco.MjSpec:
  """Load the Bipedal_AMP BSRL serial model without its standalone scene."""
  spec = mujoco.MjSpec.from_file(str(BSRL_XML))
  update_assets(spec.assets, BSRL_XML.parent.parent / "meshes", spec.meshdir)
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for body in spec.bodies:
    foot_index = 0
    for index, geom in enumerate(body.geoms):
      if geom.contype or geom.conaffinity:
        if body.name in ("link_left_ankle_roll", "link_right_ankle_roll"):
          foot_index += 1
          side = "left" if "left" in body.name else "right"
          geom.name = f"{side}_foot{foot_index}_collision"
        else:
          geom.name = f"{body.name}_{index}_collision"
      else:
        geom.name = f"{body.name}_{index}_visual"
  return spec


def get_bsrl_closed_spec() -> mujoco.MjSpec:
  """Load the explicit parallel ankles and scalarize passive ball joints.

  mjlab currently assumes one scalar position per named non-free joint.  Three
  co-located hinges are kinematically equivalent to each passive ball joint and
  keep the asset compatible with that batched state interface.
  """
  spec = mujoco.MjSpec.from_file(str(BSRL_CLOSED_XML))
  mesh_root = BSRL_CLOSED_XML.parent.parent / "meshes"
  # update_assets intentionally handles flat robot bundles.  The closed ankle
  # CAD is nested and must retain its relative path when the entity is attached
  # to the batched scene (which otherwise prefixes unresolved disk paths).
  for path in mesh_root.rglob("*"):
    if path.is_file():
      relative = path.relative_to(mesh_root).as_posix()
      spec.assets[f"{spec.meshdir}{relative}"] = path.read_bytes()
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for joint in list(spec.joints):
    if not joint.name.startswith("rod_ball_"):
      continue
    body, name = joint.parent, joint.name
    spec.delete(joint)
    # Rz is axial rod spin and does not affect the endpoint.  Omitting that
    # unconstrained coordinate avoids a rank-deficient equality Jacobian.
    for suffix, axis in zip("xy", ((1, 0, 0), (0, 1, 0))):
      body.add_joint(
        name=f"{name}_{suffix}",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=axis,
        damping=1.0e-4,
      )
  for body in spec.bodies:
    foot_index = 0
    for index, geom in enumerate(body.geoms):
      if geom.contype or geom.conaffinity:
        if body.name in ("link_left_ankle_roll", "link_right_ankle_roll"):
          foot_index += 1
          side = "left" if "left" in body.name else "right"
          geom.name = f"{side}_foot{foot_index}_collision"
        else:
          geom.name = f"{body.name}_{index}_collision"
      elif not geom.name:
        geom.name = f"{body.name}_{index}_visual"
  return spec


def _bsrl_motor(
  names: tuple[str, ...], stiffness: float, damping: float, torque: float,
  velocity: float, power: float,
) -> DelayedActuatorCfg:
  return DelayedActuatorCfg(
    base_cfg=PowerLimitedPdActuatorCfg(
      target_names_expr=names,
      armature=0.01,
      stiffness=stiffness,
      damping=damping,
      effort_limit=torque,
      velocity_limit=velocity,
      peak_power=power,
    ),
    delay_target=("position", "velocity", "effort"),
    delay_min_lag=0,
    delay_max_lag=4,
    # Bipedal_AMP samples one fixed lag per episode through the reset event.
    delay_update_period=0,
  )


def get_bsrl_robot_cfg() -> EntityCfg:
  """BSRL dynamics copied from the successful Bipedal_AMP configuration."""
  large = (
    _bsrl_motor(
      ("joint_.*_hip_pitch",), 150.0, 2.0,
      BSRL_LARGE_MOTOR_PEAK_TORQUE, BSRL_LARGE_MOTOR_PEAK_SPEED,
      BSRL_LARGE_MOTOR_PEAK_POWER,
    ),
    _bsrl_motor(
      ("joint_.*_hip_roll",), 150.0, 2.0,
      BSRL_LARGE_MOTOR_PEAK_TORQUE, BSRL_LARGE_MOTOR_PEAK_SPEED,
      BSRL_LARGE_MOTOR_PEAK_POWER,
    ),
    _bsrl_motor(
      ("joint_.*_knee_pitch",), 200.0, 2.0,
      BSRL_LARGE_MOTOR_PEAK_TORQUE, BSRL_LARGE_MOTOR_PEAK_SPEED,
      BSRL_LARGE_MOTOR_PEAK_POWER,
    ),
  )
  small = (
    _bsrl_motor(
      ("joint_.*_hip_yaw",), 150.0, 2.0,
      BSRL_SMALL_MOTOR_PEAK_TORQUE, BSRL_SMALL_MOTOR_PEAK_SPEED,
      BSRL_SMALL_MOTOR_PEAK_POWER,
    ),
    _bsrl_motor(
      ("joint_.*_ankle_pitch",), 100.0, 1.0,
      BSRL_SMALL_MOTOR_PEAK_TORQUE, BSRL_SMALL_MOTOR_PEAK_SPEED,
      BSRL_SMALL_MOTOR_PEAK_POWER,
    ),
    _bsrl_motor(
      ("joint_.*_ankle_roll",), 100.0, 1.0,
      BSRL_SMALL_MOTOR_PEAK_TORQUE, BSRL_SMALL_MOTOR_PEAK_SPEED,
      BSRL_SMALL_MOTOR_PEAK_POWER,
    ),
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.8665),
      joint_pos={
        "joint_.*_hip_yaw": 0.0,
        "joint_.*_hip_roll": 0.0,
        "joint_.*_hip_pitch": -0.2,
        "joint_.*_knee_pitch": 0.4,
        "joint_.*_ankle_pitch": -0.2,
        "joint_.*_ankle_roll": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_bsrl_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=large + small,
      soft_joint_pos_limit_factor=0.9,
    ),
    sort_actuators=True,
    collisions=(
      CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype=1,
        conaffinity=1,
        condim={BSRL_FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
        priority={BSRL_FOOT_COLLISION_EXPR: 1},
      ),
    ),
  )


def get_bsrl_closed_robot_cfg() -> EntityCfg:
  """BSRL with passive ankle pitch/roll and four physical crank motors."""
  direct = (
    _bsrl_motor(
      ("joint_.*_hip_pitch",), 150.0, 2.0,
      BSRL_LARGE_MOTOR_PEAK_TORQUE, BSRL_LARGE_MOTOR_PEAK_SPEED,
      BSRL_LARGE_MOTOR_PEAK_POWER,
    ),
    _bsrl_motor(
      ("joint_.*_hip_roll",), 150.0, 2.0,
      BSRL_LARGE_MOTOR_PEAK_TORQUE, BSRL_LARGE_MOTOR_PEAK_SPEED,
      BSRL_LARGE_MOTOR_PEAK_POWER,
    ),
    _bsrl_motor(
      ("joint_.*_knee_pitch",), 200.0, 2.0,
      BSRL_LARGE_MOTOR_PEAK_TORQUE, BSRL_LARGE_MOTOR_PEAK_SPEED,
      BSRL_LARGE_MOTOR_PEAK_POWER,
    ),
    _bsrl_motor(
      ("joint_.*_hip_yaw",), 150.0, 2.0,
      BSRL_SMALL_MOTOR_PEAK_TORQUE, BSRL_SMALL_MOTOR_PEAK_SPEED,
      BSRL_SMALL_MOTOR_PEAK_POWER,
    ),
  )
  ankle = DelayedActuatorCfg(
    base_cfg=ClosedAnkleActuatorCfg(
      target_names_expr=("joint_.*_ankle_(pitch|roll)", "motor_.*"),
      armature=0.0,
      stiffness=100.0,
      damping=1.0,
      effort_limit=BSRL_SMALL_MOTOR_PEAK_TORQUE,
      motor_speed=BSRL_SMALL_MOTOR_PEAK_SPEED,
      motor_power=BSRL_SMALL_MOTOR_PEAK_POWER,
    ),
    delay_target=("position", "velocity", "effort"),
    delay_min_lag=0,
    delay_max_lag=4,
    delay_update_period=0,
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.881216),
      joint_pos={
        "joint_.*_hip_yaw": 0.0,
        "joint_.*_hip_roll": 0.0,
        "joint_.*_hip_pitch": -0.2,
        "joint_.*_knee_pitch": 0.4,
        "joint_.*_ankle_(pitch|roll)": 0.0,
        "motor_.*": 0.0,
        "rod_ball_.*": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_bsrl_closed_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=direct + (ankle,),
      soft_joint_pos_limit_factor=0.9,
    ),
    sort_actuators=True,
    collisions=(
      CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype=1,
        conaffinity=1,
        condim={BSRL_FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
        priority={BSRL_FOOT_COLLISION_EXPR: 1},
      ),
    ),
  )
def get_supermimic_robot_cfg(robot: str = "g1_29dof"):
  """Retain full-body ground contacts while disabling robot self contacts."""
  if robot == "bsrl_12dof":
    return get_bsrl_robot_cfg()
  if robot == "bsrl_closed_ankle":
    return get_bsrl_closed_robot_cfg()
  config = get_g1_robot_cfg()
  layout = robot_layout(robot)
  config.spec_fn = (
    get_supermimic_spec
    if layout.num_joints == 29
    else partial(get_supermimic_spec, robot)
  )
  config.articulation = deepcopy(config.articulation)
  if layout.num_joints == 23:
    names = [j.name for j in config.spec_fn().joints]
    actuators = []
    for actuator in config.articulation.actuators:
      actuator.target_names_expr = tuple(
        p for p in actuator.target_names_expr if any(re.fullmatch(p, n) for n in names)
      )
      if actuator.target_names_expr:
        actuators.append(actuator)
    config.articulation.actuators = tuple(actuators)
  config.collisions = (
    CollisionCfg(
      geom_names_expr=(".*_collision",),
      contype=0,
      conaffinity=1,
      condim={G1_FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
      priority={G1_FOOT_COLLISION_EXPR: 1},
    ),
  )
  return config
