"""Minimal mjlab scene for collection; task signals are shared tensor functions."""

import re

import mujoco
from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, push_by_setting_velocity
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .config.robot_cfg import (
  BSRL_ACTION_SCALE,
  foot_collision_expr,
  get_supermimic_robot_cfg,
)
from .config.task_cfg import FoundationConfig
from .mdp.bsrl_events import scale_body_mass_and_inertia
from .models.closed_ankle import SemanticJointPositionActionCfg


class FoundationMujocoCfg(MujocoCfg):
  def apply(self, model: mujoco.MjModel) -> None:
    super().apply(model)
    # Observations are reconstructed from synchronized physical state, including
    # explicit observation noise. No MuJoCo built-in sensors are consumed here.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR


def make_env_cfg(config: FoundationConfig) -> ManagerBasedRlEnvCfg:
  closed_ankle = config.robot == "bsrl_closed_ankle"
  foot_expr = foot_collision_expr(config.robot)
  foot = SceneEntityCfg("robot", geom_names=foot_expr)
  torso = SceneEntityCfg("robot", body_names=(config.layout.motion_anchor_body,))
  # Constant ranges also expand model fields in debug/evaluation mode, ensuring
  # the context always comes from the actual applied model parameters.
  mu = (0.3, 1.2) if config.randomization else (1.0, 1.0)
  bias = (-0.01, 0.01) if config.randomization else (0.0, 0.0)
  com = {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
  if not config.randomization:
    com = {i: (0.0, 0.0) for i in range(3)}
  robot_cfg = get_supermimic_robot_cfg(config.robot)
  joint_names = [j.name for j in robot_cfg.spec_fn().joints]
  action_scale = BSRL_ACTION_SCALE if config.layout.num_joints == 12 else G1_ACTION_SCALE
  if config.layout.num_joints == 12:
    body = SceneEntityCfg("robot", body_names=(".*",))
    random = config.randomization
    events = {
      "physics_material": EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
          # MuJoCo exposes sliding/torsional/rolling friction rather than
          # PhysX's static/dynamic pair; match BSRL's 0.5--1.5 sliding range.
          "ranges": {0: (0.5, 1.5) if random else (1.0, 1.0)},
          "operation": "abs",
        },
      ),
      "joint_default_pos": EventTermCfg(
        func=dr.joint_default_pos,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          # Changing qpos0 changes the linkage installation reference itself.
          # Encoder/target noise remains available without violating CAD closure.
          "ranges": (
            (-0.05, 0.05)
            if random and not closed_ankle
            else (0.0, 0.0)
          ),
          "operation": "add",
        },
      ),
      "base_com": EventTermCfg(
        func=dr.body_com_offset,
        mode="startup",
        params={
          "asset_cfg": torso,
          "ranges": {
            0: (-0.025, 0.025) if random else (0.0, 0.0),
            1: (-0.05, 0.05) if random else (0.0, 0.0),
            2: (-0.05, 0.05) if random else (0.0, 0.0),
          },
          "operation": "add",
        },
      ),
      "actuator_gains": EventTermCfg(
        func=dr.pd_gains,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "kp_range": (0.8, 1.2) if random else (1.0, 1.0),
          "kd_range": (0.8, 1.2) if random else (1.0, 1.0),
          "operation": "scale",
        },
      ),
      "body_mass": EventTermCfg(
        func=scale_body_mass_and_inertia,
        mode="startup",
        params={
          "asset_cfg": body,
          "scale_range": (0.8, 1.2) if random else (1.0, 1.0),
        },
      ),
      "joint_armature": EventTermCfg(
        func=dr.joint_armature,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          "ranges": (0.8, 1.2) if random else (1.0, 1.0),
          "operation": "scale",
        },
      ),
      "joint_friction": EventTermCfg(
        func=dr.joint_friction,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          "ranges": (0.0, 0.05) if random else (0.0, 0.0),
          "operation": "abs",
        },
      ),
      "actuator_delay": EventTermCfg(
        func=dr.sync_actuator_delays,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "lag_range": (0, 4) if random else (0, 0),
        },
      ),
      "push_robot": EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "velocity_range": {
            "x": (-0.2, 0.2) if random else (0.0, 0.0),
            "y": (-0.2, 0.2) if random else (0.0, 0.0),
          },
        },
      ),
    }
  else:
    events = {
      "foot_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
          "asset_cfg": foot,
          "ranges": mu,
          "operation": "abs",
          "shared_random": True,
        },
      ),
      "encoder_bias": EventTermCfg(
        func=dr.encoder_bias, mode="reset", params={"bias_range": bias}
      ),
      "torso_com": EventTermCfg(
        func=dr.body_com_offset,
        mode="startup",
        params={"asset_cfg": torso, "ranges": com, "operation": "add"},
      ),
    }
  return ManagerBasedRlEnvCfg(
    seed=config.seed,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      num_envs=config.num_envs,
    ),
    actions={
      "joint_pos": (
        SemanticJointPositionActionCfg
        if config.robot == "bsrl_closed_ankle"
        else JointPositionActionCfg
      )(
        entity_name="robot",
        actuator_names=(".*",),
        scale={
          p: value
          for p, value in action_scale.items()
          if any(re.fullmatch(p, name) for name in joint_names)
        },
        use_default_offset=True,
      )
    },
    events=events,
    sim=SimulationCfg(
      # Falling and dancing need full-body ground contacts and adequate capacity.
      nconmax=128,
      njmax=1024,
      mujoco=FoundationMujocoCfg(
        # The explicit linkage uses the 1 ms integration rate validated by the
        # Bipedal_AMP closed-ankle model.  It also keeps the 2 ms equality
        # time-constant above MuJoCo's two-timestep safety clamp.
        timestep=0.001 if closed_ankle else 0.005,
        iterations=12 if closed_ankle else 10,
        ls_iterations=20,
      ),
    ),
    decimation=20 if closed_ankle else 4,
    episode_length_s=config.episode_length_s,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=config.layout.root_body,
    ),
  )
