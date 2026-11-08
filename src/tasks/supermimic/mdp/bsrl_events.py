"""BSRL-specific dynamics events shared with the SuperMimic environment."""

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


@requires_model_fields(
  "body_mass", "body_inertia", recompute=RecomputeLevel.set_const
)
def scale_body_mass_and_inertia(
  env,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Uniformly scale each body's mass and inertia as Bipedal_AMP does."""
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)
  local_ids = asset_cfg.body_ids
  global_ids = asset.indexing.body_ids[local_ids]
  if global_ids.ndim == 0:
    global_ids = global_ids[None]
  scale = torch.empty(
    (len(env_ids), len(global_ids), 1), device=env.device
  ).uniform_(*scale_range)
  default_mass = env.sim.get_default_field("body_mass")[global_ids]
  default_inertia = env.sim.get_default_field("body_inertia")[global_ids]
  env.sim.model.body_mass[env_ids[:, None], global_ids] = default_mass * scale[..., 0]
  env.sim.model.body_inertia[env_ids[:, None], global_ids] = default_inertia * scale
