"""Causal history, autoregressive training, BPTT and portable GRU checkpoints."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.models.kinematics import G1Kinematics, robot_metadata
from src.tasks.supermimic.models.policy import MotionActor, MotionCritic
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.diagnostics import replay_diagnostics
from src.tasks.supermimic.rl.imagination import frozen_parameters, imagine
from src.tasks.supermimic.rl.inference import load_world_model
from src.tasks.supermimic.rl.world_model_training import rollout, world_model_loss
from src.tasks.supermimic.rl.world_model_playback import WorldModelPlayback, WorldModelFuturePreview
from tests.supermimic_fixtures import make_replay


class RecurrentWorldTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)
    cls.directory = tempfile.TemporaryDirectory()
    cls.metadata = robot_metadata("bsrl_12dof")
    cls.manifest, path, _ = make_replay(Path(cls.directory.name), cls.metadata)
    cls.snapshot = torch.load(path, weights_only=True)["replay"]

  @classmethod
  def tearDownClass(cls):
    cls.directory.cleanup()

  def setUp(self):
    torch.manual_seed(127)
    self.replay = ReplayBuffer.from_state_dict(self.snapshot)

  def model(self, device="cpu", normalized=True):
    model = WorldModel(
      self.metadata, (16,), world_model_type="gru", world_history_length=32,
      world_gru_hidden_size=16, world_gru_layers=2,
      world_model_normalization=normalized,
    ).to(device)
    # Initialization is zero by design; nonzero trained heads are needed to
    # test the full action -> GRU -> physical state Jacobian.
    torch.nn.init.normal_(model.dynamics[-1].weight, std=0.003)
    return model

  def library(self, device="cpu"):
    return MotionLibrary(
      self.manifest, self.metadata["joint_names"], G1Kinematics(self.metadata).body_names,
      device=device,
    )

  def test_prefix_excludes_current_action_and_pads_only_true_resets(self):
    starts = torch.tensor([[0, 0], [3, 1], [48, 2], [51, 3], [79, 0]])
    batch = self.replay.gather_windows(starts, 8, world_history_length=32)
    self.assertEqual(batch["world_history_mask"].sum(1).tolist(), [0, 3, 0, 3, 31])
    for row, (step, env) in enumerate(starts.tolist()):
      mask = batch["world_history_mask"][row]
      count = int(mask.sum())
      for name in ("state", "action", "context"):
        actual = batch["world_history_" + name][row]
        self.assertEqual(float(actual[~mask].abs().sum()), 0)
        torch.testing.assert_close(actual[mask], self.replay.data[name][step-count:step, env])
    # Changing the logged action AT the rollout start cannot change its prefix.
    before = batch["world_history_action"][3].clone()
    self.replay.data["action"][51, 3].fill_(123)
    after = self.replay.gather_windows(starts[3:4], 8, world_history_length=32)
    torch.testing.assert_close(before, after["world_history_action"][0])

  def test_overwritten_history_rejected_and_sampler_remains_causal(self):
    for step in range(20):
      transition = {name: value[48 + step].clone() for name, value in self.snapshot["data"].items()}
      transition["episode_id"].fill_(2)
      self.replay.append(transition)
    with self.assertRaisesRegex(ValueError, "history"):
      self.replay.gather_windows(torch.tensor([[25, 0]]), 8, world_history_length=32)
    batch = self.replay.sample_windows(100, 8, world_history_length=32)
    self.assertTrue((batch["world_history_mask"].sum(1) <= batch["episode_step"][:, 0]).all())
    starts = self.replay.sample_starts(100, world_history_length=32)
    self.assertTrue((starts["world_history_mask"].sum(1) <= starts["episode_step"]).all())
    # Save/restore the RNG at the same point, then compare complete sampled batches.
    retained = ReplayBuffer.from_state_dict(self.replay.state_dict())
    a, b = self.replay.sample_windows(4, world_history_length=32), retained.sample_windows(4, world_history_length=32)
    for key in a:
      torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

  def test_history_changes_prediction_and_padding_stays_zero_after_normalization(self):
    model = self.model()
    batch = self.replay.gather_windows(torch.tensor([[0, 0], [35, 0]]), 8, world_history_length=32)
    model.update_normalization(batch["state"].flatten(0, 1), batch["action"].flatten(0, 1),
                               batch["context"].flatten(0, 1), batch["next_state"].flatten(0, 1))
    history = model.history_from_batch(batch)
    self.assertEqual(float(history[0].abs().sum()), 0)
    state, action, context = (batch[name][:1, 0] for name in ("state", "action", "context"))
    a, _ = model.step(state, action, context, history[:1])
    b, _ = model.step(state, action, context, history[1:])
    self.assertGreater(float((a-b).detach().abs().max()), 1e-7)
    with self.assertRaisesRegex(ValueError, "history"):
      model(state, action, context)

  def test_eight_step_supervision_trains_gru_and_masks_terminal_tail(self):
    model = self.model()
    batch = self.replay.gather_windows(torch.tensor([[35, 0], [47, 1]]), 8, world_history_length=32)
    prediction = rollout(model, batch, self.library())
    loss, _ = world_model_loss(prediction, batch, analytic_losses=False)
    loss.backward()
    norm = sum(float(p.grad.abs().sum()) for p in model.gru.parameters())
    self.assertGreater(norm, 0)
    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.gru.parameters()))
    torch.testing.assert_close(prediction["next_state"][1], prediction["next_state"][1, :1].expand(8, -1))

  def check_bptt(self, device, eval_mode=False):
    model = self.model(device)
    if eval_mode:
      model.eval()
    actor = MotionActor(self.metadata["default_q"], (16,)).to(device)
    critic = MotionCritic(self.metadata["default_q"], (16,)).to(device)
    replay = ReplayBuffer.from_state_dict(self.snapshot, device=device)
    library = self.library(device)
    starts = replay.sample_starts(2, horizon=32, clip_lengths=library.lengths, world_history_length=32)
    with frozen_parameters(model, critic):
      trace = imagine(actor, model, critic, library, starts, horizon=32, sensor_noise=False)
      trace["actions"][0].retain_grad()
      # Only final physical state: no direct action penalty can make this pass.
      trace["states"][:, -1].square().mean().backward()
    grad = trace["actions"][0].grad
    self.assertTrue(torch.isfinite(grad).all())
    self.assertGreater(float(grad.abs().sum()), 0)
    self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None), 0)
    self.assertTrue(all(p.grad is None for m in (model, critic) for p in m.parameters()))
    self.assertTrue(all(p.requires_grad for p in model.parameters()))

  def test_32_step_physical_state_gradient_reaches_first_action_cpu(self):
    self.check_bptt("cpu")

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
  def test_32_step_physical_state_gradient_reaches_first_action_cuda(self):
    self.check_bptt("cuda:0")

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
  def test_cuda_eval_mode_input_gradient_avoids_cudnn_inference_backward_error(self):
    self.check_bptt("cuda:0", eval_mode=True)

  def test_diagnostics_and_checkpoint_roundtrip(self):
    model = self.model()
    snapshots = {}
    rng = self.replay.generator.get_state().clone()
    report = replay_diagnostics(model, self.replay, self.library(), 2, snapshots, 1)
    self.assertTrue(model.training)
    torch.testing.assert_close(rng, self.replay.generator.get_state())
    for horizon in ("1", "8", "32"):
      self.assertTrue(report["horizons"][horizon]["model"]["finite"])
      self.assertIn("world_history_mask", snapshots[horizon]["batch"])
    config = dict(world_dims=(16,), world_model_type="gru", world_history_length=32,
                  world_gru_hidden_size=16, world_gru_layers=2, world_prediction="state_delta",
                  world_model_normalization=True)
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "world.pt"
      torch.save(dict(kind="supermimic_world_model", schema_version=1, config=config,
                      robot_metadata=self.metadata, simulation={}, clips=[], world=model.state_dict()), path)
      restored, _ = load_world_model(path, None, "cpu")
    batch = self.replay.sample_windows(2, world_history_length=32)
    a = rollout(model, batch, self.library())["next_state"]
    b = rollout(restored, batch, self.library())["next_state"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)

  def test_config_retains_mlp_default_and_rejects_unretained_gru_context(self):
    self.assertEqual(OnlineConfig().world_model_type, "mlp")
    with self.assertRaisesRegex(ValueError, "no-compile"):
      OnlineConfig(world_model_type="gru")
    with self.assertRaisesRegex(ValueError, "capacity"):
      OnlineConfig(world_model_type="gru", compile=False, capacity=32)

  def test_playback_resync_uses_real_history_and_future_preview_resets_at_clip_start(self):
    model = self.model().eval()
    library = self.library()
    state = self.replay.data["state"][0, :1]
    context = self.replay.data["context"][0, :1]
    observation = self.replay.data["observation"][0, :1]
    env = SimpleNamespace(num_envs=1, context=context, device="cpu", library=library,
                          step_dt=0.02, foundation_config=SimpleNamespace(action_clip=None),
                          policy_input=lambda: {"history": observation[:, None].expand(-1, 10, -1)})
    playback = WorldModelPlayback(env, model, horizon=2)
    playback.on_frame(env, 0, 0, state, 0)
    real_history = model.empty_history(state)
    for t in range(3):
      action = self.replay.data["action"][t, :1]
      actual = self.replay.data["state"][t, :1]
      if t == 2:
        expected, _ = model.step(actual, action, context, real_history)
      playback.predict(action)
      real_history = model.append_history(real_history, actual, action, context)
      torch.testing.assert_close(playback.real_world_history, real_history)
      playback.on_frame(env, 0, t+1, self.replay.data["next_state"][t, :1], 0)
    torch.testing.assert_close(playback.state, expected)
    actor = MotionActor(self.metadata["default_q"], (16,), input_normalization=True)
    goal = actor.reference_features(library.get_future(torch.tensor([0]), torch.tensor([0])), actor.default_q)
    actor.update_normalization(torch.cat((observation, observation + 0.5)), torch.cat((goal, goal + 0.5)))
    actor.eval()
    preview = WorldModelFuturePreview(env, model, actor, horizon=2, refresh=1)
    preview.on_frame(env, 0, 0, state, 0)
    self.assertEqual(float(preview.real_world_history.abs().sum()), 0)
    history = env.policy_input()["history"]
    reference = library.get_future(torch.tensor([0]), torch.tensor([0]))
    action = preview.policy(actor)(history, reference)
    expected, _ = model.step(state, action, context, model.empty_history(state))
    torch.testing.assert_close(preview.cache[1], expected)
    self.assertGreater(float(preview.real_world_history.abs().sum()), 0)
    preview.on_frame(env, 1, 0, state, 0)
    self.assertEqual(float(preview.real_world_history.abs().sum()), 0)


if __name__ == "__main__":
  unittest.main()
