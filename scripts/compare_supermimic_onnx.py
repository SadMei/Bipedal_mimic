"""Record synchronized ONNX policies with their matched world-model forecasts.

Each panel starts the same motion at frame zero. Physics stops at its first
height/tilt failure; that panel is then held and explicitly marked. Orange is
a future prediction, blue is the XY/yaw-aligned reference, solid is physics.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNTIME = Path(os.environ.get("SUPERMIMIC_ORT_PATH", ROOT.parent / ".setup/onnx_runtime"))
if RUNTIME.exists():
  sys.path.insert(0, str(RUNTIME))
import onnx
import onnxruntime as ort
import mujoco

from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.rl.inference import load_actor, load_world_model, validate_environment
from src.tasks.supermimic.rl.evaluation import ERROR_METRICS
from src.tasks.supermimic.rl.world_model_playback import WorldModelFuturePreview
from src.tasks.supermimic.rl import visualization


def write_json(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class ActorExport(torch.nn.Module):
  def __init__(self, actor):
    super().__init__()
    self.actor = actor

  def forward(self, history, reference_features):
    return self.actor.forward_features(history, reference_features)


def session(path):
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  options.inter_op_num_threads = 1
  return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


@torch.no_grad()
def export_actor(actor, saved, checkpoint, path):
  actor.cpu().eval()
  rng = torch.Generator().manual_seed(4401)
  history = torch.randn(64, 10, actor.observation_dim, generator=rng)
  goal = torch.randn(64, actor.reference_dim, generator=rng)
  if actor.input_normalization:
    history = actor.obs_norm.mean.float() + history * actor.obs_norm.std
    goal = actor.goal_norm.mean.float() + goal * actor.goal_norm.std
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    ActorExport(actor), (history[:1], goal[:1]), str(path),
    input_names=["history", "reference_features"], output_names=["action"],
    dynamic_axes={"history": {0: "batch"}, "reference_features": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17, dynamo=False,
  )
  graph = onnx.load(str(path))
  metadata = {
    "iteration": str(saved["iteration"]),
    "source_checkpoint": str(checkpoint.resolve()),
    "source_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
    "reference_schema": saved["reference_schema"],
    "observation_dim": str(actor.observation_dim),
    "reference_dim": str(actor.reference_dim),
    "normalization": "checkpoint observation and goal normalization included in graph",
    "world_model_type": saved["config"].get("world_model_type", "mlp"),
    "input_contract": "history [B,10,obs_dim]; reference_features [B,reference_dim]; output action [B,12]",
    "joint_names": json.dumps(saved["robot_metadata"]["joint_names"]),
  }
  onnx.helper.set_model_props(graph, metadata)
  onnx.checker.check_model(graph)
  onnx.save(graph, str(path))
  runtime = session(path)
  expected = actor.forward_features(history, goal).numpy()
  actual = runtime.run(["action"], {"history": history.numpy(), "reference_features": goal.numpy()})[0]
  np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
  error = float(np.max(np.abs(actual-expected)))
  print(f"ONNX validated: {path.name}, max error={error:.3g}", flush=True)
  return runtime, dict(metadata, validation_samples=64, validation_max_abs_error=error)


class OnnxActor:
  def __init__(self, original, runtime):
    self.original, self.runtime = original, runtime
    for key in ("default_q", "reference_features", "relative_orientation", "motion_anchor_ori_b"):
      setattr(self, key, getattr(original, key))
    self.max_action_error = 0.0
    self.real_calls = 0
    self.imagined_calls = 0

  @torch.no_grad()
  def forward_features(self, history, goal):
    self.imagined_calls += 1
    action = self.runtime.run(["action"], {
      "history": history.detach().cpu().numpy(),
      "reference_features": goal.detach().cpu().numpy(),
    })[0]
    if not np.isfinite(action).all():
      raise FloatingPointError("ONNX produced nonfinite actions")
    return torch.from_numpy(action).to(history.device)

  @torch.no_grad()
  def __call__(self, history, reference):
    goal = self.reference_features(reference, self.default_q)
    action = self.forward_features(history, goal)
    self.imagined_calls -= 1
    self.real_calls += 1
    expected = self.original.forward_features(history, goal)
    error = float((action-expected).abs().max())
    self.max_action_error = max(self.max_action_error, error)
    torch.testing.assert_close(action, expected, atol=5e-5, rtol=5e-5)
    return action


class OnnxFuturePreview(WorldModelFuturePreview):
  """Use the full ONNX Actor, including both normalizers, in both WM variants."""

  @torch.no_grad()
  def _rollout(self, clip, frame, steps):
    motion = torch.tensor([clip], device=self.env.device, dtype=torch.long)
    frames = torch.tensor([frame], device=self.env.device, dtype=torch.long)
    future = self.env.library.get_future(motion, frames, steps + 10)
    windows = {key: value.unfold(1, 10, 1).movedim(-1, 2) for key, value in future.items()}
    goals = self.actor.reference_features(windows, self.actor.default_q)
    state, history = self.actual.clone(), self.history.clone()
    world_history = self.real_world_history.clone() if self.model.recurrent else None
    states = [state]
    for step in range(steps):
      action = self._executed(self.actor.forward_features(history, goals[:, step]))
      if self.model.recurrent:
        state, world_history = self.model.step(state, action, self.context, world_history)
      else:
        state = self.model(state, action, self.context)
      observation = self.model.kinematics.observation(
        state, action, self.context, future["body_quat_w"][:, step+1],
        relative_orientation=self.actor.relative_orientation,
        motion_anchor_orientation=self.actor.motion_anchor_ori_b,
      )
      history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
      states.append(state)
    return torch.stack(states)


class PanelRecorder:
  width, render_height, header, footer = 960, 640, 96, 80

  def __init__(self, env, preview, path, title, subtitle, clip, fps):
    visualization.WORLD_MODEL_GHOST_RGBA = (1.0, 0.42, 0.03, 0.45)
    visualization.WORLD_MODEL_PATH_RGBA = (1.0, 0.42, 0.03, 0.85)
    visualization.REFERENCE_GHOST_RGBA = (0.20, 0.60, 1.0, 0.25)
    self.scene = visualization.EvaluationScene(env, ghost=True, global_ghost=False, world_model=preview)
    self.scene.camera.distance = 2.8
    self.scene.camera.azimuth = 135
    self.scene.camera.elevation = -15
    self.scene.model.vis.global_.offwidth = self.width
    self.scene.model.vis.global_.offheight = self.render_height
    self.renderer = mujoco.Renderer(self.scene.model, height=self.render_height, width=self.width)
    self.height = self.header + self.render_height + self.footer
    self.title, self.subtitle, self.clip = title, subtitle, clip
    self.preview, self.path, self.frames = preview, path, 0
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    self.font = ImageFont.truetype(font, 19)
    self.small = ImageFont.truetype(font, 16)
    self.big = ImageFont.truetype(bold, 29)
    self.encoder = subprocess.Popen([
      "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
      "-s", f"{self.width}x{self.height}", "-r", str(fps), "-i", "-", "-an",
      "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "18",
      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)

  def frame(self, env, step, state, metrics, failure=None):
    self.scene.update(env, 0, step, state)
    self.scene.camera.lookat[2] -= 0.1
    self.renderer.update_scene(self.scene.data, camera=self.scene.camera)
    self.scene.add_ghost(self.renderer.scene)
    canvas = Image.new("RGB", (self.width, self.height), (16, 22, 31))
    canvas.paste(Image.fromarray(self.renderer.render()), (0, self.header))
    draw = ImageDraw.Draw(canvas)
    accent = (98, 216, 211) if self.title.startswith("GRU") else (188, 167, 255)
    draw.text((22, 10), self.title, font=self.big, fill=accent)
    draw.text((22, 49), self.subtitle, font=self.font, fill=(220, 225, 234))
    draw.text((22, 75), self.clip + " | same initial state, seed 0", font=self.small, fill=(167, 181, 197))
    draw.text((744, 17), f"t = {step*env.step_dt:6.2f} s", font=self.font, fill=(238, 241, 246))
    top = self.header+14
    draw.rounded_rectangle((14, top-5, 655, top+36), radius=7, fill=(20, 28, 38))
    draw.text((25, top), "Solid: physics", font=self.small, fill=(235, 235, 235))
    draw.text((158, top), "Blue: aligned reference", font=self.small, fill=(90, 173, 255))
    draw.text((371, top), f"Orange: WM +{self.preview.horizon*env.step_dt:.2f}s", font=self.small, fill=(255, 147, 55))
    y = self.header+self.render_height+9
    body = metrics.get("body_position_error")
    root = metrics.get("root_position_error")
    track = "Tracking error: --" if body is None else f"Body tracking: {body*100:.1f} cm  |  Root drift: {root:.2f} m"
    draw.text((22, y), track, font=self.font, fill=(225, 232, 240))
    error = self.preview.metrics
    prediction = (f"WM matched-time error: body {error['body_position_mpjpe_m']*100:.1f} cm | root {error['root_position_m']:.3f} m"
                  if error else "WM matched-time error: waiting for the forecast timestamp")
    if self.preview.diverged:
      prediction = "WM forecast diverged; nonfinite ghost hidden"
    draw.text((22, y+30), prediction, font=self.small, fill=(255, 167, 80))
    if failure:
      box = (100, self.header+self.render_height-115, self.width-100, self.header+self.render_height-20)
      draw.rounded_rectangle(box, radius=9, fill=(102, 28, 35))
      draw.text((box[0]+20, box[1]+12), f"EPISODE ENDED at {step*env.step_dt:.2f}s", font=self.big, fill=(255, 234, 232))
      draw.text((box[0]+20, box[1]+55), f"{failure} | final physical frame held", font=self.font, fill=(255, 205, 200))
    self.encoder.stdin.write(canvas.tobytes())
    if self.frames == 0 or step in (250, 500, 750, 1000) or failure:
      canvas.save(self.path.with_name(self.path.stem+f"_frame_{step:06d}.jpg"), quality=92)
    self.frames += 1

  def close(self):
    self.encoder.stdin.close()
    status = self.encoder.wait()
    self.renderer.close()
    if status:
      raise RuntimeError(f"ffmpeg exited with {status}")


@torch.no_grad()
def record(args, run, label):
  from src.tasks.supermimic.env import SuperMimicEnv
  checkpoint = run / f"policy_iteration_{args.iteration:06d}.pt"
  world_checkpoint = run / f"world_model_iteration_{args.iteration:06d}.pt"
  actor, saved, _ = load_actor(checkpoint, None, "cpu")
  world, world_saved = load_world_model(world_checkpoint, None, args.device)
  if saved["iteration"] != args.iteration or world_saved["iteration"] != args.iteration:
    raise ValueError("Policy and world model must match the requested iteration")
  if label.lower() != world.world_model_type:
    raise ValueError("Left must be GRU and right must be MLP")
  cfg = saved["config"]
  onnx_path = args.output / f"{label.lower()}_policy_iteration_{args.iteration:06d}.onnx"
  runtime, export = export_actor(actor, saved, checkpoint, onnx_path)
  actor = actor.to(args.device).eval()
  policy = OnnxActor(actor, runtime)
  env = SuperMimicEnv(FoundationConfig(
    robot=cfg["robot"], clips=(args.clip,), num_envs=1, seed=0,
    episode_length_s=1e9, terminate_on_failure=True,
    relative_orientation=cfg["relative_orientation"], motion_anchor_ori_b=cfg["motion_anchor_ori_b"],
  ), device=args.device)
  recorder = None
  start = time.perf_counter()
  try:
    validate_environment(env, saved)
    validate_environment(env, world_saved)
    inputs, _ = env.reset(seed=0, options={"motion_id": 0, "frame_id": 0})
    state = env.read_state().detach().clone()
    initial_state = state.cpu().tolist()
    preview = OnnxFuturePreview(env, world, policy, horizon=args.prediction_horizon, refresh=4)
    observed_policy = preview.policy(policy)
    preview.on_frame(env, 0, 0, state, 0.0)
    params = sum(p.numel() for p in world.parameters())
    path = args.output / f"{label.lower()}_panel.mp4"
    recorder = PanelRecorder(env, preview, path, f"{label} WORLD MODEL", f"Iteration {args.iteration:,} | ONNX Actor | WM {params/1e6:.2f}M params", args.clip, 25)
    recorder.frame(env, 0, state, {})
    max_steps = min(int(round(args.seconds/env.step_dt)), int(env.library.lengths[0])-1)
    sums, reward_sum, rows, failure = {}, 0.0, [], None
    for step in range(1, max_steps+1):
      action = observed_policy(inputs["history"], inputs["future_reference"])
      inputs, reward, terminal, cutoff, extras = env.step(action)
      state = extras["transition"]["next_state"].detach().clone()
      if not bool(torch.isfinite(state).all() and torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite physics rollout")
      reward_sum += float(reward[0])
      metrics = {k: float(extras["metrics"][k][0]) for k in ERROR_METRICS}
      for k, value in metrics.items():
        sums[k] = sums.get(k, 0.0) + value
      reasons = [k for k in ("height_failure", "tilt_failure") if bool(extras["metrics"][k][0])]
      if bool(terminal[0]):
        failure = "+".join(reasons) or "physical_failure"
      elif bool(cutoff[0]):
        failure = "clip end / time limit"
      preview.on_frame(env, 0, step, state, reward_sum)
      rows.append(dict(step=step, seconds=step*env.step_dt, **metrics, failure=failure,
                       world_forecast_error=preview.metrics))
      if step % 2 == 0 or failure:
        recorder.frame(env, step, state, metrics, failure)
      if step % 100 == 0 or failure:
        print(f"{label}: {step}/{max_steps} | t={step*env.step_dt:.2f}s | body={metrics['body_position_error']:.3f}m | {failure or 'running'}", flush=True)
      if failure:
        break
    report = dict(label=label, checkpoint=str(checkpoint), world_checkpoint=str(world_checkpoint),
                  iteration=args.iteration, motion=args.clip, seed=0, initial_state=initial_state,
                  seconds=step*env.step_dt, steps=step, requested_seconds=args.seconds,
                  failure=failure, first_failure_seconds=step*env.step_dt if failure else None,
                  mean_errors={k:v/step for k,v in sums.items()}, reward_per_step=reward_sum/step,
                  actor_onnx=export, live_action_max_abs_error=policy.max_action_error,
                  actual_onnx_calls=policy.real_calls, imagined_onnx_calls=policy.imagined_calls,
                  world_model=preview.report(), elapsed_seconds=time.perf_counter()-start,
                  video=str(path), video_frames=recorder.frames,
                  reference_display="aligned XY/yaw, matching body tracking reward; root drift reported separately",
                  prediction_display="orange transparent future pose, +0.64s, using matched world weights and ONNX policy",
                  physics="MuJoCo Warp; original BSRL configuration; standard deterministic evaluation; stop on first failure")
    write_json(args.output/f"{label.lower()}_report.json", report)
    write_json(args.output/f"{label.lower()}_timeseries.json", rows)
    return report
  finally:
    if recorder:
      recorder.close()
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--left-run", type=Path, required=True)
  parser.add_argument("--right-run", type=Path, required=True)
  parser.add_argument("--iteration", type=int, default=1000)
  parser.add_argument("--clip", default="walk1_subject1")
  parser.add_argument("--seconds", type=float, default=40.0)
  parser.add_argument("--prediction-horizon", type=int, default=32)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error("Use a new output directory to preserve prior recordings")
  if args.seconds <= 0 or args.prediction_horizon < 1:
    parser.error("Duration and prediction horizon must be positive")
  args.output.mkdir(parents=True)
  torch.set_num_threads(2)
  torch.backends.cuda.matmul.allow_tf32 = False
  reports = [record(args, args.left_run, "GRU"), record(args, args.right_run, "MLP")]
  np.testing.assert_allclose(reports[0]["initial_state"], reports[1]["initial_state"], atol=0, rtol=0)
  video = args.output / f"gru_vs_mlp_iter{args.iteration:06d}_{args.clip}.mp4"
  filters = (
    f"[0:v]tpad=stop_mode=clone:stop_duration={args.seconds},trim=duration={args.seconds},setpts=PTS-STARTPTS[l];"
    f"[1:v]tpad=stop_mode=clone:stop_duration={args.seconds},trim=duration={args.seconds},setpts=PTS-STARTPTS[r];"
    "[l][r]hstack=inputs=2,drawbox=x=958:y=0:w=4:h=ih:color=0x10161f:t=fill[v]"
  )
  subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-i", str(args.output/"gru_panel.mp4"),
    "-i", str(args.output/"mlp_panel.mp4"), "-filter_complex_threads", "1",
    "-filter_complex", filters, "-map", "[v]", "-an", "-c:v", "libx264",
    "-threads", "4", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart", str(video),
  ], check=True)
  write_json(args.output/"comparison.json", dict(video=str(video), left="GRU", right="MLP",
      iteration=args.iteration, motion=args.clip, seconds=args.seconds,
      same_initial_state=True, models=reports, versions=dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__, mujoco=mujoco.__version__)))
  print(f"COMPARISON VIDEO: {video}", flush=True)


if __name__ == "__main__":
  main()
