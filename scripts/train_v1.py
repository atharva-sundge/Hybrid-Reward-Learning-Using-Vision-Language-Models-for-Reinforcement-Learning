import os
import argparse
from typing import Optional, Dict, Callable

import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import clip

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback


# --------------------------
# Prompts per environment
# --------------------------
DEFAULT_PROMPTS: Dict[str, Dict[str, Optional[str]]] = {
    "CartPole-v1": {
        "goal": "the pole remains vertical and balanced on the moving cart",
        "baseline": "the pole is falling or has fallen and the cart is unstable"
    },
    "MountainCar-v0": {
        "goal": "the car reaches the flag on the right hill at the top",
        "baseline": "the car stays near the bottom of the valley and does not reach the flag"
    },
    # Add more envs here as needed.
}


# -------------------------------------------------
# CLIP-based reward
# --------------------------------------------------
class CLIPReward:
    """
    Frozen CLIP reward:
      r_clip = sigmoid( k * (sim(goal) - sim(baseline)) )
    If baseline is None:
      r_clip = (sim(goal)+1)/2  (maps cosine similarity [-1,1] -> [0,1])
    """
    def __init__(
        self,
        device: torch.device,
        goal_prompt: str,
        baseline_prompt: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        k: float = 5.0):  # sharpness for sigmoid)
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        with torch.no_grad():
            goal_tokens = clip.tokenize([goal_prompt]).to(device)
            self.goal_text = self.model.encode_text(goal_tokens)
            self.goal_text = self.goal_text / self.goal_text.norm(dim=-1, keepdim=True)

            self.has_baseline = baseline_prompt is not None and len(baseline_prompt.strip()) > 0
            if self.has_baseline:
                base_tokens = clip.tokenize([baseline_prompt]).to(device)
                self.base_text = self.model.encode_text(base_tokens)
                self.base_text = self.base_text / self.base_text.norm(dim=-1, keepdim=True)
            else:
                self.base_text = None

        self.k = k
        self.to_pil = transforms.ToPILImage()

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> float:
        """
        frame: HxWxC RGB numpy array -> r_clip in [0,1]
        """
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB frame, got shape {frame.shape}")

        img = Image.fromarray(frame)
        img_in = self.preprocess(img).unsqueeze(0).to(self.device)

        image_feat = self.model.encode_image(img_in)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        sim_goal = (image_feat @ self.goal_text.T).item()  # [-1,1]
        if self.has_baseline:
            sim_base = (image_feat @ self.base_text.T).item()
            score = sim_goal - sim_base
            r = torch.sigmoid(torch.tensor(self.k * score, device=self.device)).item()
        else:
            r = 0.5 * (sim_goal + 1.0)

        return float(np.clip(r, 0.0, 1.0))


# -------------------------------------------------------
# hybrid reward = env reward + clip reward
# -------------------------------------------------------
class HybridRewardEnv(gym.Wrapper):
    """
    Adds CLIP reward and blends with env reward:
      R_hybrid = (1 - alpha) * R_env_norm + alpha * R_clip

    For odd reward scales, env_reward_norm=True squashes env reward to [0,1].
    """
    def __init__(
        self,
        env: gym.Env,
        clip_reward: CLIPReward,
        alpha: float = 0.7,
        env_reward_norm: bool = True):
        super().__init__(env)
        self.clip_reward = clip_reward
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.env_reward_norm = env_reward_norm

        # running min/max for generic normalization
        self.running_min = 0.0
        self.running_max = 1.0
        self._t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t = 0
        return obs, info

    def step(self, action):
        self._t += 1
        obs, r_env, terminated, truncated, info = self.env.step(action)

        # render frame for CLIP reward
        frame = self.env.render()  # requires env created with render_mode="rgb_array"
        r_clip = self.clip_reward(frame)

        # Normalize env reward to ~[0,1]
        if self.env_reward_norm:
            self.running_min = 0.99 * self.running_min + 0.01 * min(self.running_min, r_env)
            self.running_max = 0.99 * self.running_max + 0.01 * max(self.running_max, r_env)
            rng = max(1e-6, (self.running_max - self.running_min))
            r_env_norm = (r_env - self.running_min) / rng
            r_env_norm = float(np.clip(r_env_norm, 0.0, 1.0))
        else:
            r_env_norm = r_env

        r_hybrid = (1.0 - self.alpha) * r_env_norm + self.alpha * r_clip

        info = dict(info)
        info["r_env"] = r_env
        info["r_env_norm"] = r_env_norm
        info["r_clip"] = r_clip
        info["r_hybrid"] = r_hybrid
        return obs, r_hybrid, terminated, truncated, info



# logs
class EpisodeStatsCallback(BaseCallback):
    """
    Logs per-episode stats (return, mean r_clip, mean r_hybrid) to CSV in save_dir.
    """
    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        import csv
        self.csv = csv
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, "episode_stats.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = self.csv.DictWriter(self._file, fieldnames=["episode","return","mean_r_clip","mean_r_hybrid","length"])
        self._writer.writeheader()
        self._ep = 0
        self._acc_clip = 0.0
        self._acc_hybrid = 0.0
        self._acc_return = 0.0
        self._acc_len = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if infos:
            info0 = infos[0]
            self._acc_clip += info0.get("r_clip", 0.0)
            self._acc_hybrid += info0.get("r_hybrid", 0.0)
        if rewards is not None:
            self._acc_return += float(rewards[0])
            self._acc_len += 1

        if dones is not None and bool(dones[0]):
            self._ep += 1
            mean_clip = self._acc_clip / max(1, self._acc_len)
            mean_hybrid = self._acc_hybrid / max(1, self._acc_len)
            self._writer.writerow({
                "episode": self._ep,
                "return": self._acc_return,
                "mean_r_clip": mean_clip,
                "mean_r_hybrid": mean_hybrid,
                "length": self._acc_len
            })
            self._file.flush()
            self._acc_clip = self._acc_hybrid = self._acc_return = 0.0
            self._acc_len = 0
        return True

    def _on_training_end(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass



# plots
def _auto_plot_returns(save_dir: str, exp_name: str):
    import glob, json, pandas as pd, matplotlib.pyplot as plt
    files = sorted(glob.glob("./**/monitor.csv*", recursive=True))
    if not files:
        print("[plot] no monitor.csv found; skipping."); return
    dfs = []
    for f in files:
        with open(f, "r") as fh:
            first = fh.readline()
            if first.startswith("#"):
                try: meta = json.loads(first[1:])
                except Exception: meta = {}
            else:
                fh.seek(0); meta = {}
            df = pd.read_csv(fh)
            df["source_file"] = f
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).sort_values("l")
    df["ep_idx"] = range(1, len(df)+1)
    roll = df["r"].rolling(50, min_periods=1).mean()
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    out = os.path.join(save_dir, "plots", f"{exp_name}_returns.png")
    plt.figure(figsize=(8,4))
    plt.plot(df["ep_idx"], df["r"], alpha=0.3, label="episode return")
    plt.plot(df["ep_idx"], roll, lw=2, label="moving avg (50)")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title(f"Training: {exp_name}")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[plot] saved -> {out}")


# --------------------------
# Utilities: make env, vec env, train, eval
# --------------------------
def make_env(env_id: str, goal: str, baseline: Optional[str], alpha: float,
             device: torch.device, k_sigmoid: float = 5.0):
    env = gym.make(env_id, render_mode="rgb_array")
    env = Monitor(env)  # writes monitor.csv in CWD
    clip_r = CLIPReward(device=device, goal_prompt=goal, baseline_prompt=baseline, k=k_sigmoid)
    env = HybridRewardEnv(env, clip_reward=clip_r, alpha=alpha, env_reward_norm=True)
    return env


def _make_vec_env(env_id: str, goal: str, baseline: Optional[str], alpha: float,
                  device: torch.device, num_envs: int, subproc: bool,
                  k_sigmoid: float = 5.0):
    def thunk() -> gym.Env:
        return make_env(env_id, goal, baseline, alpha, device, k_sigmoid)
    if num_envs <= 1:
        return DummyVecEnv([thunk])
    return (SubprocVecEnv if subproc else DummyVecEnv)([thunk for _ in range(num_envs)])


def train(
    env_id: str,
    algo: str = "PPO",
    alpha: float = 0.7,
    steps: int = 200_000,
    seed: int = 0,
    device_str: Optional[str] = None,
    goal: Optional[str] = None,
    baseline: Optional[str] = None,
    save_dir: str = "runs",
    num_envs: int = 1,
    subproc_env: bool = True,
    auto_plot: bool = False,
    k_sigmoid: float = 5.0):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if goal is None or len(goal.strip()) == 0:
        if env_id not in DEFAULT_PROMPTS:
            raise ValueError(f"No default prompts for env '{env_id}'. Provide --goal and optionally --baseline.")
        goal = DEFAULT_PROMPTS[env_id]["goal"]
        baseline = DEFAULT_PROMPTS[env_id]["baseline"] if baseline is None else baseline

    vec_env = _make_vec_env(env_id, goal, baseline, alpha, device,
                            num_envs=num_envs, subproc=subproc_env, k_sigmoid=k_sigmoid)

    # Choose algo
    algo = algo.upper()
    if algo == "PPO":
        # Heavier PPO update to utilize GPU better with parallel envs
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=seed,
            device=device,
            n_steps=2048,                                  # per env
            batch_size=min(65536, max(2048 * num_envs // 2, 64)),
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
        )
    elif algo == "DQN":
        model = DQN(
            "MlpPolicy", vec_env, verbose=1, seed=seed, device=device,
            learning_rate=1e-3, buffer_size=100_000, learning_starts=1_000,
            batch_size=64, target_update_interval=1000
        )
    else:
        raise ValueError(f"Unsupported --algo {algo}. Use PPO or DQN.")

    exp_name = f"{env_id}_{algo}_alpha{alpha}"
    model_path = os.path.join(save_dir, f"{exp_name}.zip")
    print(f"[train] env={env_id} algo={algo} alpha={alpha} steps={steps} device={device} "
          f"num_envs={num_envs} subproc={subproc_env} -> {model_path}")

    # ---- checkpoints & best model ----
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    eval_env = _make_vec_env(env_id, goal, baseline, alpha, device,
                             num_envs=1, subproc=False, k_sigmoid=k_sigmoid)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=checkpoints_dir,
        log_path=checkpoints_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoints_dir,
        name_prefix=f"{env_id}_{algo}_alpha{alpha}"
    )
    epi_cb = EpisodeStatsCallback(save_dir)

    model.learn(total_timesteps=int(steps), callback=[eval_cb, ckpt_cb, epi_cb])

    # Final model
    model.save(model_path)
    print(f"[train] saved final -> {model_path}")
    print(f"[train] best/periodic checkpoints in -> {checkpoints_dir}")

    # Optional auto-plot of returns
    if auto_plot:
        _auto_plot_returns(save_dir, exp_name)


def evaluate(
    env_id: str,
    model_path: str,
    episodes: int = 10,
    alpha: float = 0.7,
    device_str: Optional[str] = None,
    goal: Optional[str] = None,
    baseline: Optional[str] = None,
    k_sigmoid: float = 5.0,):
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if goal is None or len(goal.strip()) == 0:
        if env_id not in DEFAULT_PROMPTS:
            raise ValueError(f"No default prompts for env '{env_id}'. Provide --goal and optionally --baseline.")
        goal = DEFAULT_PROMPTS[env_id]["goal"]
        baseline = DEFAULT_PROMPTS[env_id]["baseline"] if baseline is None else baseline

    env = make_env(env_id, goal, baseline, alpha, device, k_sigmoid=k_sigmoid)
    _ = DummyVecEnv([lambda: env])  # shape-consistency only

    # Infer algo from filename (heuristic)
    if "_PPO_" in os.path.basename(model_path):
        algo = "PPO"
        model = PPO.load(model_path, device=device)
    elif "_DQN_" in os.path.basename(model_path):
        algo = "DQN"
        model = DQN.load(model_path, device=device)
    else:
        try:
            model = PPO.load(model_path, device=device)
            algo = "PPO"
        except Exception:
            model = DQN.load(model_path, device=device)
            algo = "DQN"

    print(f"[eval] env={env_id} algo={algo} model={model_path} episodes={episodes}")
    returns, clip_means = [], []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        clip_sum = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += r
            clip_sum += info.get("r_clip", 0.0)
            steps += 1
        returns.append(ep_ret)
        clip_means.append(clip_sum / max(1, steps))
        print(f"  episode {ep+1:02d}: return={ep_ret:.3f}  mean_clip={clip_means[-1]:.3f}")

    print(f"[eval] mean return over {episodes} eps: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"[eval] mean CLIP reward: {np.mean(clip_means):.3f} ± {np.std(clip_means):.3f}")




###############################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MountainCar-v0",
                        help="Gymnasium env id (e.g., CartPole-v1, MountainCar-v0)")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN"])
    parser.add_argument("--alpha", type=float, default=0.7, help="Blend weight for CLIP reward [0,1]")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0, etc.")
    parser.add_argument("--goal", type=str, default="", help="Override goal prompt")
    parser.add_argument("--baseline", type=str, default="", help="Override baseline prompt")
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (increase for GPU utilization)")
    parser.add_argument("--no-subproc", action="store_true", help="Use DummyVecEnv instead of SubprocVecEnv")
    parser.add_argument("--auto-plot", action="store_true", help="Auto-save training return plot at end")
    parser.add_argument("--k", type=float, default=5.0, help="CLIP sigmoid sharpness (higher = steeper)")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--model-path", type=str, default="", help="Path to saved model for --eval")

    args, _ = parser.parse_known_args()

    if args.eval:
        if not args.model_path:
            raise SystemExit("--eval requires --model-path")
        evaluate(env_id=args.env_id,
            model_path=args.model_path,
            episodes=10,
            alpha=args.alpha,
            device_str=args.device,
            goal=args.goal,
            baseline=args.baseline if len(args.baseline) > 0 else None,
            k_sigmoid=args.k)
    else:
        train(env_id=args.env_id,
            algo=args.algo,
            alpha=args.alpha,
            steps=args.steps,
            seed=args.seed,
            device_str=args.device,
            goal=args.goal,
            baseline=args.baseline if len(args.baseline) > 0 else None,
            save_dir=args.save_dir,
            num_envs=args.num_envs,
            subproc_env=not args.no_subproc,
            auto_plot=args.auto_plot,
            k_sigmoid=args.k)



###############################################################################################################################
if __name__ == "__main__":
    main()
