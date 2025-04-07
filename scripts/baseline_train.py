import torch
import clip
from PIL import Image
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym


class CLIPReward:
    def __init__(self, device='cuda', goal_prompt="pole vertically upright on top of the cart", baseline_prompt=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.goal_prompt = goal_prompt
        self.baseline_prompt = baseline_prompt

        prompts = [self.goal_prompt]
        if self.baseline_prompt is not None:
            prompts.append(self.baseline_prompt)
        tokenized = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.encode_text(tokenized)
            self.goal_embed = text_embeds[0]
            self.goal_embed /= self.goal_embed.norm(dim=-1, keepdim=True)
            if self.baseline_prompt is not None:
                self.baseline_embed = text_embeds[1]
                self.baseline_embed /= self.baseline_embed.norm(dim=-1, keepdim=True)
            else:
                self.baseline_embed = None

    def compute_reward(self, image: np.ndarray) -> float:
        pil_img = Image.fromarray(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embed = self.model.encode_image(image_input)
            image_embed /= image_embed.norm(dim=-1, keepdim=True)
        sim_goal = (image_embed @ self.goal_embed.T).item()
        if self.baseline_embed is not None:
            sim_baseline = (image_embed @ self.baseline_embed.T).item()
            reward = sim_goal - sim_baseline
        else:
            reward = sim_goal
        norm_reward = (reward + 1.0) / 2.0  # Normalize to [0,1]
        return norm_reward

    def compute_embedding(self, image: np.ndarray) -> torch.Tensor:
        pil_img = Image.fromarray(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_embed = self.model.encode_image(image_input)
            img_embed /= img_embed.norm(dim=-1, keepdim=True)
        return img_embed




class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env, clip_reward):
        super().__init__(env)
        self.clip_reward = clip_reward
        self.max_steps = 200  # Adjust based on environment requirements
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Gymnasium step returns: (obs, reward, terminated, truncated, info)
        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        image = self.env.render()  # render_mode is set in gym.make()
        clip_r = self.clip_reward.compute_reward(image)
        done = terminated or truncated or (self.current_step >= self.max_steps)
        return obs, clip_r, terminated, truncated, info




def train_cartpole():
   
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    clip_reward = CLIPReward(device='cuda', goal_prompt="pole vertically upright on top of the cart")

    env = ClipRewardEnv(env, clip_reward)
    env = Monitor(env)

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/', name_prefix='dqn_cartpole_clip')
    model.learn(total_timesteps=100_000, callback=checkpoint_callback)

    # inal model
    model.save("dqn_baseline_model.zip")
    env.close()

# Final
if __name__ == "__main__":
    train_cartpole()
