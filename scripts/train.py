import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from PIL import Image
import clip
import torch.nn.functional as F

####################################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reward- CLIP based
class CLIPReward:
    def __init__(self, device='cuda', goal_prompt="pole vertically upright on top of the cart", baseline_prompt="“pole and cart"):
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
            text_emb = self.model.encode_text(tokenized)
            self.goal_emb = text_emb[0]
            self.goal_emb /= self.goal_emb.norm(dim=-1, keepdim=True)
            if self.baseline_prompt is not None:
                self.baseline_emb = text_emb[1]
                self.baseline_emb /= self.baseline_emb.norm(dim=-1, keepdim=True)
            else:
                self.baseline_embed = None

    def compute_reward(self, image: np.ndarray) -> float:
        pil_img = Image.fromarray(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_emb = self.model.encode_image(image_input)
            image_emb /= image_emb.norm(dim=-1, keepdim=True)
        sim_goal = F.cosine_similarity(image_emb, self.goal_emb.unsqueeze(0)).item()

        # Score calculation
        if self.baseline_emb is not None:
            sim_baseline_score = F.cosine_similarity(image_emb, self.baseline_emb.unsqueeze(0)).item()
            reward = sim_goal - sim_baseline_score
        else:
            reward = sim_goal
        norm_reward = (reward + 1.0) / 2.0  
        return norm_reward

    def compute_image_embedding(self, image: np.ndarray) -> torch.Tensor:
        pil_img = Image.fromarray(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb = self.model.encode_image(image_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        return img_emb


# Preference MOdel
class PreferenceModel(nn.Module):
    def __init__(self, input_dim=512):
        super(PreferenceModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.fc(x)


# Reward- Clip + Pref based and Environment Wrapper
class HybridRewardEnv(gym.Wrapper):
    def __init__(self, env, clip_reward: CLIPReward, pref_model=None, alpha=0.5):
        super().__init__(env)
        self.clip_reward = clip_reward
        self.pref_model = pref_model  

        # Weight for preference reward
        self.alpha = alpha  

        self.max_steps = 200         
        self.step_count = 0
        self.embedding_buffer = []    

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)  

    def step(self, action):
       
        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        image = self.env.render() 
        clip_r = self.clip_reward.compute_reward(image)
        img_embed = self.clip_reward.compute_image_embedding(image)
        self.embedding_buffer.append((img_embed.cpu().squeeze(0), clip_r))


        if self.pref_model is not None:
            with torch.no_grad():
                current_embed = img_embed.to(device).float()  #float32()
                pref_score = self.pref_model(current_embed)
                pref_r = torch.sigmoid(pref_score).item()
            
            # Hybrid reward function
            hybrid_r = self.alpha * pref_r + (1 - self.alpha) * clip_r
        else:
            hybrid_r = clip_r
        done = terminated or truncated or (self.step_count >= self.max_steps)
        return obs, hybrid_r, terminated, truncated, info


# Hybrid Reward Calculation Function
def compute_hybrid_reward(clip_r, pref_r, alpha=0.5):
    return alpha * pref_r + (1 - alpha) * clip_r


# Training loop with hybrid model
def train_model():
    os.makedirs("checkpoints", exist_ok=True)
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    clip_reward = CLIPReward(device=device, goal_prompt="pole vertically upright on top of the cart", baseline_prompt="“pole and cart")
    pref_model = PreferenceModel(input_dim=512).to(device)
    alpha = 0.7
    env = HybridRewardEnv(env, clip_reward, pref_model=pref_model, alpha=alpha)
    env = Monitor(env)  # For logging metrics

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/hybrid/", device=device)
    
    pref_optimizer = optim.Adam(pref_model.parameters(), lr=1e-3)
    bce_loss = nn.BCEWithLogitsLoss()
    min_buffer_size = 100
    batch_size = 16

    total_timesteps = 100_000
    timestep = 0
    step_time_total = 0.0
    steps_since_print = 0

    while timestep < total_timesteps:
        obs, info = env.reset()
        done = False
        while not done and timestep < total_timesteps:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, hybrid_r, terminated, truncated, info = env.step(action)
            t1 = time.time()
            step_time = t1 - t0
            step_time_total += step_time
            steps_since_print += 1

            timestep += 1
            done = terminated or truncated or (env.env.step_count >= env.env.max_steps)
            
            if steps_since_print >= 100:
                avg_time = step_time_total / 100.0
                print(f"Average time per timestep over last 100 steps: {avg_time:.4f} seconds")
                step_time_total = 0.0
                steps_since_print = 0
            
            #Preference model
            if len(env.env.embedding_buffer) >= min_buffer_size:
                indices = np.random.choice(len(env.env.embedding_buffer), size=batch_size * 2, replace=True)
                batch = [env.env.embedding_buffer[i] for i in indices]
                losses = []
                for i in range(0, len(batch), 2):
                    emb1, r1 = batch[i]
                    emb2, r2 = batch[i+1]
                    label = torch.tensor([[1.0]], device=device) if r1 > r2 else torch.tensor([[0.0]], device=device)
                    emb1 = emb1.unsqueeze(0).to(device).float()  
                    emb2 = emb2.unsqueeze(0).to(device).float() 
                    score1 = pref_model(emb1)
                    score2 = pref_model(emb2)
                    pred = torch.sigmoid(score1 - score2)
                    loss = bce_loss(pred, label)
                    losses.append(loss)
                if losses:
                    total_loss = torch.stack(losses).mean()
                    pref_optimizer.zero_grad()
                    total_loss.backward()
                    pref_optimizer.step()
            
            # Save checkpoint each at 10,000 timestep
            if timestep % 10000 == 0:
                checkpoint_filename = os.path.join("checkpoints", f"hybrid_model_{timestep}.zip")
                print(f"Saving checkpoint at timestep {timestep}: {checkpoint_filename}")
                model.save(checkpoint_filename)
        
        print(f"-----------Timestep : {timestep}------------")
    
    model.save("hybrid_model_cartpole_final.zip")
    torch.save(pref_model.state_dict(), "pref_model.pth")
    env.close()

# Train
if __name__ == "__main__":
    train_model()
