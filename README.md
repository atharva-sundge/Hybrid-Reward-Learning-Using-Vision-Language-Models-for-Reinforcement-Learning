# Hybrid Reward Learning Using Vision-Language Models for Reinforcement Learning

**Name:** Atharva Vijay Sundge  
**Affiliation:** Rochester Institute of Technology  
**Contact:** [avs7774@rit.edu](mailto:avs7774@rit.edu)

---

## Project Directory Structure

```
Hybrid-Reward-Learning-Using-Vision-Language-Models/
├── data/  
│   └── (Contains a text file for requirements; no dataset available)
├── scripts/  
│   └── (Contains code for training baseline and hybrid models)
├── models/  
│   └── (Saved models for both hybrid and baseline approaches)
├── notebooks/  
│   └── (Evaluation notebooks with plots and analysis)
├── results/  
│   └── (Plots: cumulative reward vs. episode, pole angle vs. timestep, pole angle vs. reward)
├── figures/  
│   └── (Additional figures and visualizations)
└── README.md  
    (Project overview)
```

---

## Installation

Run the following commands in your terminal or Colab cell:

```bash
!pip install --upgrade pip
!pip install gymnasium
!pip install stable-baselines3
!pip install numpy
!pip install matplotlib
!pip install pillow
!pip install git+https://github.com/openai/CLIP.git

# For PyTorch with CUDA 11.8 support:
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
```

*(If using Conda, create a new environment and install these packages accordingly.)*

---

## Instructions for Running the Project

### Baseline Model

**Train Baseline:**
- Run the `baseline_train.py` file in the `scripts/` folder.

**Evaluation:**
- Run the `evaluation.ipynb` notebook (or similar) where the `CLIPReward` function is imported and the environment is created using the CLIP-based reward.
- Ensure any hybrid reward code is commented out to test the pure CLIP setup.

### Hybrid Model

**Train Hybrid Model:**
- Run the `train.py` file in the `scripts/` folder.
  - This script uses a custom `HybridRewardEnv` that combines the CLIP-based reward with a preference-based reward (via a Bradley–Terry model).
  - Model checkpoints are saved every 10,000 timesteps in the `models/` folder.

**Evaluation:**
- Use the evaluation notebook (e.g., `evaluation.ipynb` in the `notebooks/` folder) to evaluate the trained hybrid model.
  - The notebook creates the environment using `HybridRewardEnv` (comment out any pure CLIP parts).
  - It then generates plots for:
    - Cumulative Reward vs. Episode
    - Pole Angle vs. Timestep
    - Pole Angle vs. Reward

The desired outcome is for the reward to peak at a 0° pole angle (indicating balance) and decrease smoothly (ideally in a Gaussian-like fashion) as the pole deviates from vertical.

---

## Project Details

### Hybrid Reward Function
The reward for each step is computed as:
\[
R_{\text{Hybrid}}(s) = \alpha \cdot R_{\text{Pref}}(s) + (1 - \alpha) \cdot R_{\text{VLM}}(s)
\]
where \(R_{\text{VLM}}(s)\) is obtained using CLIP (via cosine similarity between the environment image and a goal prompt) and \(R_{\text{Pref}}(s)\) is learned from synthetic pairwise comparisons via a preference model.

### Preference Model
A simple one-layer neural network is trained on pairwise differences in CLIP reward to learn a more stable reward function that corrects for CLIP’s noise and misalignment.

### Evaluation Metrics
- **Cumulative Reward per Episode:** Indicates overall performance.
- **Pole Angle Distribution:** Verifies that the agent maintains a balanced (0°) pole.
- **Pole Angle vs. Reward Scatter Plot:** Should show high rewards around 0° and decreasing rewards as the pole deviates.

  
### Results:
#### Baseline Model Results
- Cumulative Reward per Episode:
- ![episode_rewards_baseline](https://github.com/user-attachments/assets/2d1f7c3b-70eb-492b-8123-3fd3e0377390)
 This plot shows the cumulative rewards obtained in each episodes when training the baseline (CLIP-based reward) model.
 
 The baseline model's cumulative rewards shows lower performance and higher variance compared to the hybrid approach.


- Pole Angle Distribution:
- ![pole_angles_baseline](https://github.com/user-attachments/assets/ea8b26c6-730b-4101-bd60-380a0c2ff1bf)

 This plot illustrates the distribution of the pole angle (converted to degrees) over timesteps for the baseline model.Ideally, most angles should cluster around 0° if the pole is balanced, but the baseline may show a wider spread which indicates instability in model training.


- Pole Angle vs. Reward plot:
- ![pole_angle_vs_reward_baseline](https://github.com/user-attachments/assets/bc9f77db-1704-4623-89ca-de40f1f06863)
 A scatter plot showing the relationship between pole angle and the reward at each step for the baseline model. For a well-performing policy, high rewards should be concentrated near 0° (balanced pole) and decrease as the pole deviates. In the baseline, the relationship is less distinct.



#### Hybrid Model Results
- Cumulative Reward per Episode:
- ![episode_rewards](https://github.com/user-attachments/assets/df7a5f61-7275-4fe8-9810-d129363036d1)


 This plot shows the cumulative reward per episode when training with the hybrid reward (CLIP + preference-based reward). The hybrid model is expected to achieve higher and more stable rewards as the additional preference feedback refines the raw CLIP reward.


- Pole Angle Distribution:
- ![pole_angles](https://github.com/user-attachments/assets/d0eb159b-b635-4787-9efd-17713f3daa24)
 This plot presents the distribution of the pole angle (in degrees) across timesteps for the hybrid model. A tighter clustering around 0° indicates that the agent is better at maintaining balance.


- Pole Angle vs. Reward plot:
- ![pole_angle_vs_reward](https://github.com/user-attachments/assets/4036b167-7e19-49a1-bbeb-a23e80afc2d4)
 A scatter plot showing the correlation between the pole angle and the reward for the hybrid model. Ideally, this plot should show high rewards for angles near 0° with a smooth, Gaussian-like decay as the angle increases, demonstrating improved reward alignment.



---

## Model Card

**Trained Models:**
- `models/dqn_hybrid_cartpole_final.zip`: Final DQN agent checkpoint.
- `models/pref_model.pth`: Preference model checkpoint.

Instructions for loading the models for inference are provided in the training and evaluation scripts.

---

## References

1. J. Rocamonde, V. Montesinos, E. Nava, E. Perez, and D. Lindner, "Vision-language models are zero-shot reward models for reinforcement learning," *arXiv preprint arXiv:2310.12921*, 2023.
2. W. Chen, O. Mees, A. Kumar, and S. Levine, "Vision-language models provide promptable representations for reinforcement learning," *arXiv preprint arXiv:2402.02651*, 2024.
3. Y. Wang, Z. Sun, J. Zhang, Z. Xian, E. Biyik, D. Held, and Z. Erickson, "RL-VLM-F: Reinforcement learning from vision language foundation model feedback," *arXiv preprint arXiv:2402.03681*, 2024.

---

## License

This project is licensed under the MIT License. 

Thank you!
