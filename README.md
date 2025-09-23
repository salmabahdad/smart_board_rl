# 🛫 Smart Airplane Boarding RL

   ### A reinforcement learning project simulating airplane boarding management using a custom Gymnasium environment and a maskable PPO agent from Stable Baselines3. This project focuses on optimizing passenger boarding, reducing waiting time in the aisle, and improving the boarding efficiency.
---
## 🚀 Features
#### Custom Gymnasium Environment:
 - AirplaneEnv simulates passengers boarding an airplane.
 - Passengers have different states: walking to seat, waiting in aisle, stowing suitcase, or seated and ready.
 - Supports terminal rendering for step-by-step visualization.
 - Action masking for valid boarding actions.

### Reinforcement Learning Agent:
- Uses MaskablePPO from SB3-contrib to learn optimal boarding strategies.
- Vectorized environment support for faster training (SubprocVecEnv).
- Callbacks for evaluation and saving best models during training.

### Reward Function:
- Encourages moving passengers forward and penalizes waiting in the aisle.
- Provides a numerical metric for boarding efficiency.
