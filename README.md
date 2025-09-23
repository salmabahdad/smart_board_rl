# 🛫 Smart Airplane Boarding RL

   ### A reinforcement learning project simulating airplane boarding management using a custom Gymnasium environment and a maskable PPO agent from Stable Baselines3. This project focuses on optimizing passenger boarding, reducing waiting time in the aisle, and improving the boarding efficiency.
---
## 🚀 Features
### Custom Gymnasium Environment:
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

## 🛠 Installation
### 1. Clone the repository:
<pre> <code>
  git clone https://github.com/yourusername/smart-board.git
  cd smart-airplane-boarding
</code> </pre>

### 2. Create a Python environment (recommended: conda or venv):
<pre> <code>
  conda create -n smartboard python=3.11
  conda activate smartboard
</code> </pre>

### 3. Install dependencies:
<pre> <code>
   pip install requirements.txt
</code> </pre>

## 📝 Usage
### Run Environment in Terminal
<pre> <code>
   python smart_board_env.py
</code> </pre>
- Random actions will be executed, showing the boarding process, passenger states, and reward.

### Train RL Agent
<pre> <code>
   python agent.py
</code> </pre>
- Trains the MaskablePPO agent on the boarding environment.
- Logs are saved to logs/, models to models/.

### Test Trained Agent
<pre> <code>
  from agent import test
  test("best_model_name", render=True)
</code> </pre>
- Loads a saved model and runs it in human-readable mode.

## 🧠 How It Works
### 1.Environment Setup:
 - The airplane has rows of seats and a lobby where passengers wait.
 - Passengers are represented by a Passenger class with status and seat info.

### 2.Action & Reward:
 - Each action corresponds to moving a passenger from a specific lobby row.
 - Reward = passengers walking toward seats − passengers waiting in the aisle.

### RL Agent:
 - Learns to choose the best lobby row to minimize aisle congestion.
 - Uses masking to ignore invalid actions.

## 🔍 Results
- The agent learns to minimize aisle waiting and efficiently board passengers.
- Can be visualized step-by-step in terminal mode or rendered graphically with render_mode='human'.
