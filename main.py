from Agent import QLearningAgent
from GameEnvironemnt import GameEnv

env = GameEnv()
agent = QLearningAgent()

try:
    agent.load_q_table()
    print("Loaded saved Q-table")
except FileNotFoundError:
    print("No saved Q-table, starting fresh...")

num_episodes = 1000000
wins = 0

for episode in range(num_episodes):
    state = env.reset()
    action = agent.choose_action(state)
    observation, reward, info = env.step(action)

    if reward > 0:
        wins += 1

    agent.learn(state, action, reward)

agent.save_q_table()
print(f"Win rate: {wins/num_episodes}")

