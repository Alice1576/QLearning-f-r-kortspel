import random
from Actions import Action
from collections import defaultdict
import pickle

class QLearningAgent:
    def __init__(self, epsilon = 0.0, alpha = 0.01):
        self.q_table = defaultdict(lambda: {action: 0.0 for action in Action})
        self.epsilon = epsilon
        self.alpha = alpha

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        else:
            actions = self.q_table[state]
            return max(actions, key=actions.get)

    def learn(self, state, action, reward):
        old_q = self.q_table[state][action]
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[state][action] = new_q

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: {action: 0.0 for action in Action}, data)
