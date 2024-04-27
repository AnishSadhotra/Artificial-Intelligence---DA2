import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)  # Exploration
        else:
            return np.argmax(self.q_table[state, :])  # Exploitation

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

def simulate_environment(agent, num_episodes, max_steps):
    for episode in range(num_episodes):
        state = np.random.randint(0, agent.num_states)  # Start from a random state
        for _ in range(max_steps):
            action = agent.choose_action(state)
            # Simulate taking action in the environment
            next_state, reward = simulate_step(state, action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state

def simulate_step(state, action):
    # Define transition dynamics and rewards for the grid world
    # For simplicity, assume deterministic transitions and fixed rewards
    if action == 0:  # Move up
        next_state = max(state - 3, 0)
    elif action == 1:  # Move down
        next_state = min(state + 3, 8)
    elif action == 2:  # Move left
        next_state = max(state - 1, 0)
    else:  # Move right
        next_state = min(state + 1, 8)
    
    if next_state == 8:  # Goal state
        reward = 1
    elif next_state in [3, 5, 7]:  # Obstacle states
        reward = -1
    else:
        reward = 0
    
    return next_state, reward

# Main
num_states = 9  # 3x3 grid world
num_actions = 4  # Up, down, left, right
agent = QLearningAgent(num_states, num_actions)
simulate_environment(agent, num_episodes=1000, max_steps=100)

# Extract learned policy
optimal_policy = np.argmax(agent.q_table, axis=1)
print("Learned Optimal Policy:")
print(optimal_policy.reshape((3, 3)))
