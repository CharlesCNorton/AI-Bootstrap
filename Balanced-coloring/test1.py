import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from numba import cuda
from collections import deque

# Define Q-learning parameters and actions
ACTIONS = ['increase_k', 'decrease_k', 'increase_lookahead', 'decrease_lookahead', 'no_change']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the neural network for Deep Q-Learning (DQN)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # Increased size for better GPU utilization
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize DQN
input_size = 3  # Discrepancy, Lookahead, and k
output_size = len(ACTIONS)
q_network = DQN(input_size, output_size).to(device)
target_network = DQN(input_size, output_size).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Replay buffer
replay_buffer = deque(maxlen=200000)  # Increased buffer size to store more experiences
batch_size = 512  # Larger batch size for improved GPU usage
gamma = 0.9
update_frequency = 50  # Increased frequency to utilize GPU more effectively

# Statistics
action_counter = {action: 0 for action in ACTIONS}
discrepancy_history = []
reward_history = []

# Epsilon-greedy action selection with adaptive epsilon
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 500000  # Faster decay to promote exploitation sooner

def choose_action(state, step):
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * step / epsilon_decay)
    if random.random() < epsilon:
        return random.choice(range(output_size))
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = q_network(state_tensor)
            return torch.argmax(q_values).item()

# Update the Q-network using a batch of experiences with prioritized replay
def learn_batch(batch):
    states, actions, rewards, next_states = zip(*batch)
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)

    # Predict Q-values
    q_values = q_network(states_tensor)
    q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_network(next_states_tensor)
        max_next_q_values = next_q_values.max(1)[0]

    target_q_values = rewards_tensor + (gamma * max_next_q_values)

    # Compute loss and optimize
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# CUDA kernel to parallelize heavy calculations
def parallel_lookahead_calculation(n, red_sum, blue_sum, lookahead_window):
    threads_per_block = 512
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    @cuda.jit
    def calculate_lookahead(n, red_sum, blue_sum, lookahead_window, results):
        idx = cuda.grid(1)
        if idx < n:
            temp_red_sum = red_sum
            temp_blue_sum = blue_sum
            for j in range(lookahead_window):
                current_value = idx + 2 + j  # Starting from 2 onwards
                temp_red_sum += 1 / current_value
                temp_blue_sum += 1 / current_value
                # Store the difference between the red and blue sums
                results[idx] = abs(temp_red_sum - temp_blue_sum)

    results = cuda.device_array(n, dtype=np.float32)
    calculate_lookahead[blocks_per_grid, threads_per_block](n, red_sum, blue_sum, lookahead_window, results)
    return results.copy_to_host()

# Main function for the hybrid coloring with DQN and Double Q-Learning
def hybrid_coloring_q_learning(n, initial_lookahead=3, initial_k=0.1):
    red_sum, blue_sum = 0.0, 0.0
    red_set, blue_set = [], []
    lookahead_window = initial_lookahead
    k = initial_k
    step = 0

    for i in range(2, n + 1):
        discrepancy = abs(red_sum - blue_sum)
        state = [discrepancy, lookahead_window, k]

        # Choose action for adjustment using DQN with adaptive epsilon
        action_index = choose_action(state, step)
        action = ACTIONS[action_index]
        action_counter[action] += 1  # Update action counter

        # Execute action
        if action == 'increase_k':
            k = min(k + 0.01, 0.5)  # Smaller, more incremental adjustments
        elif action == 'decrease_k':
            k = max(k - 0.01, 0.05)
        elif action == 'increase_lookahead':
            lookahead_window = min(lookahead_window + 1, 5)  # Cap lookahead to 5
        elif action == 'decrease_lookahead':
            lookahead_window = max(lookahead_window - 1, 1)

        # Calculate adaptive probability adjustment
        if red_sum + blue_sum == 0:
            p_adjusted = 0.5
        else:
            normalized_difference = (blue_sum - red_sum) / (red_sum + blue_sum)
            p_adjusted = 0.5 + k * normalized_difference
        p_adjusted = max(0, min(1, p_adjusted))

        # Lookahead decision with reduced complexity and parallel processing
        best_difference = float('inf')
        best_choice = 'red'

        if lookahead_window > 1:
            lookahead_results = parallel_lookahead_calculation(lookahead_window, red_sum, blue_sum, lookahead_window)
            for idx, diff in enumerate(lookahead_results):
                if diff < best_difference:
                    best_difference = diff
                    best_choice = 'red' if idx % 2 == 0 else 'blue'
        else:
            # If lookahead is minimal, just use immediate evaluation
            for choice in ['red', 'blue']:
                temp_red_sum, temp_blue_sum = red_sum, blue_sum
                current_value = i
                if choice == 'red':
                    temp_red_sum += 1 / current_value
                else:
                    temp_blue_sum += 1 / current_value
                current_difference = abs(temp_red_sum - temp_blue_sum)
                if current_difference < best_difference:
                    best_difference = current_difference
                    best_choice = choice

        # Apply probabilistic choice
        if best_choice == 'red':
            if random.random() < p_adjusted:
                blue_set.append(i)
                blue_sum += 1 / i
            else:
                red_set.append(i)
                red_sum += 1 / i
        else:
            if random.random() < p_adjusted:
                red_set.append(i)
                red_sum += 1 / i
            else:
                blue_set.append(i)
                blue_sum += 1 / i

        # Store experience in replay buffer
        next_state = [abs(red_sum - blue_sum), lookahead_window, k]
        reward = -discrepancy
        reward_history.append(reward)
        discrepancy_history.append(discrepancy)
        replay_buffer.append((state, action_index, reward, next_state))

        # Update network every few steps
        if len(replay_buffer) >= batch_size and step % update_frequency == 0:
            batch = random.sample(replay_buffer, batch_size)
            learn_batch(batch)

        # Update target network
        if step % (update_frequency * 10) == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Print progress periodically
        if i % (n // 10) == 0:
            print(f"Step {i}/{n} - Current Discrepancy: {discrepancy}, k: {k}, Lookahead: {lookahead_window}")

        step += 1

    return red_sum, blue_sum, red_set, blue_set

# Main function to conduct the full GPU-based test with DQN enhancements
def main():
    test_values_n = [100000000]
    results = {}

    for n in test_values_n:
        start_time = time.time()
        red_sum, blue_sum, red_set, blue_set = hybrid_coloring_q_learning(n)
        execution_time = time.time() - start_time
        discrepancy = abs(red_sum - blue_sum)
        results[f"n={n}, Q-learning (GPU-based)"] = {
            "discrepancy": discrepancy,
            "execution_time": execution_time
        }
        print(f"Completed n={n}, Q-learning (GPU-based): Discrepancy = {discrepancy}, Execution Time = {execution_time:.2f} seconds")

    # Print final statistics
    print("\nAction Distribution:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    # Plotting results
    plt.figure(figsize=(14, 10))
    discrepancies = [results[key]['discrepancy'] for key in results]
    labels = list(results.keys())

    plt.plot(labels, discrepancies, marker='o', linestyle='-', label=f"Q-learning-enhanced Hybrid Coloring (GPU-based)")
    plt.xlabel('Configurations')
    plt.ylabel('|S_red(n) - S_blue(n)| (Discrepancy)')
    plt.title('Discrepancy Comparison with Q-learning Enhanced Hybrid Coloring (GPU-based)')
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
