# dqn_training.py - V1 (Simple version)
# DQN para Exploding Kittens usando el entorno de exploding_env.py

import numpy as np
import random
from collections import deque

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from exploding_env import ExplodingKittensEnv, valid_actions_from_state, action_name_from_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Usando dispositivo:", device)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(q_net, state, epsilon):
    valid_actions = valid_actions_from_state(state)
    if random.random() < epsilon:
        return random.choice(valid_actions)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(s).cpu().numpy().flatten()
    best_a = max(valid_actions, key=lambda a: q_values[a])
    return int(best_a)


def train_dqn(
    num_episodes=800,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=600,
    target_update_interval=50,
):
    env = ExplodingKittensEnv()
    state_dim = len(env._get_obs())
    action_dim = 6

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=100_000)

    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_episodes)

    episode_rewards = []
    episode_wins = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        win_flag = 0

        while not done:
            action = select_action(q_net, state, epsilon)
            next_state, reward, done, info = env.step(action)

            replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done and info.get('winner') == 0:
                win_flag = 1

            if len(replay) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay.sample(batch_size)

                states_t = torch.tensor(states_b, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions_b, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.tensor(next_states_b, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = q_net(states_t).gather(1, actions_t)

                with torch.no_grad():
                    next_q_values = target_net(next_states_t).max(1, keepdim=True)[0]
                    target = rewards_t + gamma * (1 - dones_t) * next_q_values

                loss = nn.functional.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_end:
            epsilon -= epsilon_decay
            epsilon = max(epsilon, epsilon_end)

        if (ep + 1) % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(total_reward)
        episode_wins.append(win_flag)

        if (ep + 1) % 50 == 0:
            win_rate = np.mean(episode_wins[-50:])
            print(f"Ep {ep+1}/{num_episodes} | R_media_50={np.mean(episode_rewards[-50:]):.3f} | WinRate_50={win_rate:.3f} | eps={epsilon:.3f}")

    return q_net, episode_rewards, episode_wins


def evaluate_agent(q_net, num_episodes=500):
    env = ExplodingKittensEnv()
    wins = 0
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = valid_actions_from_state(state)
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(s).cpu().numpy().flatten()
            action = max(valid_actions, key=lambda a: q_values[a])

            state, reward, done, info = env.step(int(action))
            total_reward += reward

            if done and info.get('winner') == 0:
                wins += 1

        rewards.append(total_reward)

    win_rate = wins / num_episodes
    return win_rate, rewards


def evaluate_random_agent(num_episodes=500):
    env = ExplodingKittensEnv()
    wins = 0
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = valid_actions_from_state(state)
            action = random.choice(valid_actions)
            state, reward, done, info = env.step(action)
            total_reward += reward

            if done and info.get('winner') == 0:
                wins += 1

        rewards.append(total_reward)

    win_rate = wins / num_episodes
    return win_rate, rewards


def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')


if __name__ == "__main__":
    # Entrenar
    num_episodes = 2000
    q_net, episode_rewards, episode_wins = train_dqn(num_episodes=num_episodes)
    print("Entrenamiento terminado.")

    # Gráficas de entrenamiento
    window = 50
    ma_rewards = moving_average(episode_rewards, window)
    ma_wins = moving_average(episode_wins, window)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ma_rewards)
    plt.title("Recompensa media (ventana=50)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")

    plt.subplot(1, 2, 2)
    plt.plot(ma_wins)
    plt.title("Win rate aproximado (ventana=50)")
    plt.xlabel("Episodio")
    plt.ylabel("Proporción victorias")
    plt.tight_layout()
    plt.show()

    # Evaluación
    eval_episodes = 400
    win_rate_dqn, _ = evaluate_agent(q_net, num_episodes=eval_episodes)
    print(f"Win rate DQN vs heurístico ({eval_episodes} eps): {win_rate_dqn:.3f}")

    win_rate_rand, _ = evaluate_random_agent(num_episodes=eval_episodes)
    print(f"Win rate random vs heurístico ({eval_episodes} eps): {win_rate_rand:.3f}")

    # Guardar modelo entrenado
    torch.save(q_net.state_dict(), "dqn_exploding_kittens.pth")
    print("Modelo guardado en dqn_exploding_kittens.pth")
