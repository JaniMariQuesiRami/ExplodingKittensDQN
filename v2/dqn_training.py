# dqn_training.py
# DQN para Exploding Kittens usando el entorno de exploding_env.py

import numpy as np
import random
from collections import deque
import csv
from datetime import datetime
import os

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
    num_episodes=1500,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=1000,
    target_update_interval=50,
    use_double_dqn=True,
):
    env = ExplodingKittensEnv()
    state_dim = len(env._get_obs())
    action_dim = 9  # ← 0-8: Draw, Skip, Attack, Defuse(x3), SeeFuture, DrawBottom, Shuffle

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=100_000)

    epsilon = epsilon_start

    episode_rewards = []
    episode_wins = []
    
    # Early Stopping variables
    best_win_rate = 0.0
    best_model_state = None
    best_episode = 0
    patience = 10
    no_improvement_count = 0
    min_episodes_before_stopping = 800
    min_winrate_for_stopping = 0.85

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
                    if use_double_dqn:
                        # Double DQN: usar q_net para seleccionar acción, target_net para evaluar
                        next_actions = q_net(next_states_t).argmax(1, keepdim=True)
                        next_q_values = target_net(next_states_t).gather(1, next_actions)
                    else:
                        # DQN clásico
                        next_q_values = target_net(next_states_t).max(1, keepdim=True)[0]
                    target = rewards_t + gamma * (1 - dones_t) * next_q_values

                loss = nn.functional.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()

        # Linear epsilon decay (como V1)
        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epsilon_decay_episodes
            epsilon = max(epsilon_end, epsilon)

        if (ep + 1) % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(total_reward)
        episode_wins.append(win_flag)

        if (ep + 1) % 50 == 0:
            win_rate = np.mean(episode_wins[-50:])
            r_mean = np.mean(episode_rewards[-50:])
            print(f"Ep {ep+1}/{num_episodes} | R_media_50={r_mean:.3f} | WinRate_50={win_rate:.3f} | eps={epsilon:.3f}")
            
            # Early Stopping: solo si WR > 80% y después de min_episodes
            if (ep + 1) >= min_episodes_before_stopping and win_rate >= min_winrate_for_stopping:
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_episode = ep + 1
                    best_model_state = q_net.state_dict().copy()
                    no_improvement_count = 0
                    print(f"  ✅ Nuevo mejor modelo! WinRate: {best_win_rate:.3f}")
                    # Guardar checkpoint del mejor modelo
                    os.makedirs('output', exist_ok=True)
                    torch.save(best_model_state, 'output/best_model_checkpoint.pth')
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print(f"\n🛑 Early Stopping activado!")
                        print(f"   Mejor WinRate: {best_win_rate:.3f} en episodio {best_episode}")
                        print(f"   Sin mejora durante {patience * 50} episodios")
                        # Restaurar mejor modelo
                        if best_model_state is not None:
                            q_net.load_state_dict(best_model_state)
                            print(f"   Modelo restaurado al episodio {best_episode}")
                        break
            elif (ep + 1) >= min_episodes_before_stopping:
                # Guardar mejor modelo incluso si no alcanza 80%, pero no hacer early stopping
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_episode = ep + 1
                    best_model_state = q_net.state_dict().copy()
                    print(f"  ✅ Nuevo mejor modelo! WinRate: {best_win_rate:.3f} (early stopping inactivo hasta 80%)")
                    os.makedirs('output', exist_ok=True)
                    torch.save(best_model_state, 'output/best_model_checkpoint.pth')

    # Si no hubo early stopping, usar el mejor modelo encontrado
    if best_model_state is not None and no_improvement_count < patience:
        print(f"\n📊 Entrenamiento completo. Restaurando mejor modelo (ep {best_episode}, WR: {best_win_rate:.3f})")
        q_net.load_state_dict(best_model_state)
    
    print("Entrenamiento terminado.")
    return q_net, episode_rewards, episode_wins


def evaluate_agent(q_net, num_episodes=500):
    """Evalúa el agente y retorna estadísticas + log detallado de cada juego"""
    env = ExplodingKittensEnv()
    wins = 0
    rewards = []
    game_logs = []  # Lista de logs detallados

    for game_idx in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        turn_count = 0
        actions_attempted = []
        actions_executed = []

        while not done:
            valid_actions = valid_actions_from_state(state)
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(s).cpu().numpy().flatten()
            action = max(valid_actions, key=lambda a: q_values[a])

            before_hand = env.hands[0].copy() if hasattr(env, 'hands') else {}

            state, reward, done, info = env.step(int(action))
            total_reward += reward
            turn_count += 1
            actions_attempted.append(int(action))

            # Detect if action was actually executed by checking hand changes
            executed = False
            if action == 0:
                executed = True
            elif action == 1 and before_hand.get('Skip', 0) > env.hands[0].get('Skip', 0):
                executed = True
            elif action == 2 and before_hand.get('Attack', 0) > env.hands[0].get('Attack', 0):
                executed = True
            elif action == 6 and before_hand.get('SeeFuture', 0) > env.hands[0].get('SeeFuture', 0):
                executed = True
            elif action == 7 and before_hand.get('DrawBottom', 0) > env.hands[0].get('DrawBottom', 0):
                executed = True
            elif action == 8 and before_hand.get('Shuffle', 0) > env.hands[0].get('Shuffle', 0):
                executed = True

            if executed:
                actions_executed.append(int(action))

            if done and info.get('winner') == 0:
                wins += 1

        rewards.append(total_reward)
        
        game_log = {
            'game_id': game_idx + 1,
            'turns': turn_count,
            'total_reward': total_reward,
            'won': 1 if (done and info.get('winner') == 0) else 0,
            'actions_attempted_sequence': ','.join(map(str, actions_attempted)),
            'actions_executed_sequence': ','.join(map(str, actions_executed)),
        }
        game_logs.append(game_log)

    win_rate = wins / num_episodes
    return win_rate, rewards, game_logs


def evaluate_random_agent(num_episodes=500):
    """Evalúa agente random y retorna estadísticas + log detallado.

    Similar a evaluate_agent, registra acciones intentadas y ejecutadas.
    """
    env = ExplodingKittensEnv()
    wins = 0
    rewards = []
    game_logs = []

    for game_idx in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        turn_count = 0
        actions_attempted = []
        actions_executed = []

        while not done:
            valid_actions = valid_actions_from_state(state)
            action = int(random.choice(valid_actions))

            before_hand = env.hands[1].copy() if hasattr(env, 'hands') else {}

            state, reward, done, info = env.step(action)
            total_reward += reward
            turn_count += 1
            actions_attempted.append(action)

            # Detect execution for random agent
            # Note: random agent is player 1 in _opponent_turn, but we keep symmetrical logging
            # For simplicity mark draw as executed; for other actions try to detect hand change
            executed = False
            if action == 0:
                executed = True
            elif action == 1 and before_hand.get('Skip', 0) > env.hands[1].get('Skip', 0):
                executed = True
            elif action == 2 and before_hand.get('Attack', 0) > env.hands[1].get('Attack', 0):
                executed = True
            elif action == 6 and before_hand.get('SeeFuture', 0) > env.hands[1].get('SeeFuture', 0):
                executed = True
            elif action == 7 and before_hand.get('DrawBottom', 0) > env.hands[1].get('DrawBottom', 0):
                executed = True
            elif action == 8 and before_hand.get('Shuffle', 0) > env.hands[1].get('Shuffle', 0):
                executed = True

            if executed:
                actions_executed.append(action)

            if done and info.get('winner') == 0:
                wins += 1

        rewards.append(total_reward)
        
        game_log = {
            'game_id': game_idx + 1,
            'turns': turn_count,
            'total_reward': total_reward,
            'won': 1 if (done and info.get('winner') == 0) else 0,
            'actions_attempted_sequence': ','.join(map(str, actions_attempted)),
            'actions_executed_sequence': ','.join(map(str, actions_executed)),
        }
        game_logs.append(game_log)

    win_rate = wins / num_episodes
    return win_rate, rewards, game_logs


def save_validation_logs_to_csv(game_logs, filename='validation_log.csv'):
    """Guarda logs de validación en CSV con columnas: game_id, turns, reward, won, actions"""
    os.makedirs('output', exist_ok=True)
    filepath = os.path.join('output', filename)
    with open(filepath, 'w', newline='') as f:
        # Soportar tanto actions_attempted_sequence como actions_executed_sequence
        fieldnames = ['game_id', 'turns', 'total_reward', 'won', 'actions_attempted_sequence', 'actions_executed_sequence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(game_logs)
    print(f"📝 Log guardado en: {filepath}")


def print_validation_summary(game_logs, agent_name="DQN"):
    """Imprime tabla resumen de estadísticas de validación"""
    turns_list = [g['turns'] for g in game_logs]
    rewards_list = [g['total_reward'] for g in game_logs]
    wins = sum(g['won'] for g in game_logs)
    
    # Análisis de acciones (usar executed si existe, sino attempted)
    action_counts = {i: 0 for i in range(9)}
    for game in game_logs:
        seq_key = 'actions_executed_sequence' if 'actions_executed_sequence' in game and game['actions_executed_sequence'] else 'actions_attempted_sequence'
        seq = game.get(seq_key, '')
        if seq:
            actions = list(map(int, seq.split(',')))
        else:
            actions = []
        for a in actions:
            action_counts[a] += 1
    
    total_actions = sum(action_counts.values())
    action_names = ["Draw", "Skip", "Attack", "Defuse1", "Defuse2", "Defuse3", "SeeFuture", "DrawBottom", "Shuffle"]
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE VALIDACIÓN - {agent_name}")
    print(f"{'='*60}")
    print(f"Juegos totales: {len(game_logs)}")
    print(f"Win Rate: {wins/len(game_logs)*100:.2f}% ({wins}/{len(game_logs)})")
    print(f"Turnos promedio: {np.mean(turns_list):.2f} ± {np.std(turns_list):.2f}")
    print(f"Turnos [min, max]: [{min(turns_list)}, {max(turns_list)}]")
    print(f"Reward promedio: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"\n{'Acción':<15} {'Total':<10} {'%':<10}")
    print(f"{'-'*35}")
    for i in range(9):
        pct = (action_counts[i] / total_actions * 100) if total_actions > 0 else 0
        print(f"{action_names[i]:<15} {action_counts[i]:<10} {pct:>6.2f}%")
    print(f"{'='*60}\n")


def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')


if __name__ == "__main__":
    # Entrenar
    num_episodes = 2000
    q_net, episode_rewards, episode_wins = train_dqn(num_episodes=num_episodes)
    
    # Gráficas de entrenamiento
    window = 50
    ma_rewards = moving_average(episode_rewards, window)
    ma_wins = moving_average(episode_wins, window)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ma_rewards)
    plt.title("ENTRENAMIENTO - Recompensa media (ventana=50)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")

    plt.subplot(1, 2, 2)
    plt.plot(ma_wins)
    plt.axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label='Target 85%')
    plt.axhline(y=max(ma_wins), color='r', linestyle='--', alpha=0.5, label=f'Pico {max(ma_wins):.1%}')
    plt.title("ENTRENAMIENTO - Win rate (ventana=50)")
    plt.xlabel("Episodio")
    plt.ylabel("Proporcion victorias")
    plt.legend()
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/training_curves_v2.png', dpi=150)
    print("📊 Gráficas de entrenamiento guardadas en output/training_curves_v2.png")
    plt.show()

    # ================================
    # VALIDACIÓN CON LOGGING DETALLADO
    # ================================
    print("\n" + "="*60)
    print("🧪 FASE DE VALIDACIÓN")
    print("="*60)
    
    eval_episodes = 400
    
    # Evaluar DQN
    print(f"\n🤖 Evaluando DQN Agent ({eval_episodes} juegos)...")
    win_rate_dqn, rewards_dqn, game_logs_dqn = evaluate_agent(q_net, num_episodes=eval_episodes)
    
    # Guardar CSV y mostrar resumen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"validation_dqn_{timestamp}.csv"
    save_validation_logs_to_csv(game_logs_dqn, filename=csv_filename)
    print_validation_summary(game_logs_dqn, agent_name="DQN Agent")
    
    # Evaluar Random
    print(f"\n🎲 Evaluando Random Agent ({eval_episodes} juegos)...")
    win_rate_rand, rewards_rand, game_logs_rand = evaluate_random_agent(num_episodes=eval_episodes)
    
    csv_filename_rand = f"validation_random_{timestamp}.csv"
    save_validation_logs_to_csv(game_logs_rand, filename=csv_filename_rand)
    print_validation_summary(game_logs_rand, agent_name="Random Agent")
    
    # ================================
    # GRÁFICAS DE VALIDACIÓN
    # ================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Comparación de Win Rates
    axes[0, 0].bar(['DQN', 'Random'], [win_rate_dqn, win_rate_rand], color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_ylabel('Win Rate')
    axes[0, 0].set_title('VALIDACION - Win Rate Comparison')
    axes[0, 0].set_ylim([0, 1])
    for i, (agent, wr) in enumerate([('DQN', win_rate_dqn), ('Random', win_rate_rand)]):
        axes[0, 0].text(i, wr + 0.02, f'{wr:.1%}', ha='center', fontweight='bold')
    
    # 2. Distribución de Turnos
    axes[0, 1].hist([g['turns'] for g in game_logs_dqn], bins=20, alpha=0.7, label='DQN', color='#3498db')
    axes[0, 1].hist([g['turns'] for g in game_logs_rand], bins=20, alpha=0.7, label='Random', color='#e67e22')
    axes[0, 1].set_xlabel('Turnos por juego')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('VALIDACION - Distribucion de Turnos')
    axes[0, 1].legend()
    
    # 3. Distribución de Rewards
    axes[1, 0].hist([g['total_reward'] for g in game_logs_dqn], bins=20, alpha=0.7, label='DQN', color='#9b59b6')
    axes[1, 0].hist([g['total_reward'] for g in game_logs_rand], bins=20, alpha=0.7, label='Random', color='#34495e')
    axes[1, 0].set_xlabel('Total Reward')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('VALIDACION - Distribucion de Rewards')
    axes[1, 0].legend()
    
    # 4. Porcentaje de uso de acciones (DQN)
    action_counts_dqn = {i: 0 for i in range(9)}
    for game in game_logs_dqn:
        # Usar actions_executed_sequence si existe, sino actions_attempted_sequence
        seq = game.get('actions_executed_sequence', game.get('actions_attempted_sequence', ''))
        if seq:
            actions = list(map(int, seq.split(',')))
            for a in actions:
                action_counts_dqn[a] += 1
    
    total_actions_dqn = sum(action_counts_dqn.values())
    action_names = ["Draw", "Skip", "Attack", "Def1", "Def2", "Def3", "SeeFut", "DrwBot", "Shuff"]
    action_pcts = [(action_counts_dqn[i] / total_actions_dqn * 100) if total_actions_dqn > 0 else 0 for i in range(9)]
    
    bars = axes[1, 1].bar(action_names, action_pcts, color='#1abc9c')
    axes[1, 1].set_xlabel('Accion')
    axes[1, 1].set_ylabel('% de uso')
    axes[1, 1].set_title('VALIDACION - Uso de Acciones (DQN)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Agregar valores en las barras
    for bar, pct in zip(bars, action_pcts):
        height = bar.get_height()
        if height > 1:  # Solo mostrar si es > 1%
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                          f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    validation_plot_filename = f'validation_analysis_{timestamp}.png'
    plt.savefig(os.path.join('output', validation_plot_filename), dpi=150)
    print(f"\n📊 Gráficas de validación guardadas en output/{validation_plot_filename}")
    plt.show()

    # Guardar modelo final (ya tiene el mejor modelo cargado gracias a early stopping)
    os.makedirs('output', exist_ok=True)
    torch.save(q_net.state_dict(), "output/dqn_exploding_kittens_v2.pth")
    print("\n✅ Modelo final guardado en output/dqn_exploding_kittens_v2.pth")
    
    # También conservar el checkpoint del mejor modelo
    if os.path.exists('output/best_model_checkpoint.pth'):
        print("✅ Mejor modelo checkpoint disponible en output/best_model_checkpoint.pth")
    
    print("\n" + "="*60)
    print("🎉 ENTRENAMIENTO Y VALIDACIÓN COMPLETADOS")
    print("="*60)
