import os
import sys
from collections import deque
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

class QNetwork(nn.Module):
    """Нейронная сеть для аппроксимации Q-функции."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    """Буфер воспроизведения опыта для стабильного обучения DQN."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.bool_))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Агент DQN с целевой сетью и буфером воспроизведения."""
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256),
                 lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64,
                 target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Основная и целевая сети
        self.policy_net = QNetwork(state_dim, hidden_dims, action_dim)
        self.target_net = QNetwork(state_dim, hidden_dims, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, epsilon):
        """Выбор действия по epsilon-жадной стратегии."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Сохранение перехода в буфер воспроизведения."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """Выполнение одного шага обучения DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Выборка из буфера
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Конвертация в тензоры PyTorch
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.BoolTensor(dones).unsqueeze(1)

        # Вычисление целевых Q-значений с использованием целевой сети
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q_values = rewards_t + (self.gamma * next_q_values * ~dones_t)

        # Вычисление текущих Q-значений
        current_q_values = self.policy_net(states_t).gather(1, actions_t)

        # Вычисление и оптимизация функции потерь
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Обрезка градиентов для стабильности
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Периодическое обновление целевой сети
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

def linear_decay(initial_value, final_value, decay_steps, step):
    """Линейное затухание значения (например, для epsilon)."""
    if step >= decay_steps:
        return final_value
    fraction = step / decay_steps
    return initial_value + fraction * (final_value - initial_value)

def run_dqn_custom(
    total_steps=10000,
    hidden_dims=(256, 256),
    lr=1e-3,
    gamma=0.99,
    eps_start=0.4,
    eps_end=0.02,
    eps_decay_steps=2500,
    train_freq=4,
    replay_buffer_size=10000,
    batch_size=64,
    target_update_freq=100,
    num_seconds=1000,  # Уменьшено для быстрого теста
    use_gui=False,
    save_dir="outputs/big-intersection",
    csv_filename="dqn_custom_rewards.csv"
):
    """Основная функция для запуска пользовательского DQN."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, csv_filename)

    env = SumoEnvironment(
        net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
        route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
        single_agent=True,
        use_gui=use_gui,
        num_seconds=num_seconds,
        yellow_time=3,
        min_green=5,
        max_green=20,
    )

    state_dim = np.prod(env.observation_space.shape).item()
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        gamma=gamma,
        buffer_size=replay_buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )

    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["episode", "total_reward", "steps", "avg_loss"]).to_csv(csv_path, index=False)

    state, _ = env.reset()
    state = np.asarray(state, dtype=np.float32).flatten()
    episode_reward = 0.0
    episode_steps = 0
    episode_loss_sum = 0.0
    episode_loss_count = 0
    episode_num = 1

    print("Начало обучения DQN...")
    for step in range(1, total_steps + 1):

        epsilon = linear_decay(eps_start, eps_end, eps_decay_steps, step)

        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.asarray(next_state, dtype=np.float32).flatten()
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        episode_reward += reward
        episode_steps += 1


        if step % train_freq == 0:
            loss = agent.update()
            if loss is not None:
                episode_loss_sum += loss
                episode_loss_count += 1

        if done:
            avg_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0.0

            new_row = pd.DataFrame([{
                "episode": episode_num,
                "total_reward": round(episode_reward, 3),
                "steps": episode_steps,
                "avg_loss": round(avg_loss, 5)
            }])
            new_row.to_csv(csv_path, mode='a', header=False, index=False)

            print(f"Эпизод {episode_num}: награда = {episode_reward:.2f}, шагов = {episode_steps}, "
                  f"средняя потеря = {avg_loss:.4f}, epsilon = {epsilon:.3f}")

            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32).flatten()
            episode_reward = 0.0
            episode_steps = 0
            episode_loss_sum = 0.0
            episode_loss_count = 0
            episode_num += 1
        else:
            state = next_state

    env.close()
    print(f"Обучение завершено. Результаты сохранены в {csv_path}")

if __name__ == "__main__":
    run_dqn_custom(
        total_steps=5000,          
        num_seconds=1000,          
        use_gui=False,            
        replay_buffer_size=5000,
        batch_size=32,
        eps_decay_steps=2000,
        csv_filename="dqn_custom_test_results.csv"
    )