import sys
import os

# Aggiungi la directory corrente al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Importa i moduli locali: WarehouseEnv e ParallelEnvToGymEnv
from warehouse_env import WarehouseEnv  # Ambiente multi-robot (aggiornato con ID su robot e nuovo reward shaping)
from wrappers import ParallelEnvToGymEnv  # Wrapper PettingZoo->Gym

class TQDMCallback(BaseCallback):
    """
    Callback con progress bar tqdm che mostra la percentuale di training.
    """
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_num_timesteps = 0

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training progress: 0.0%")

    def _on_step(self) -> bool:
        current_num_timesteps = self.model.num_timesteps
        delta = current_num_timesteps - self.last_num_timesteps
        if delta > 0:
            self.pbar.update(delta)
            self.last_num_timesteps = current_num_timesteps
            pct = (current_num_timesteps / self.total_timesteps) * 100
            if pct > 100:
                pct = 100
            self.pbar.set_description(f"Training progress: {pct:.1f}%")
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

def linear_schedule(initial_value, final_value=1e-4):
    """
    Schedulazione lineare del learning rate da initial_value a final_value.
    progress_remaining: 1.0 (inizio training) -> 0.0 (fine training)
    """
    def schedule(progress_remaining):
        return initial_value * progress_remaining + final_value * (1 - progress_remaining)
    return schedule

def main():
    # ================= (1) Configurazione dell'ambiente e creazione =================
    config = {
        "grid_size": 10,
        "num_robots": 3,
        "num_tasks": 4,
        "num_obstacles": 3,
        "num_charging_stations": 2,
        "max_steps": 1000
    }

    # Crea l'ambiente WarehouseEnv in stile PettingZoo (parallel)
    parallel_env = WarehouseEnv(config=config)

    # Wrappa l'ambiente parallel in single-agent Gym per SB3
    env = ParallelEnvToGymEnv(parallel_env)

    # ================= (2) Configurazione logger SB3 =================
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv"])

    # ================= (3) Creazione del modello PPO =================
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(0.0003),  # schedule lineare del LR
        n_steps=2048,       # batch di rollouts
        batch_size=512,     # batch size per l'update
        max_grad_norm=0.5,
        vf_coef=2.0,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256,256], vf=[256,256]),
            activation_fn=torch.nn.ReLU
        )
    )
    model.set_logger(new_logger)

    # Callback con progress bar
    total_timesteps = 2_000_000
    callback = TQDMCallback(total_timesteps=total_timesteps)

    # ================= (4) Training =================
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Salvataggio modello
    model.save("ppo_warehouse")

    # ================= (5) Valutazione =================
    num_episodes = 10
    obs, info = env.reset(seed=42)
    print("Reset dell'ambiente:", obs)

    all_rewards = []
    for ep in range(num_episodes):
        done = False
        ep_reward = 0.0
        while not done:
            # Azione deterministica
            action, _states = model.predict(obs, deterministic=True)
            step_res = env.step(action)

            # Compatibilità con versioni precedenti di Gymnasium
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                obs, reward, done, info = step_res

            ep_reward += reward

        all_rewards.append(ep_reward)
        obs, info = env.reset(seed=42)
        print(f"Episodio {ep+1}: reward = {ep_reward:.2f}")

    mean_reward = np.mean(all_rewards)
    print(f"Ricompensa media su {num_episodes} episodi: {mean_reward:.2f}")

    # ================= (6) Plot delle ricompense =================
    plt.figure()
    plt.plot(range(1, num_episodes+1), all_rewards, marker='o', label='Reward')
    plt.axhline(mean_reward, color='r', linestyle='--', label=f'Mean={mean_reward:.2f}')
    plt.legend()
    plt.title("Valutazione PPO su WarehouseEnv")
    plt.xlabel("Episodio")
    plt.ylabel("Ricompensa")
    plt.grid()
    plt.savefig("performance_plot.png")
    plt.show()

    # ================= (7) Chiusura =================
    env.close()

if __name__ == "__main__":
    main()
