import os
import numpy as np
import torch
from stable_baselines3 import PPO

# Import dei tuoi moduli locali
# Assicurati che "warehouse_env.py", "wrappers.py" e "actions.py" siano
# importabili (es. nella stessa cartella o nel path)
from warehouse_env import WarehouseEnv
from wrappers import ParallelEnvToGymEnv

def main():
    # 1) Controlla se il file salvato esiste nella cartella corrente
    model_path = "ppo_warehouse.zip"
    if not os.path.isfile(model_path):
        print(f"ERRORE: File {model_path} non trovato!")
        return

    # 2) Carica il modello
    print(f"Carico il modello da {model_path}...")
    model = PPO.load(model_path)
    print("Modello caricato con successo.")

    # 3) Crea l'ambiente di test
    config = {
        "grid_size": 10,
        "num_robots": 3,
        "num_tasks": 4,
        "num_obstacles": 3,
        "num_charging_stations": 2,
        "max_steps": 500  # un po' più corto per test
    }
    # Abilita il render se vuoi vedere i robot muoversi
    env_pz = WarehouseEnv(config=config, enable_render=True)
    env = ParallelEnvToGymEnv(env_pz)

    # 4) Esegui N episodi di test
    n_episodes = 5
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # Usa la politica "deterministic=True" per evitare esplorazione casuale
            action, _states = model.predict(obs, deterministic=True)
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            # Se vuoi vedere i robot muoversi "lentamente", puoi aggiungere un piccolo sleep
            # import time
            # time.sleep(0.05)

        print(f"Episodio {ep+1} completato. Reward totale: {ep_reward:.2f}")

    env.close()
    print("Test completato.")

if __name__ == "__main__":
    main()
