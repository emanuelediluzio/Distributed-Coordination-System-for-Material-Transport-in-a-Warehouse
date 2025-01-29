import os
import torch
import numpy as np

# Importa la definizione dell'ambiente
from warehouse_env import WarehouseEnv  # Il tuo environment PettingZoo stile CTDE
# Importa la classe MAPPO (o almeno l'actor) da dove l'hai definita
from mappo import MAPPO

def test_mappo(
    env_config=None,
    model_path="mappo_actor.pth",
    num_episodes=5,
    max_steps=1000,
    render=True
):
    """
    Esegue alcune partite di test con un MAPPO addestrato,
    caricando i pesi dal file `model_path`.
    """
    if env_config is None:
        env_config = {
            "grid_size": 10,
            "num_robots": 3,
            "num_tasks": 3,
            "num_obstacles": 2,
            "num_charging_stations": 1,
            "max_steps": max_steps
        }

    # 1) Crea l'ambiente abilitando il render (se vuoi vederlo) 
    env = WarehouseEnv(config=env_config, enable_render=render)

    # 2) Crea un'istanza di MAPPO (o solo l'actor) con le dimensioni giuste
    #    Devi replicare i parametri usati in fase di training
    #    Esempio: obs_dim (local), act_dim, state_dim (critic)...
    actor_obs_dim = 8   # come definito in _observation_space=(8,)
    critic_state_dim = 3*3  # Esempio: se get_global_state() => 3 robot * 3 info a robot
    act_dim = env.action_space[env.possible_agents[0]].n  # Esempio se Discrete(16) ?

    # Inizializza MAPPO: 
    # (oppure, se la classe MAPPO e i costruttori differiscono, adattalo)
    mappo = MAPPO(
        env=env, 
        actor_obs_dim=actor_obs_dim, 
        critic_state_dim=critic_state_dim, 
        act_dim=act_dim,
        lr=1e-4,
        gamma=0.99,
        lambd=0.95,
        clip_range=0.2,
        n_epochs=10,
        batch_size=1024
    )

    # 3) Carica i pesi del tuo attore (assumendo che MAPPO.actor sia un nn.Module).
    #    Se hai definito un metodo mappo.load(path) usalo. Altrimenti, manuale:
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        mappo.actor.load_state_dict(checkpoint)
        print(f"Caricati i pesi dell'attore da '{model_path}'")
    except Exception as e:
        print(f"Impossibile caricare i pesi da '{model_path}': {e}")
        return

    # 4) Loop su più episodi di test
    for ep in range(num_episodes):
        obs_dict, _info = env.reset()
        done_dict = {aid: False for aid in env.agents}
        done_global = False
        episode_reward = 0.0

        steps = 0
        while not done_global and steps < max_steps:
            steps += 1

            # Costruiamo la lista di local-obs per i robot
            obs_list = []
            agent_ids = env.agents  # es. ["robot_0", "robot_1", ...]
            for i, aid in enumerate(agent_ids):
                obs_list.append(obs_dict[aid])  # shape=(8,)

            # Passiamo a tensore
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
            # Attore produce logits => azioni => campionamento
            with torch.no_grad():
                logits = mappo.actor(obs_tensor)  # shape [N, act_dim]
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()  # shape=[N]

            # Convertiamole in dizionario per PettingZoo
            actions_np = actions.numpy()
            action_dict = {}
            for i, aid in enumerate(agent_ids):
                action_dict[aid] = actions_np[i]

            # Step
            obs_next, rew_dict, term_dict, trunc_dict, _info_dict = env.step(action_dict)

            # Somma reward
            for aid in rew_dict:
                episode_reward += rew_dict[aid]

            # Aggiorna done e obs
            done_dict = {aid: (term_dict[aid] or trunc_dict[aid]) for aid in agent_ids}
            done_global = any(done_dict.values())
            obs_dict = obs_next

        print(f"Episodio {ep+1}/{num_episodes}: reward totale = {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_mappo()
