import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
import numpy as np

from actions import Action  # Assicurati che 'actions.py' sia corretto

class ParallelEnvToGymEnv(gym.Env):
    """
    Wrapper multi-agent su un PettingZoo ParallelEnv.
    Rappresenta l'ambiente come un singolo agente Gym (single-agent),
    che riceve un'osservazione concatenata di tutti i robot e un'azione MultiDiscrete.
    """
    def __init__(self, parallel_env: ParallelEnv):
        super().__init__()
        self.parallel_env = parallel_env
        self.num_robots = self.parallel_env.num_robots
        self.num_charging_stations = self.parallel_env.num_charging_stations

        # Lo spazio di azione è MultiDiscrete: un'azione per ciascuno dei robot
        self.action_space = spaces.MultiDiscrete([len(Action) for _ in range(self.num_robots)])

        # Lo spazio di osservazione è la concatenazione delle osservazioni di tutti i robot
        single_obs_space = self.parallel_env.observation_space[self.parallel_env.possible_agents[0]]
        low_concat = np.tile(single_obs_space.low, self.num_robots)
        high_concat = np.tile(single_obs_space.high, self.num_robots)
        self.observation_space = spaces.Box(
            low=low_concat,
            high=high_concat,
            dtype=np.float32
        )

    def step(self, action):
        """
        Esegue un passo nell'ambiente Gym a singolo agente,
        traducendolo in azioni multiple (una per ogni robot) nell'ambiente parallel.
        
        Args:
            action (np.ndarray): array di lunghezza 'num_robots', ciascun elemento
                                 è un indice corrispondente a un'azione in 'Action'.

        Returns:
            obs (np.ndarray): concatenazione di tutte le osservazioni dei robot
            reward (float): somma delle ricompense individuali
            terminated (bool): True se l'episodio è terminato
            truncated (bool): True se l'episodio è troncato
            info (dict): informazioni aggiuntive
        """
        # Costruisci il dizionario delle azioni per i robot
        actions_dict = {
            aid: action[i] for i, aid in enumerate(self.parallel_env.possible_agents)
        }

        # Esegui step parallelo
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.parallel_env.step(actions_dict)

        # Concatena le osservazioni in un singolo vettore
        obs = np.concatenate([obs_dict[aid] for aid in self.parallel_env.possible_agents], axis=0)
        obs = obs.astype(np.float32)

        # La ricompensa complessiva è la somma delle ricompense di ciascun robot
        reward = sum(rew_dict.values())

        # L'episodio è terminato/troncato se qualunque robot è terminato/troncato
        terminated = any(term_dict.values())
        truncated = any(trunc_dict.values())

        # Combina info di tutti i robot in un unico dictionary
        info = {}
        for aid, inf in info_dict.items():
            info[aid] = inf

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Resetta l'ambiente parallel e restituisce la concatenazione delle osservazioni.

        Returns:
            obs (np.ndarray): concatenazione di tutte le osservazioni
            info (dict): informazioni aggiuntive
        """
        obs_dict, infos = self.parallel_env.reset(seed=seed)
        obs = np.concatenate([obs_dict[aid] for aid in self.parallel_env.possible_agents], axis=0)
        obs = obs.astype(np.float32)

        # Combina info in un unico dict
        merged_info = {}
        for aid, info in infos.items():
            merged_info.update(info)
        
        return obs, merged_info

    def render(self, mode='human'):
        """
        Richiama la funzione render dell'ambiente parallel.
        """
        return self.parallel_env.render(mode)

    def close(self):
        """
        Chiude l'ambiente parallel.
        """
        self.parallel_env.close()


def test_wrapper():
    """
    Funzione di test per verificare il corretto funzionamento del wrapper.
    """
    from actions import Action
    from warehouse_env import WarehouseEnv

    # Crea l'ambiente parallel
    env = WarehouseEnv(config={
        "grid_size": 10,
        "num_robots": 3,
        "num_tasks": 4,
        "num_obstacles": 3,
        "num_charging_stations": 2,
        "max_steps": 1000
    })

    # Wrappa in single-agent Gym
    wrapped_env = ParallelEnvToGymEnv(env)

    # Reset
    obs, info = wrapped_env.reset(seed=42)
    print("Osservazione iniziale:", obs.shape)
    print("Info iniziali:", info)

    # Esegui un passo con azioni di test (es. SEND_BATTERY_STATUS, REQUEST_HELP, NOOP)
    actions = np.array([
        Action.SEND_BATTERY_STATUS.value,
        Action.REQUEST_HELP.value,
        Action.NOOP.value
    ])
    obs, reward, terminated, truncated, info = wrapped_env.step(actions)

    print("\nOsservazione dopo un passo:", obs.shape)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    wrapped_env.render()
    wrapped_env.close()


# Se esegui direttamente questo file, fa partire il test
if __name__ == "__main__":
    test_wrapper()
