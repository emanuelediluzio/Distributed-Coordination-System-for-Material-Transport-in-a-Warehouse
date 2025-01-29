# file: train_mappo.py

from warehouse_env import WarehouseEnv
from mappo import MAPPO

def main():
    config = {
        "grid_size": 10,
        "num_robots": 3,
        "num_tasks": 4,
        "num_obstacles": 3,
        "num_charging_stations": 2,
        "max_steps": 200
    }
    env = WarehouseEnv(config=config, enable_render=False)

    # dimensione local obs
    obs_dim = env.observation_space[env.possible_agents[0]].shape[0]
    # dimensione azione
    act_dim = env.action_space[env.possible_agents[0]].n
    # dimensione global state, supponiamo definita come 3 * num_robots
    state_dim = 3 * env.num_robots

    mappo_trainer = MAPPO(
        env=env,
        actor_obs_dim=obs_dim,
        critic_state_dim=state_dim,
        act_dim=act_dim,
        lr=1e-4,
        gamma=0.99,
        lambd=0.95,
        clip_range=0.2,
        n_epochs=5,
        batch_size=1024
    )
    mappo_trainer.train(max_episodes=5000)

if __name__ == "__main__":
    main()
