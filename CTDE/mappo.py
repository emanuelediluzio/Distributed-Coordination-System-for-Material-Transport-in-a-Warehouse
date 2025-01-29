import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    """
    Attore (policy) condiviso per tutti i robot.
    Input: local obs (dim = obs_dim)
    Output: logits (dim = act_dim) per ciascun robot.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        """
        x.shape = [B, obs_dim]
        output.shape = [B, act_dim]
        """
        return self.net(x)

class CriticNetwork(nn.Module):
    """
    Critic centralizzato, input = global state (dim = state_dim)
    Output = stima scalare V(s)
    """
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s):
        """
        s.shape = [B, state_dim]
        output.shape = [B, 1]
        """
        return self.net(s)

class MAPPO:
    def __init__(self, 
                 env, 
                 actor_obs_dim, 
                 critic_state_dim, 
                 act_dim, 
                 lr=1e-4,
                 gamma=0.99,
                 lambd=0.95,
                 clip_range=0.2,
                 n_epochs=4,
                 batch_size=1024):
        """
        env: environment PettingZoo parallel con:
             - get_local_observation(i)
             - get_global_state()
        actor_obs_dim: dimensione dell'osservazione locale
        critic_state_dim: dimensione dello stato globale
        act_dim: numero di azioni discrete (es. len(Action))
        """
        self.env = env
        self.num_agents = env.num_robots  # es: 3

        # Reti
        self.actor = ActorNetwork(actor_obs_dim, act_dim)
        self.critic = CriticNetwork(critic_state_dim)

        # Ottimizzatori
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Iperparametri PPO / GAE
        self.gamma = gamma
        self.lambd = lambd
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def collect_trajectory(self, max_steps=1000):
        """
        Esegue un singolo rollout (episodio) e raccoglie le transizioni:
        - local obs di ogni agente
        - global state
        - azioni
        - log_prob
        - reward e done (team-based, semplificato)
        - next_global_state
        """
        obs_dict, _ = self.env.reset()
        done = {aid: False for aid in self.env.agents}
        done_global = False

        trajectory = []

        steps = 0
        while not done_global and steps < max_steps:
            steps += 1

            # 1) Prepariamo local obs per tutti i robot
            obs_list = []
            for i, aid in enumerate(self.env.agents):
                obs_list.append(obs_dict[aid])  # shape = (obs_dim,)
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)  # [N, obs_dim]

            # 2) Stato globale
            global_state = torch.tensor(self.env.get_global_state(), dtype=torch.float32)  # shape = (state_dim,)

            # 3) Calcoliamo la distribuzione di azioni
            logits = self.actor(obs_tensor)  # [N, act_dim]
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()           # [N]
            log_prob = dist.log_prob(action) # [N]

            action_np = action.cpu().numpy()

            # 4) Convertiamo in action_dict
            action_dict = {}
            for i, aid in enumerate(self.env.agents):
                action_dict[aid] = action_np[i]

            # 5) Step in env
            obs_next, rew_dict, term_dict, trunc_dict, _info = self.env.step(action_dict)

            # 6) Controllo done globale
            done = {aid: (term_dict[aid] or trunc_dict[aid]) for aid in self.env.agents}
            done_global = any(done.values())

            # 7) Prossimo stato globale
            next_global_state = torch.tensor(self.env.get_global_state(), dtype=torch.float32)

            # 8) Raccolta reward
            #    Semplificazione fully-coop: sum delle reward di tutti i robot
            rew_list = [rew_dict[aid] for aid in self.env.agents]
            team_reward = np.sum(rew_list)  # float

            # Salviamo la transizione
            transition = {
                "obs_local": obs_list,  # shape [N, obs_dim],  ma salviamo come list di array
                "global_state": global_state.numpy(),  # (state_dim,)
                "actions": action_np,                  # shape [N]
                "log_probs": log_prob.detach().cpu().numpy(), # shape [N]
                "team_reward": team_reward,            # float
                "done_global": done_global,
                "next_global_state": next_global_state.numpy()
            }
            trajectory.append(transition)

            obs_dict = obs_next

        return trajectory

    def compute_gae(self, trajectory):
        """
        Calcola i vantage GAE e i returns. 
        - approach fully-coop: usiamo un solo "team_reward" a step
        - un unico value V(s) => la dimensione T della trajectory
        """
        T = len(trajectory)
        v_vals = np.zeros(T+1, dtype=np.float32)  # V(s) su T+1
        advantages = np.zeros(T, dtype=np.float32)
        rewards = np.zeros(T, dtype=np.float32)

        # Estraiamo reward di team
        for t in range(T):
            rewards[t] = trajectory[t]["team_reward"]

        # Controlliamo se l'ultimo step era effettivamente done_global
        if not trajectory[-1]["done_global"]:
            # bootstrap dall'ultimo next_global_state
            last_s = torch.tensor(trajectory[-1]["next_global_state"], dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
            with torch.no_grad():
                v_vals[T] = self.critic(last_s).item()
        else:
            v_vals[T] = 0.0

        # calcoliamo i v_vals[t] da 0..T-1
        for t in reversed(range(T)):
            s_t = torch.tensor(trajectory[t]["global_state"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                v_vals[t] = self.critic(s_t).item()
            delta = rewards[t] + self.gamma * v_vals[t+1] - v_vals[t]
            advantages[t] = delta
            if t < T-1:
                advantages[t] += self.gamma * self.lambd * advantages[t+1]

        returns = v_vals[:T] + advantages
        return advantages, returns

    def build_training_batch(self, trajectory, advantages, returns):
        """
        Costruisce i batch per actor e critic:
         - local obs e azioni (per l'actor)
         - log_probs old (per ratio PPO)
         - global_state e returns (per il critic)
         - advantage
         
        Per un environment con N agenti e T step:
         - la dimensione diventa T*N per l'actor
         - rimane T per il critic (lo usiamo in modo fully-coop).
         
        Semplifichiamo replicando i vantage e i returns per i N agent in quell'istante.
        """
        obs_local_list = []
        action_list = []
        old_logprob_list = []
        advantage_list = []

        global_state_list = []
        return_list = []

        T = len(trajectory)

        for t in range(T):
            trans = trajectory[t]
            obs_locals = trans["obs_local"]     # shape [N, obs_dim], salvati come list
            actions_np = trans["actions"]       # shape [N]
            logprob_np = trans["log_probs"]     # shape [N]
            s_global = trans["global_state"]    # (state_dim,)

            # adv_t e return_t => scalari (della team-based) 
            # replichiamo su dimension N
            adv_t = advantages[t]
            ret_t = returns[t]

            # Per N agenti 
            for i in range(self.num_agents):
                obs_local_list.append(obs_locals[i])    # shape (obs_dim,)
                action_list.append(actions_np[i])       # int
                old_logprob_list.append(logprob_np[i])  # float
                advantage_list.append(adv_t)            # scalar

                global_state_list.append(s_global)      # (state_dim,)
                return_list.append(ret_t)               # scalar

        # Ora convertiamo in tensori
        obs_tensor = torch.tensor(obs_local_list, dtype=torch.float32)            # [T*N, obs_dim]
        action_tensor = torch.tensor(action_list, dtype=torch.long)               # [T*N]
        old_logprob_tensor = torch.tensor(old_logprob_list, dtype=torch.float32)  # [T*N]
        adv_tensor = torch.tensor(advantage_list, dtype=torch.float32)            # [T*N]

        global_state_tensor = torch.tensor(global_state_list, dtype=torch.float32)# [T*N, state_dim]
        returns_tensor = torch.tensor(return_list, dtype=torch.float32)           # [T*N]

        return obs_tensor, action_tensor, old_logprob_tensor, adv_tensor, global_state_tensor, returns_tensor

    def update(self, trajectory):
        """
        Esegue l'aggiornamento stile PPO MAPPO su un singolo batch 
        estratto dall'intero episodio. 
        """
        advantages, returns = self.compute_gae(trajectory)
        # Normalizziamo i vantage (team-based)
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)

        # Creiamo i tensori per (actor, critic)
        (obs_tensor,
         action_tensor,
         old_logprob_tensor,
         adv_tensor,
         global_state_tensor,
         returns_tensor) = self.build_training_batch(trajectory, advantages_norm, returns)

        # Esegui n_epochs di PPO
        for _ in range(self.n_epochs):
            # ========== ACTOR UPDATE ========== 
            logits = self.actor(obs_tensor)   # [T*N, act_dim]
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(action_tensor)  # [T*N]
            ratio = (new_log_probs - old_logprob_tensor).exp()

            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_tensor
            actor_loss = -torch.min(surr1, surr2).mean()

            dist_entropy = dist.entropy().mean()

            # ========== CRITIC UPDATE ========== 
            # Il critic vede global_state_tensor: [T*N, state_dim]
            # ma in realta' i returns sono duplicati. E' un 'trucco' 
            # fully-coop: ognuno "condivide" lo stesso return. 
            # Quindi facciamo media su T*N. E' semplificato.
            
            v_pred = self.critic(global_state_tensor).squeeze(-1)  # [T*N]
            critic_loss = nn.MSELoss()(v_pred, returns_tensor)

            # ========== BACKWARD & STEP ========== 
            # Actor
            self.actor_opt.zero_grad()
            (actor_loss - 0.01 * dist_entropy).backward()
            self.actor_opt.step()

            # Critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # restituiamo i valori di loss finali per logging
        return actor_loss.item(), critic_loss.item()

    def train(self, max_episodes=1000, max_steps=1000):
        """
        Loop di training su 'max_episodes'. 
        """
        for ep in range(max_episodes):
            # Colleziono le transizioni di un episodio
            trajectory = self.collect_trajectory(max_steps=max_steps)

            # Eseguo l'aggiornamento MAPPO
            actor_loss_val, critic_loss_val = self.update(trajectory)

            # Calcoliamo la reward totale di team
            total_reward = sum([tr["team_reward"] for tr in trajectory])
            print(f"[EP {ep+1}] TeamReward={total_reward:.2f}  "
                  f"ActorLoss={actor_loss_val:.3f}  CriticLoss={critic_loss_val:.3f}")
