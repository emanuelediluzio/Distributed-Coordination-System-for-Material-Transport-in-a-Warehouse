import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import logging
import os
from datetime import datetime
import pygame
from enum import Enum
import random
from actions import Action

# Configurazione del logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = f'{log_dir}/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#######################################################
# Definizione Enums e classi base: Direction, Agent
#######################################################

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def turn_left(self):
        if self == Direction.UP:
            return Direction.LEFT
        elif self == Direction.LEFT:
            return Direction.DOWN
        elif self == Direction.DOWN:
            return Direction.RIGHT
        elif self == Direction.RIGHT:
            return Direction.UP

    def turn_right(self):
        if self == Direction.UP:
            return Direction.RIGHT
        elif self == Direction.RIGHT:
            return Direction.DOWN
        elif self == Direction.DOWN:
            return Direction.LEFT
        elif self == Direction.LEFT:
            return Direction.UP

class Agent:
    counter = 0

    def __init__(self, x: int, y: int, color: int, direction: Direction, msg_bits: int = 16):
        Agent.counter += 1
        self.id = Agent.counter
        self.x = x
        self.y = y
        self.color = color
        self.direction = direction
        self.message = np.zeros(msg_bits, dtype=int)
        self.carrying_shelf: int = -1

    def reset(self):
        self.carrying_shelf = -1
        self.message.fill(0)

    def update_position(self, new_x: int, new_y: int):
        self.x = new_x
        self.y = new_y

#######################################################
# Visualizzatore Pygame
#######################################################

class WarehouseVisualizer:
    def __init__(self, env, window_size=800, enable_render=True):
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = window_size // self.grid_size
        self.window_size = window_size
        self.enable_render = enable_render

        if self.enable_render:
            pygame.init()
            self.legend_width = 200
            self.screen = pygame.display.set_mode((window_size + self.legend_width, window_size))
            pygame.display.set_caption("Warehouse Simulation")
            self.clock = pygame.time.Clock()
            # Font per griglia/robot
            self.font = pygame.font.SysFont(None, 20)
            # Font un po' più piccolo per la legenda
            self.font_legend = pygame.font.SysFont(None, 16)

    def render(self):
        if not self.enable_render:
            return

        self.screen.fill((255, 255, 255))

        self._draw_grid()
        self._draw_tasks()
        self._draw_dropoff_zones()
        self._draw_obstacles()
        self._draw_charging_stations()
        self._draw_robots()
        self._draw_legend()      
        self._draw_robot_info()

        pygame.display.flip()
        self.clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def _draw_grid(self):
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.window_size, y))

    def _draw_tasks(self):
        for i, delivered in enumerate(self.env.state['tasks_delivered']):
            tx, ty = self.env.state['tasks'][i]
            color = (255, 0, 0) if not delivered else (128, 128, 128)
            pygame.draw.rect(
                self.screen, color,
                (tx * self.cell_size, ty * self.cell_size, self.cell_size, self.cell_size)
            )

    def _draw_dropoff_zones(self):
        for i, delivered in enumerate(self.env.state['tasks_delivered']):
            dx, dy = self.env.state['dropoff_zones'][i]
            color = (0, 255, 0) if not delivered else (150, 255, 150)
            pygame.draw.rect(
                self.screen, color,
                (dx * self.cell_size, dy * self.cell_size, self.cell_size, self.cell_size)
            )

    def _draw_obstacles(self):
        for ox, oy in self.env.state['obstacles']:
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                (ox * self.cell_size, oy * self.cell_size, self.cell_size, self.cell_size)
            )

    def _draw_charging_stations(self):
        for cx, cy in self.env.state['charging_stations']:
            pygame.draw.circle(
                self.screen, (255, 165, 0),
                (cx * self.cell_size + self.cell_size // 2, cy * self.cell_size + self.cell_size // 2),
                self.cell_size // 3
            )

    def _draw_robots(self):
        for ag in self.env.agent_objects:
            rx, ry = ag.x, ag.y
            # Robot come cerchio
            pygame.draw.circle(
                self.screen, (0, 0, 255),
                (rx * self.cell_size + self.cell_size // 2,
                 ry * self.cell_size + self.cell_size // 2),
                self.cell_size // 3
            )
            # ID centrato sul cerchio
            robot_id_text = self.font.render(f"R{ag.id}", True, (255, 255, 255))
            text_rect = robot_id_text.get_rect(center=(
                rx * self.cell_size + self.cell_size // 2,
                ry * self.cell_size + self.cell_size // 2
            ))
            self.screen.blit(robot_id_text, text_rect)

    def _draw_legend(self):
        """
        Legenda più piccola:
          - uso self.font_legend
          - dimensioni icone ridotte
          - riduco line_spacing
        """
        x_start = self.window_size + 10
        y_start = 10
        line_spacing = 18  # piu piccolo di 25

        legend_elements = [
            ("Robot", (0, 0, 255)),
            ("Task", (255, 0, 0)),
            ("Delivered Task", (128, 128, 128)),
            ("Dropoff Zone", (0, 255, 0)),
            ("Delivered Dropoff", (150, 255, 150)),
            ("Obstacle", (0, 0, 0)),
            ("Charging Station", (255, 165, 0)),
        ]

        # Titolo legenda
        title = self.font_legend.render("Legenda", True, (0, 0, 0))
        self.screen.blit(title, (x_start, y_start))
        y = y_start + line_spacing

        # Più piccole le icone: raggio=5, rect=10
        circle_radius = 5
        rect_size = 10

        for element, color in legend_elements:
            if element in ["Robot", "Charging Station"]:
                pygame.draw.circle(self.screen, color, (x_start + 5, y + 5), circle_radius)
            else:
                pygame.draw.rect(self.screen, color, (x_start, y, rect_size, rect_size))
            text = self.font_legend.render(element, True, (0, 0, 0))
            self.screen.blit(text, (x_start + 15, y - 1))
            y += line_spacing

        # Mini info su comunicazione
        extra_title = self.font_legend.render("Bits:", True, (0, 0, 0))
        self.screen.blit(extra_title, (x_start, y))
        y += line_spacing

        bit_info = [
            "0-1: Batteria",
            "2: Vicinanza Carico",
            "3: Vicinanza Scarico",
            "4-5: Ricarica",
            "6: HELP",
            "7: ASSIGN_TASK",
            "8: PATH_BLOCKED",
            "9: PRIORITY_UPD",
            "10-15: Riservati"
        ]
        # Riduco la spaziatura ulteriormente
        mini_line_spacing = 12
        for desc in bit_info:
            txt = self.font_legend.render(desc, True, (0, 0, 0))
            self.screen.blit(txt, (x_start, y))
            y += mini_line_spacing

    def _draw_robot_info(self):
        line_height = self.font.get_linesize()
        start_y = self.window_size - line_height * self.env.num_robots - 10

        for idx, ag in enumerate(self.env.agent_objects):
            battery = self.env.state['robot_batteries'][idx]
            carrying = self.env.state['robot_carrying'][idx]
            msg = ag.message.tolist()
            info_text = (f"Robot {idx+1} | Pos=({ag.x},{ag.y}) "
                         f"| Batt={battery}% | Carry={carrying} | Msg={msg}")
            text_surf = self.font.render(info_text, True, (0, 0, 0))
            self.screen.blit(text_surf, (10, start_y + idx * line_height))

    def close(self):
        if self.env and self.enable_render:
            pygame.display.quit()
            pygame.quit()

#######################################################
# WarehouseEnv (PettingZoo ParallelEnv)
#######################################################

class WarehouseEnv(ParallelEnv):

    """
    Ambiente multi-robot parallel (PettingZoo), con:
      - Azioni definite in actions.py (NOOP, FORWARD, LEFT, RIGHT, ecc.)
      - Batteria che si decrementa ogni step
      - pick/deliver di tasks
      - Comunicazione tra robot via bit
      - max_steps => truncated
    """
    metadata = {"render_modes": ["human"], "name": "warehouse_v0"}

    def __init__(self, config=None, enable_render=True):
        super().__init__()
        if config is None:
            config = {}

        # Parametri
        self.grid_size = config.get("grid_size", 10)
        self.num_robots = config.get("num_robots", 3)
        self.num_tasks = config.get("num_tasks", 3)
        self.num_obstacles = config.get("num_obstacles", 2)
        self.num_charging_stations = config.get("num_charging_stations", 1)
        self.max_steps = config.get("max_steps", 2000)

        self.current_step = 0
        self.success_counter = 0

        self.possible_agents = [f"robot_{i}" for i in range(self.num_robots)]
        self.agents = self.possible_agents[:]

        self.agent_objects = []
        for i in range(self.num_robots):
            ax = random.randint(0, self.grid_size - 1)
            ay = random.randint(0, self.grid_size - 1)
            adir = Direction(random.randint(0, 3))
            self.agent_objects.append(Agent(ax, ay, i, adir, msg_bits=16))

        self.agent_dict = {aid: aobj for aid,aobj in zip(self.agents, self.agent_objects)}

        # Stato globale
        self.state = {
            'robots': [],
            'tasks': [],
            'dropoff_zones': [],
            'obstacles': [],
            'charging_stations': [],
            'tasks_delivered': [],
            'robot_carrying': [],
            'robot_batteries': [],
            'assigned_tasks': [],
            'robot_directions': [],
            'just_picked': []
        }

        self.robot_visualizer = WarehouseVisualizer(self, enable_render=enable_render)
        self.prev_positions = [(-1, -1)] * self.num_robots
        self.stay_counter = [0]*self.num_robots

        self._action_space = {aid: spaces.Discrete(len(Action)) for aid in self.possible_agents}
        obs_shape = 9 + 16*(self.num_robots - 1) + 2*self.num_charging_stations
        self._observation_space = {
            aid: spaces.Box(low=0.0, high=1.0, shape=(obs_shape,), dtype=np.float32)
            for aid in self.possible_agents
        }

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.success_counter = 0

        tasks = np.random.randint(0, self.grid_size, (self.num_tasks, 2))
        dropoff = np.random.randint(0, self.grid_size, (self.num_tasks, 2))
        obstacles = np.random.randint(0, self.grid_size, (self.num_obstacles, 2))
        ch_stations = np.random.randint(0, self.grid_size, (self.num_charging_stations, 2))

        occupied = set((ag.x, ag.y) for ag in self.agent_objects)

        # Ostacoli
        for i in range(self.num_obstacles):
            while (obstacles[i][0], obstacles[i][1]) in occupied:
                obstacles[i] = np.random.randint(0, self.grid_size, 2)
            occupied.add((obstacles[i][0], obstacles[i][1]))

        # Tasks e dropoff
        for i in range(self.num_tasks):
            while ((tasks[i][0], tasks[i][1]) in occupied or
                   (dropoff[i][0], dropoff[i][1]) in occupied):
                tasks[i] = np.random.randint(0, self.grid_size, 2)
                dropoff[i] = np.random.randint(0, self.grid_size, 2)
            occupied.add((tasks[i][0], tasks[i][1]))
            occupied.add((dropoff[i][0], dropoff[i][1]))

        # Charging
        for i in range(self.num_charging_stations):
            while (ch_stations[i][0], ch_stations[i][1]) in occupied:
                ch_stations[i] = np.random.randint(0, self.grid_size, 2)
            occupied.add((ch_stations[i][0], ch_stations[i][1]))

        # Reset agenti
        for ag in self.agent_objects:
            ag.reset()

        self.state['robots'] = [[ag.x, ag.y] for ag in self.agent_objects]
        self.state['tasks'] = tasks
        self.state['dropoff_zones'] = dropoff
        self.state['obstacles'] = obstacles
        self.state['charging_stations'] = ch_stations
        self.state['tasks_delivered'] = [False]*self.num_tasks
        self.state['robot_carrying'] = [-1]*self.num_robots
        self.state['robot_batteries'] = [100]*self.num_robots
        self.state['assigned_tasks'] = [None]*self.num_robots
        self.state['robot_directions'] = [ag.direction for ag in self.agent_objects]
        self.state['just_picked'] = [False]*self.num_robots

        self.stay_counter = [0]*self.num_robots
        self.prev_positions = [(-1,-1)]*self.num_robots

        obs_dict = {}
        for aid in self.agents:
            idx = int(aid.split('_')[1])
            obs_dict[aid] = self._get_observation(idx)
        infos = {aid: {} for aid in self.agents}

        return obs_dict, infos

    def step(self, actions):
        self.current_step += 1
        rewards = {aid: 0.0 for aid in self.agents}
        terminated = {aid: False for aid in self.agents}
        truncated = {aid: False for aid in self.agents}
        infos = {aid: {} for aid in self.agents}

        # Applica le azioni
        for aid, act in actions.items():
            idx = int(aid.split('_')[1])
            self._apply_action(idx, act)
            try:
                action_name = Action(act).name
            except ValueError:
                action_name = "UNKNOWN"
            logger.debug(f"Robot {idx} ha eseguito l'azione {action_name}")

        # Calcolo priorità
        priorities = []
        for idx, ag in enumerate(self.agent_objects):
            battery = self.state['robot_batteries'][idx]
            carrying = self.state['robot_carrying'][idx]
            if carrying == -1:
                # cerco il task più vicino
                not_del = [t_i for t_i, d in enumerate(self.state['tasks_delivered']) if not d]
                if not_del:
                    distances = [
                        np.linalg.norm([ag.x - self.state['tasks'][t][0],
                                        ag.y - self.state['tasks'][t][1]])
                        for t in not_del
                    ]
                    dist = min(distances) if distances else self.grid_size
                else:
                    dist = self.grid_size
            else:
                dx, dy = self.state['dropoff_zones'][carrying]
                dist = np.linalg.norm([ag.x - dx, ag.y - dy])

            priority = battery / (dist + 1)
            priorities.append((priority, idx))

        # Ordina per priorità
        priorities.sort(reverse=True)
        num_allowed = max(1, self.num_robots - self.num_robots // 3)
        allowed_idxs = [idx for (_, idx) in priorities[:num_allowed]]

        logger.debug(f"Robots autorizzati a muoversi in questo step: {allowed_idxs}")

        # Se un robot non è in allowed_idxs => NOOP
        for idx, ag in enumerate(self.agent_objects):
            if idx not in allowed_idxs:
                self._apply_action(idx, Action.NOOP.value)

        # Coordinamento basato sui messaggi (REQUEST_HELP)
        for idx, ag in enumerate(self.agent_objects):
            for other_idx, other_ag in enumerate(self.agent_objects):
                if other_idx == idx:
                    continue
                if other_ag.message[12] == 1:  # REQUEST_HELP
                    # se l'altro ha batteria < 30
                    if self.state['robot_batteries'][other_idx] < 30:
                        if self.state['robot_carrying'][idx] == -1:
                            not_del = [t_i for t_i, d in enumerate(self.state['tasks_delivered']) if not d]
                            for t_i in not_del:
                                if t_i not in self.state['assigned_tasks']:
                                    self.state['assigned_tasks'][idx] = t_i
                                    self.state['robot_carrying'][idx] = t_i
                                    self.state['just_picked'][idx] = True
                                    logger.info(f"Robot {idx} aiuta Robot {other_idx} prendendo task {t_i}")
                                    break

        # Aggiorna posizioni e direzioni
        for i, ag in enumerate(self.agent_objects):
            old_pos = self.prev_positions[i]
            new_pos = (ag.x, ag.y)
            if new_pos == old_pos:
                self.stay_counter[i] += 1
            else:
                self.stay_counter[i] = 0
            self.prev_positions[i] = new_pos
            self.state['robots'][i] = [ag.x, ag.y]
            self.state['robot_directions'][i] = ag.direction

        # Batteria -1
        for i in range(self.num_robots):
            self.state['robot_batteries'][i] = max(0, self.state['robot_batteries'][i] - 1)

        # Calcolo reward
        for aid in self.agents:
            idx = int(aid.split('_')[1])
            rew = self._compute_reward(idx)
            rewards[aid] += rew

        # Batteria zero => terminato
        for i in range(self.num_robots):
            if self.state['robot_batteries'][i] <= 0:
                agent_id = f"robot_{i}"
                rewards[agent_id] -= 200.0
                terminated[agent_id] = True
                logger.info(f"Robot {i} batteria zero => terminato.")

        # Tutti i tasks consegnati?
        if all(self.state['tasks_delivered']):
            for aid in self.agents:
                terminated[aid] = True
            logger.info("Tutti i tasks completati => fine ep.")

        # max_steps => truncated
        if self.current_step >= self.max_steps:
            for aid in self.agents:
                truncated[aid] = True
            logger.info("Raggiunto max_steps => truncated.")

        self.robot_visualizer.render()

        # Nuove osservazioni
        obs_dict = {}
        for aid in self.agents:
            idx = int(aid.split('_')[1])
            obs_dict[aid] = self._get_observation(idx)
        infos = {aid: {} for aid in self.agents}

        return obs_dict, rewards, terminated, truncated, infos

    def _apply_action(self, idx, act):
        ag = self.agent_objects[idx]
        try:
            action_enum = Action(act)
        except ValueError:
            logger.error(f"Azione non valida: {act} per Robot {idx}")
            action_enum = Action.NOOP

        if action_enum == Action.NOOP:
            pass
        elif action_enum == Action.LEFT:
            ag.direction = ag.direction.turn_left()
        elif action_enum == Action.RIGHT:
            ag.direction = ag.direction.turn_right()
        elif action_enum == Action.FORWARD:
            dx, dy = 0, 0
            if ag.direction == Direction.UP:
                dy = -1
            elif ag.direction == Direction.DOWN:
                dy = 1
            elif ag.direction == Direction.LEFT:
                dx = -1
            elif ag.direction == Direction.RIGHT:
                dx = 1
            nx, ny = ag.x + dx, ag.y + dy
            if self._is_free(nx, ny):
                ag.update_position(nx, ny)
            else:
                # Penalità se urta ostacolo
                self.state['robot_batteries'][idx] = max(0, self.state['robot_batteries'][idx] - 5)
                logger.info(f"Robot {idx} urta un ostacolo => penalità")
        elif action_enum == Action.TOGGLE_LOAD:
            self._pick_or_deliver(idx)
        elif action_enum == Action.SHIFT_BATTERY:
            self._shift_battery(idx)
        elif action_enum == Action.SET_MESSAGE_BIT0:
            ag.message[0] = 1
            logger.info(f"Robot {idx} imposta bit0 del messaggio a 1")
        elif action_enum == Action.SET_MESSAGE_BIT1:
            ag.message[1] = 1
            logger.info(f"Robot {idx} imposta bit1 del messaggio a 1")
        elif action_enum == Action.CLEAR_MESSAGE:
            ag.message.fill(0)
            logger.info(f"Robot {idx} azzera i bit di messaggio")
        elif action_enum == Action.SEND_BATTERY_STATUS:
            battery = self.state['robot_batteries'][idx]
            if battery >= 75:
                ag.message[0] = 0
                ag.message[1] = 0
            elif battery >= 50:
                ag.message[0] = 0
                ag.message[1] = 1
            elif battery >= 25:
                ag.message[0] = 1
                ag.message[1] = 0
            else:
                ag.message[0] = 1
                ag.message[1] = 1
            logger.info(f"Robot {idx} aggiorna stato batteria a {ag.message[0:2]}")
        elif action_enum == Action.SEND_PROXIMITY_INFO:
            rx, ry = self.state['robots'][idx]
            proximity_load = 0
            proximity_unload = 0
            for task in self.state['tasks']:
                tx, ty = task
                if abs(rx - tx) <= 1 and abs(ry - ty) <= 1:
                    proximity_load = 1
                    break
            for drop in self.state['dropoff_zones']:
                dx, dy = drop
                if abs(rx - dx) <= 1 and abs(ry - dy) <= 1:
                    proximity_unload = 1
                    break
            ag.message[2] = proximity_load
            ag.message[3] = proximity_unload
            logger.info(f"Robot {idx} aggiorna vicinanza carico={proximity_load}, scarico={proximity_unload}")
        elif action_enum == Action.SEND_CHARGING_STATION_INFO:
            charging_status = 0
            for cs in self.state['charging_stations']:
                cx, cy = cs
                if any(aobj.x == cx and aobj.y == cy for aobj in self.agent_objects):
                    charging_status = 1
                    break
            if charging_status == 0:
                ag.message[4] = 0
                ag.message[5] = 0
            else:
                ag.message[4] = 0
                ag.message[5] = 1
            logger.info(f"Robot {idx} aggiorna stato base ricarica a {ag.message[4:6]}")
        elif action_enum == Action.REQUEST_HELP:
            ag.message[12] = 1
            logger.info(f"Robot {idx} richiede aiuto (bit12 = 1)")
        elif action_enum == Action.SEND_TASK_ASSIGNMENT:
            ag.message[13] = 1
            logger.info(f"Robot {idx} invia assegnazione task (bit13 = 1)")
        elif action_enum == Action.SEND_PATH_BLOCKED:
            ag.message[14] = 1
            logger.info(f"Robot {idx} segnala percorso bloccato (bit14 = 1)")
        elif action_enum == Action.SEND_PRIORITY_UPDATE:
            ag.message[15] = 1
            logger.info(f"Robot {idx} invia aggiornamento priorità (bit15 = 1)")

    def _shift_battery(self, idx):
        rx, ry = self.agent_objects[idx].x, self.agent_objects[idx].y
        for j, other_ag in enumerate(self.agent_objects):
            if j == idx:
                continue
            if (other_ag.x, other_ag.y) == (rx, ry):
                amt = 10
                if self.state['robot_batteries'][idx] >= amt:
                    self.state['robot_batteries'][idx] -= amt
                    self.state['robot_batteries'][j] = min(100, self.state['robot_batteries'][j] + amt)
                    logger.info(f"Robot {idx} cede {amt}% di batteria al Robot {j}")
                else:
                    logger.info(f"Robot {idx} non ha abbastanza batteria => SHIFT_BATTERY fallito.")

    def _pick_or_deliver(self, idx):
        agent = self.agent_objects[idx]
        rx, ry = agent.x, agent.y
        carrying = self.state['robot_carrying'][idx]
        if carrying == -1:
            # pick
            for i, delivered in enumerate(self.state['tasks_delivered']):
                if delivered:
                    continue
                tx, ty = self.state['tasks'][i]
                if (rx, ry) == (tx, ty):
                    self.state['robot_carrying'][idx] = i
                    self.state['just_picked'][idx] = True
                    logger.info(f"Robot {idx} pick task {i}")
                    break
        else:
            # deliver
            dx, dy = self.state['dropoff_zones'][carrying]
            if (rx, ry) == (dx, dy):
                self.state['tasks_delivered'][carrying] = True
                self.state['robot_carrying'][idx] = -1
                self.success_counter += 1
                logger.info(f"Robot {idx} deliver task {carrying}")

    def _is_free(self, nx, ny):
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            return False
        if (nx, ny) in self.state['obstacles']:
            return False
        return True

    def _compute_reward(self, idx):
        """
        Reward shaping leggermente modificato per incoraggiare
        il movimento e la consegna task, ed evitare lo spam di comunicazioni.
        """
        # Penalità step meno severa
        reward = -0.01

        # Penalità se rimangono fermi troppo a lungo
        if self.stay_counter[idx] > 10:
            reward -= 2.0

        # Se appena pickato un task => +50
        if self.state['just_picked'][idx]:
            reward += 50.0
            self.state['just_picked'][idx] = False

        carrying = self.state['robot_carrying'][idx]
        rx, ry = self.state['robots'][idx]

        # Se ha consegnato un task => +500
        if carrying != -1:
            if self.state['tasks_delivered'][carrying]:
                reward += 500.0
            else:
                # Bonus per avvicinarsi al dropoff
                dx, dy = self.state['dropoff_zones'][carrying]
                dist = np.linalg.norm([rx - dx, ry - dy])
                reward += 5.0 * np.exp(-0.2 * dist)
        else:
            # Bonus se ci si avvicina a un task
            not_del = [t_i for t_i, d in enumerate(self.state['tasks_delivered']) if not d]
            if not_del:
                t_i = not_del[0]
                tx, ty = self.state['tasks'][t_i]
                dist = np.linalg.norm([rx - tx, ry - ty])
                reward += 4.0 * np.exp(-0.2 * dist)

        # Penalità batteria bassa
        battery = self.state['robot_batteries'][idx]
        if battery < 20:
            reward -= 10.0
        elif battery < 50:
            reward -= 5.0

        # Riduciamo i bonus/penalità dei bit per evitare spam
        # Coerenza bit batteria
        expected_battery_status = self._encode_battery_status(battery)
        actual_battery_status = self.agent_objects[idx].message[0]*2 + self.agent_objects[idx].message[1]
        if actual_battery_status == expected_battery_status:
            reward += 0.2
        else:
            reward -= 0.2

        # Prossimità carico
        proximity_load = self.agent_objects[idx].message[2]
        if proximity_load == self._is_near_task(idx):
            reward += 0.2
        else:
            reward -= 0.2

        # Prossimità scarico
        proximity_unload = self.agent_objects[idx].message[3]
        if proximity_unload == self._is_near_dropoff(idx):
            reward += 0.2
        else:
            reward -= 0.2

        # Stato base ricarica
        charging_status = (self.agent_objects[idx].message[4]*2
                           + self.agent_objects[idx].message[5])
        if charging_status == self._get_charging_station_status():
            reward += 0.2
        else:
            reward -= 0.2

        # REQUEST_HELP => +2 se <30% altrimenti -1
        if self.agent_objects[idx].message[12] == 1:
            if battery < 30:
                reward += 2.0
            else:
                reward -= 1.0

        # SEND_TASK_ASSIGNMENT => +1 se carrying != -1, altrimenti -1
        if self.agent_objects[idx].message[13] == 1:
            if carrying != -1:
                reward += 1.0
            else:
                reward -= 1.0

        # SEND_PATH_BLOCKED => +1 se stai fermo, -1 se no
        if self.agent_objects[idx].message[14] == 1:
            if self.stay_counter[idx] > 0:
                reward += 1.0
            else:
                reward -= 1.0

        # SEND_PRIORITY_UPDATE => +1
        if self.agent_objects[idx].message[15] == 1:
            reward += 1.0

        return reward

    def _encode_battery_status(self, battery):
        if battery >= 75:
            return 0
        elif battery >= 50:
            return 1
        elif battery >= 25:
            return 2
        else:
            return 3

    def _is_near_task(self, idx):
        rx, ry = self.state['robots'][idx]
        for task in self.state['tasks']:
            tx, ty = task
            if abs(rx - tx) <= 1 and abs(ry - ty) <= 1:
                return True
        return False

    def _is_near_dropoff(self, idx):
        carrying = self.state['robot_carrying'][idx]
        if carrying == -1:
            return False
        rx, ry = self.state['robots'][idx]
        dx, dy = self.state['dropoff_zones'][carrying]
        return (abs(rx - dx) <= 1) and (abs(ry - dy) <= 1)

    def _get_charging_station_status(self):
        for cs in self.state['charging_stations']:
            cx, cy = cs
            if any(ag.x == cx and ag.y == cy for ag in self.agent_objects):
                return 1
        return 0

    def _get_observation(self, i):
        num_other_robots = self.num_robots - 1
        num_charging_stations = self.num_charging_stations
        obs_len = 9 + 16*num_other_robots + 2*num_charging_stations
        obs = np.zeros(obs_len, dtype=np.float32)

        rx, ry = self.state['robots'][i]
        carrying = self.state['robot_carrying'][i]
        obs[0:2] = np.array([rx, ry]) / self.grid_size
        obs[2] = 1.0 if carrying != -1 else 0.0

        # Offset per il task assegnato (se esiste)
        if self.state['assigned_tasks'][i] is not None:
            t_idx = self.state['assigned_tasks'][i]
            tx, ty = self.state['tasks'][t_idx]
            obs[3:5] = (np.array([tx, ty]) - np.array([rx, ry])) / self.grid_size
        else:
            obs[3:5] = 0.0

        obs[5] = self.state['robot_batteries'][i] / 100.0
        obs[6] = float(i) / self.num_robots
        obs[7] = self.current_step / self.max_steps
        dval = float(self.state['robot_directions'][i].value)
        obs[8] = dval / 3.0

        message_index = 9
        # Bit degli altri robot
        for j, other_ag in enumerate(self.agent_objects):
            if j != i:
                obs[message_index + 0] = other_ag.message[0] / 3.0
                obs[message_index + 1] = other_ag.message[1] / 1.0
                obs[message_index + 2] = other_ag.message[2] / 1.0
                obs[message_index + 3] = other_ag.message[3] / 1.0
                obs[message_index + 4] = other_ag.message[4] / 3.0
                obs[message_index + 5] = other_ag.message[5] / 1.0
                obs[message_index + 6] = other_ag.message[12] / 1.0
                obs[message_index + 7] = other_ag.message[13] / 1.0
                obs[message_index + 8] = other_ag.message[14] / 1.0
                obs[message_index + 9] = other_ag.message[15] / 1.0
                for bit_i in range(10, 16):
                    obs[message_index + bit_i] = other_ag.message[bit_i] / 1.0
                message_index += 16

        # Posizioni charging station
        for cx, cy in self.state['charging_stations']:
            obs[message_index]   = cx / self.grid_size
            obs[message_index+1] = cy / self.grid_size
            message_index += 2

        return obs

    def render(self, mode='human'):
        self.robot_visualizer.render()

    def close(self):
        self.robot_visualizer.close()
