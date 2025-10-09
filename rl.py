import argparse
import pickle
import time

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from graph_explorer import GraphExplorer
from gymnasium_env import ACTION_STR_TO_INT, PokeEnv
from q import QAgent, QState


def parse_args():
    parser = argparse.ArgumentParser(description="PokeEnv Gym Environment")
    parser.add_argument(
        "--port",
        type=int,
        default=6363,
        help="Port number for the server (default: 5000)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable recording of the session",
    )
    parser.add_argument(
        "--pygame",
        action="store_true",
        help="Use pygame window for rendering",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development server with no anti-cheat (faster for training)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--timelimit",
        type=int,
        default=50_000,
        help="Number of steps per episode (default: 100,000)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0,
        help="Entropy coefficient for PPO (default: 0)",
    )
    parser.add_argument(
        "--human-timelimit",
        type=int,
        default=(60 * 60 * 5.9)  # 5.9 hours
    )
    return parser.parse_args()


def random():
    args = parse_args()
    env = PokeEnv(args, seed=0)
    episode = 0
    obs, info = env.reset()
    for i in range(1_000_000_000):
        print(f"Step {i}")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # print(f"Episode {episode} total reward: {info['episode']['r']}")
            obs = env.reset()
    env.close()


def graph_explorer():
    args = parse_args()
    env = PokeEnv(args, seed=0)
    agent = GraphExplorer()

    done = False
    _, info = env.reset()
    step = 0

    start_time = time.time()

    while not done:
        before_coord = info["coord"]
        agent.add_coord_to_graph(before_coord)

        action = agent.get_action(before_coord)
        action_number = ACTION_STR_TO_INT[action]
        _, _, terminated, truncated, info = env.step(action_number)

        after_coord = info["coord"]

        if info["game_state"] == "overworld" and action in ["LEFT", "RIGHT", "UP", "DOWN"]:
            agent.add_edge_to_graph(before_coord, after_coord, action)

        done = terminated or truncated
        step += 1

        if step % 1000 == 0:
            with open(f"graph_{step}.pkl", "wb") as f:
                pickle.dump(agent, f)

        if time.time() - start_time > args.human_timelimit:
            print("Time limit reached, ending episode.")
            break

    env.close()
    print(f"Total time: {time.time() - start_time} seconds")


def q():
    args = parse_args()
    env = PokeEnv(args, seed=0)
    agent = QAgent(
        lr=0.1, gamma=0.99, num_actions=4, init_q=0.1
    )

    epsilon = 0.1
    total_steps = 0

    def get_game_state(info: dict) -> str:
        try:
            return info["game_state"]
        except KeyError:
            return "unknown"

    while total_steps < 1_000_000:
        done = False
        _, info = env.reset()
        state = QState(
            coord=info["coord"],
            milestones=info["total_reward"],
            game_state=get_game_state(info),
        )

        while not done:
            if np.random.rand() < 0.5:
                action = np.random.randint(2)
                _, reward, terminated, truncated, info = env.step(action)
            else:
                action = agent.get_action(state, epsilon)
                _, reward, terminated, truncated, info = env.step(action + 2)

            next_state = QState(
                coord=info["coord"],
                milestones=info["total_reward"],
                game_state=get_game_state(info),
            )
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done=done)

            state = next_state

            total_steps += 1


    with open("q_agent.pkl", "wb") as f:
        pickle.dump(agent, f)
    
    env.close()


def ppo():
    args = parse_args()
    envs = DummyVecEnv(
        [lambda: PokeEnv(args, seed=i) for i in range(args.num_envs)]
    )
    model = RecurrentPPO("MultiInputLstmPolicy", envs, verbose=1, ent_coef=args.entropy_coef)
    model.learn(total_timesteps=1_000_000)
    envs.close()


def main():
    q()


if __name__ == "__main__":
    main()
