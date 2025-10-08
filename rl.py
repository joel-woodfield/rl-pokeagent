import argparse

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gymnasium_env import PokeEnv


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
        default=2_000,
        help="Number of steps per episode (default: 100,000)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0,
        help="Entropy coefficient for PPO (default: 0)",
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


def ppo():
    args = parse_args()
    envs = DummyVecEnv(
        [lambda: PokeEnv(args, seed=i) for i in range(args.num_envs)]
    )
    model = RecurrentPPO("MultiInputLstmPolicy", envs, verbose=1, ent_coef=args.entropy_coef)
    model.learn(total_timesteps=1_000_000)
    envs.close()


def main():
    ppo()


if __name__ == "__main__":
    main()
