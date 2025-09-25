import argparse

from gymnasium_env import PokeEnv


def parse_args():
    parser = argparse.ArgumentParser(description="PokeEnv Gym Environment")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
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
    return parser.parse_args()
    

def main():
    print("Hello, World!")
    args = parse_args()
    env = PokeEnv(args)

    episode = 0
    obs, info = env.reset()
    for i in range(100):
        print(f"Step {i}")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # print(f"Episode {episode} total reward: {info['episode']['r']}")
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()
