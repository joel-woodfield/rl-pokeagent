import argparse
import base64
import io
import pygame
import requests
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
from PIL import Image


ACTION_INT_TO_STR = {
    0: "A",
    1: "B",
    2: "SELECT",
    3: "START",
    4: "UP",
    5: "DOWN",
    6: "LEFT",
    7: "RIGHT",
}


def start_server(args):
    python_exe = sys.executable
    server_cmd = [python_exe, "-m", "server.app", "--port", str(args.port)]

    if args.record:
        server_cmd.append("--record")

    try:
        server_process = subprocess.Popen(
            server_cmd,
            universal_newlines=True,
            bufsize=1,
        )
        print("Server started successfully. Wating 3s")
        time.sleep(3)

        return server_process

    except Exception as e:
        print(f"Failed to start server: {e}")
        return None


def end_server(url: str):
    try:
        response = requests.post(f"{url}/stop")
        if response.status_code != 200:
            print(f"Failed to terminate server: {response.text}")
        else:
            print("Server terminated successfully.")
    except requests.RequestException as e:
        print(f"Request to terminate server failed: {e}")


def base64_to_numpy(b64_string: str) -> np.ndarray:
    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


class PokeEnv(gym.Env):
    PER_STEP_SLEEP = 1

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._server_url = f"http://localhost:{args.port}"
        self._server = None

        self._pygame = args.pygame
        # used if not headless
        self._pygame_screen = None

        self.action_space = gym.spaces.Discrete(len(ACTION_INT_TO_STR))

    def reset(self) -> tuple[dict, dict]:
        self.close()
        self._server = start_server(self.args)
        if self._server is None:
            raise RuntimeError("Failed to restart server")

        if self._pygame:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((480, 320))
            pygame.display.set_caption("PokeEnv")
            print("Pygame window initialized.")

        state = self._get_state()
        obs = self._get_obs(state)
        info = {}

        if self._pygame:
            self._update_pygame_window(obs["image"])

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        action = ACTION_INT_TO_STR.get(action, None)
        if action is None:
            raise ValueError(f"Invalid action: {action}")
        try:
            response = requests.post(
                f"{self._server_url}/action", 
                json={"buttons": [action]},
                timeout=5,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Failed to send action: {response.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {e}") from e

        time.sleep(self.PER_STEP_SLEEP)       

        next_state = self._get_state()
        next_obs = self._get_obs(next_state)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self._pygame:
            self._update_pygame_window(next_obs["image"])

        return next_obs, reward, terminated, truncated, info

    def close(self):
        if self._server:
            end_server(self._server_url)
            self._server.terminate()
            self._server.wait()
            print("Server process terminated.")

    def _get_state(self) -> dict:
        try:
            request = requests.get(f"{self._server_url}/state")
            if request.status_code != 200:
                raise RuntimeError(f"Failed to get observation: {request.text}")
            return dict(request.json())
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {e}") from e

    def _get_obs(self, state: dict) -> dict:
        obs = {}

        if "visual" not in state or "screenshot_base64" not in state["visual"]:
            raise RuntimeError("No visual data in state")
        img_data = state["visual"]["screenshot_base64"]
        image = base64_to_numpy(img_data)
        obs["image"] = image

        if "player" not in state or "position" not in state["player"]:
            raise RuntimeError("No position data in state")
        position = state["player"]["position"]
        if "x" not in position or "y" not in position:
            raise RuntimeError("Incomplete position data in state")
        obs["position"] = np.array([position["x"], position["y"]], dtype=np.float32)

        return obs

    def _update_pygame_window(self, frame: np.ndarray) -> None:
        if self._pygame_screen is None:
            return
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, self._pygame_screen.get_size())
        self._pygame_screen.blit(surface, (0, 0))
        pygame.display.flip()


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


def test():
    args = parse_args()
    env = PokeEnv(args)
    try:
        obs, info = env.reset()
        print(obs)
    except RuntimeError as e:
        print(f"Error during environment reset: {e}")
        env.close()

    time.sleep(10)
    print(f"Stepping")
    try:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(obs)
    except RuntimeError as e:
        print(f"Error during environment step: {e}")
        env.close()

    env.close()


if __name__ == "__main__":
    test()
    