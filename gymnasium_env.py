import argparse
import base64
from dataclasses import dataclass
import io
import pygame
import requests
import subprocess
import sys
import time

import cv2
import gymnasium as gym
import numpy as np
from PIL import Image


ACTION_INT_TO_STR = {
    0: "A",
    1: "B",
    2: "UP",
    3: "DOWN",
    4: "LEFT",
    5: "RIGHT",
    # 6: "SELECT",
    # 7: "START",
}

MOVEMENT_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}

MAX_MOVEMENT_ACTION_SENDS = 2


def start_server(args, port):
    python_exe = sys.executable
    if args.dev:
        server_file = "server.app-dev"
    else:
        server_file = "server.app"

    server_cmd = [python_exe, "-m", server_file, "--port", str(port)]

    if args.dev:
        server_cmd.append("--no-ocr")

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


@dataclass
class Coordinate:
    x: int
    y: int
    loc: str


class PokeEnv(gym.Env):
    def __init__(self, args, seed):
        super().__init__()
        self.args = args
        self._port = args.port + 3 * seed
        self._server_url = f"http://localhost:{self._port}"
        self._server = None

        self._pygame = args.pygame
        # used if not headless
        self._pygame_screen = None

        self.action_space = gym.spaces.Discrete(len(ACTION_INT_TO_STR))
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(84, 84),
                dtype=np.uint8,
            ),
        })

        if args.dev:
            self.per_step_sleep = 0.01
        else:
            self.per_step_sleep = 1

        self._seen_locations = set()
        self._seen_dialog = set()

        self.time_limit = args.timelimit
        self._steps = 0

        self._total_reward = 0
        self._last_seen_coord = None

    def reset(self, seed=None) -> tuple[dict, dict]:
        self._steps = 0
        self._total_reward = 0
        self._seen_dialog.clear()
        self._seen_locations.clear()

        self.close()
        self._server = start_server(self.args, self._port)
        if self._server is None:
            raise RuntimeError("Failed to restart server")

        if self._pygame:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((480, 320))
            pygame.display.set_caption("PokeEnv")
            print("Pygame window initialized.")

        state = self._get_state()
        obs = self._get_obs(state)
        coordinate = self._get_coord(state)
        info = {"coord": coordinate}
        self._last_seen_coord = coordinate

        if self._pygame:
            self._update_pygame_window(obs["image"])

        self._seen_locations.add(state["player"]["location"])

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        action = ACTION_INT_TO_STR.get(action, None)
        if action is None:
            raise ValueError(f"Invalid action: {action}")

        action_send_count = 0
        while True:
            try:
                response = requests.post(
                    f"{self._server_url}/action", 
                    json={"buttons": [action]},
                    timeout=5,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to send action: {response.text}")
                action_send_count += 1
            except requests.RequestException as e:
                raise RuntimeError(f"Request failed: {e}") from e

            # only one action for non-movement
            if not action in MOVEMENT_ACTIONS:
                break
            
            time.sleep(self.per_step_sleep)       
            state = self._get_state()
            current_coordinate = self._get_coord(state)
            if current_coordinate != self._last_seen_coord:
                break
            if action_send_count >= MAX_MOVEMENT_ACTION_SENDS:
                break

        time.sleep(self.per_step_sleep)       

        next_state = self._get_state()
        reward = 0

        dialog = str(next_state["game"]["dialog_text"])
        if dialog not in self._seen_dialog:
            reward += 1
            self._seen_dialog.add(dialog)

        next_obs = self._get_obs(next_state)
        next_coordinate = self._get_coord(next_state)
        info = {"coord": next_coordinate}
        self._last_seen_coord = next_coordinate

        location = next_coordinate.loc
        if location not in self._seen_locations:
            reward += 1
            self._seen_locations.add(location)

        if self._steps >= self.time_limit:
            truncated = True
        else:
            truncated = False
        terminated = False

        if self._pygame:
            self._update_pygame_window(next_obs["image"])

        self._steps += 1
        self._total_reward += reward
        if self._steps % 20 == 0:
            with open("progress.txt", "a") as f:
                f.write(f"{self._steps}: {self._total_reward}\n")

        return next_obs, reward, terminated, truncated, info

    def close(self):
        if self._server:
            end_server(self._server_url)
            self._server.terminate()
            self._server.wait()
            print("Server process terminated.")
        if self._pygame:
            pygame.quit()
            self._pygame_screen = None
            print("Pygame window closed.")

    def _get_state(self) -> dict:
        while True:
            try:
                request = requests.get(f"{self._server_url}/state")
                if request.status_code != 200:
                    print(f"Failed to get observation: {request.text}, retrying...")
                    continue
                    # raise RuntimeError(f"Failed to get observation: {request.text}")
                return dict(request.json())
            except requests.RequestException as e:
                raise RuntimeError(f"Request failed: {e}") from e

    def _get_obs(self, state: dict) -> dict:
        obs = {}

        if "visual" not in state or "screenshot_base64" not in state["visual"]:
            raise RuntimeError("No visual data in state")
        img_data = state["visual"]["screenshot_base64"]
        image = base64_to_numpy(img_data)
        image = cv2.resize(image, (84, 84))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        obs["image"] = image

        return obs

    def _get_coord(self, state: dict) -> Coordinate:
        player = state.get("player", {})
        pos = player.get("position", {})
        x = pos.get("x", 0)
        y = pos.get("y", 0)
        loc = player.get("location", "unknown")
        return Coordinate(x=x, y=y, loc=loc)

    def _update_pygame_window(self, frame: np.ndarray) -> None:
        if self._pygame_screen is None:
            return
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
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
