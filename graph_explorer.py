from collections import deque
from dataclasses import dataclass

import numpy as np

from gymnasium_env import Coordinate


USEFUL_ACTIONS = ["A", "B"]


@dataclass
class CoordNeighbors:
    left: Coordinate | None = None
    right: Coordinate | None = None
    up: Coordinate | None = None
    down: Coordinate | None = None


class Graph:
    def __init__(self):
        self._graph: dict[Coordinate, CoordNeighbors] = {}

    def add_node(self, coord: Coordinate) -> None:
        self._graph[coord] = CoordNeighbors()

    def add_edge(
        self, 
        from_coord: Coordinate, 
        to_coord: Coordinate, 
        direction: str,
        replace: bool = True,
    ) -> None:
        if from_coord not in self._graph or to_coord not in self._graph:
            raise ValueError("Both coordinates must be added to the graph before adding an edge.")

        from_coord_neighbors = self._graph[from_coord]
        if direction == "LEFT":
            if replace or from_coord_neighbors.left is None:
                from_coord_neighbors.left = to_coord
        elif direction == "RIGHT":
            if replace or from_coord_neighbors.right is None:
                from_coord_neighbors.right = to_coord
        elif direction == "UP":
            if replace or from_coord_neighbors.up is None:
                from_coord_neighbors.up = to_coord
        elif direction == "DOWN":
            if replace or from_coord_neighbors.down is None:
                from_coord_neighbors.down = to_coord
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def has_node(self, coord: Coordinate) -> bool:
        return coord in self._graph

    def get_neighbors(self, coord: Coordinate) -> CoordNeighbors:
        if coord not in self._graph:
            raise ValueError(f"Coordinate {coord} not found in the graph.")
        return self._graph[coord]
    
    def num_nodes(self) -> int:
        return len(self._graph)

    def num_edges(self) -> int:
        return sum(
            int(n is not None) 
            for neighbors in self._graph.values() 
            for n in (neighbors.left, neighbors.right, neighbors.up, neighbors.down)
        )

    def num_nodes_with_location(self, location: str) -> int:
        return sum(1 for coord in self._graph if coord.location == location)

    def bfs_condition(self, start: Coordinate, condition: callable) -> list[Coordinate]:
        if start not in self._graph:
            raise ValueError(f"Start coordinate {start} not found in the graph.")

        visited = {start}
        parents = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if condition(current):
                reverse_path = []
                while current is not None:
                    reverse_path.append(current)
                    current = parents[current]
                return reverse_path[::-1]
            
            for neighbor in (
                self._graph[current].left,
                self._graph[current].right,
                self._graph[current].up,
                self._graph[current].down
            ):
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    parents[neighbor] = current

        return []

    def trajectory_to_directions(self, trajectory: list[Coordinate]) -> list[str]:
        directions = []
        for i in range(len(trajectory) - 1):
            from_coord = trajectory[i]
            to_coord = trajectory[i + 1]

            neighbors = self.get_neighbors(from_coord)
            if neighbors.left == to_coord:
                directions.append("LEFT")
            elif neighbors.right == to_coord:
                directions.append("RIGHT")
            elif neighbors.up == to_coord:
                directions.append("UP")
            elif neighbors.down == to_coord:
                directions.append("DOWN")
            else:
                raise ValueError(f"No edge from {from_coord} to {to_coord}.")

        return directions
    
    def unexplored(self, coordinate: Coordinate) -> bool:
        if coordinate not in self._graph:
            raise ValueError(f"Coordinate {coordinate} not found in the graph.")

        neighbors = self.get_neighbors(coordinate)
        return any(n is None for n in (neighbors.left, neighbors.right, neighbors.up, neighbors.down))

    def make_frontier_unexplored(self) -> None:
        for coord, neighbors in self._graph.items():
            for direction in ["left", "right", "up", "down"]:
                neighbor = getattr(neighbors, direction)
                if neighbor == coord:
                    setattr(neighbors, direction, None)

    def make_frontier_unexplored_for_location(self, location: str) -> None:
        for coord, neighbors in self._graph.items():
            if coord.loc != location:
                continue
            for direction in ["left", "right", "up", "down"]:
                neighbor = getattr(neighbors, direction)
                if neighbor == coord:
                    setattr(neighbors, direction, None)


class GraphExplorer:
    def __init__(self):
        self._graph = Graph()

    def add_coord_to_graph(self, coord: Coordinate) -> None:
        if not self._graph.has_node(coord):
            self._graph.add_node(coord)

    def add_edge_to_graph(
        self, 
        coord1: Coordinate, 
        coord2: Coordinate, 
        direction: str, 
        replace: bool = True,
    ) -> None:
        if not self._graph.has_node(coord1):
            self._graph.add_node(coord1)
        if not self._graph.has_node(coord2):
            self._graph.add_node(coord2)

        self._graph.add_edge(coord1, coord2, direction, replace=replace)

    def get_explore_direction(self, coord: Coordinate) -> int:
        if not self._graph.has_node(coord):
            raise ValueError(f"Coordinate {coord} not found in the graph.")

        neighbors = self._graph.get_neighbors(coord)
        directions = {
            "LEFT": neighbors.left,
            "RIGHT": neighbors.right,
            "UP": neighbors.up,
            "DOWN": neighbors.down
        }
        for direction, neighbor in directions.items():
            if neighbor is None:
                return direction

        # Find closest node with unvisited directions
        trajectory = self._graph.bfs_condition(coord, lambda c: self._graph.unexplored(c))

        # if all nodes are explored: make all fronteir nodes unexplored
        if not trajectory:
            # self._graph.make_frontier_unexplored()
            self._graph.make_frontier_unexplored_for_location(coord.loc)

        directions = self._graph.trajectory_to_directions(trajectory)

        if directions:
            return directions[0]
        
        # give up and take random direction
        return np.random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    def get_action(self, coord: Coordinate) -> int:
        if np.random.rand() < 0.5:
            action = self.get_explore_direction(coord)
        else:
            action = np.random.choice(USEFUL_ACTIONS)

        return action

    def fill_graph_walkable(self, map: list, player_coord: Coordinate) -> None:
        map = np.array(map)

        shift_x = np.inf
        shift_y = np.inf
        for y, line in enumerate(map):
            for x, char in enumerate(line):
                if char == "P":
                    shift_x = player_coord.x - x
                    shift_y = player_coord.y - y
                    break

        if shift_x == np.inf or shift_y == np.inf:
            print("Player not found in map.")
            return
        
        walkable = [".", "~"]
        for y in range(map.shape[0]):
            for x in range(map.shape[1]):
                if map[y, x] == ".":
                    coord = Coordinate(x=x + shift_x, y=y + shift_y, loc=player_coord.loc)
                    self.add_coord_to_graph(coord)
                    if x - 1 >= 0:
                        if map[y, x - 1] in walkable:
                            neighbor = Coordinate(x=x - 1 + shift_x, y=y + shift_y, loc=player_coord.loc)
                        elif map[y, x - 1] == "#":
                            neighbor = coord
                        else:
                            continue
                        self.add_edge_to_graph(coord, neighbor, "LEFT", replace=False)
                    if x + 1 < map.shape[1]:
                        if map[y, x + 1] in walkable:
                            neighbor = Coordinate(x=x + 1 + shift_x, y=y + shift_y, loc=player_coord.loc)
                        elif map[y, x + 1] == "#":
                            neighbor = coord
                        else:
                            continue
                        self.add_edge_to_graph(coord, neighbor, "RIGHT", replace=False)
                    if y - 1 >= 0:
                        if map[y - 1, x] in walkable:
                            neighbor = Coordinate(x=x + shift_x, y=y - 1 + shift_y, loc=player_coord.loc)
                        elif map[y - 1, x] == "#":
                            neighbor = coord
                        else:
                            continue
                        self.add_edge_to_graph(coord, neighbor, "UP", replace=False)
                    if y + 1 < map.shape[0]:
                        if map[y + 1, x] in walkable:
                            neighbor = Coordinate(x=x + shift_x, y=y + 1 + shift_y, loc=player_coord.loc)
                        elif map[y + 1, x] == "#":
                            neighbor = coord
                        else:
                            continue
                        self.add_edge_to_graph(coord, neighbor, "DOWN", replace=False)
                    