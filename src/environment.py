from minigrid.minigrid_env import MiniGridEnv  
from minigrid.core.grid import Grid  
from minigrid.core.world_object import Goal, Wall, Ball
from minigrid.core.mission import MissionSpace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional


class Seeker(Ball):
    """A red ball that acts as a seeker. Marked as overlappable so the agent
    and other seekers can share cells (collision is checked manually)."""

    def __init__(self):
        super().__init__('red')

    def can_overlap(self):
        return True


class RLEnvironment(MiniGridEnv):

    def __init__(self, grid_size=16, max_steps=200, n_seekers=1, render_mode=None):
        self.n_seekers = n_seekers
        self.seeker_positions = []
        self.seeker_objects = []

        self.render_mode = render_mode

        mission_space = MissionSpace(mission_func=lambda: "reach the goal")

        self.vision_radius = 5  # cells illuminated around the runner

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            render_mode=render_mode,
            highlight=True,          # we override get_full_render to draw a circular mask
        )

        self.action_space = spaces.Discrete(8)

        self.action_map = {
            0: np.array([0, 1]),   # Right
            1: np.array([-1, 0]),  # Up
            2: np.array([0, -1]),  # Left
            3: np.array([1, 0]),   # Down
            4: np.array([1, 1]),   # Down-Right
            5: np.array([-1, 1]),  # Up-Right
            6: np.array([1, -1]),  # Down-Left
            7: np.array([-1, -1]), # Up-Left
        }

        self.observation_space = spaces.Dict({
            'agent': spaces.Box(0, grid_size-1, shape=(2,), dtype=np.int32),
            'goal': spaces.Box(0, grid_size-1, shape=(2,), dtype=np.int32),
            'seekers': spaces.Box(0, grid_size-1, shape=(n_seekers, 2), dtype=np.int32),
        })

    def get_full_render(self, highlight: bool, tile_size: int):
        """Replace the default directional FOV cone with a circular vision
        radius centred on the runner."""
        highlight_mask = np.zeros((self.width, self.height), dtype=bool)
        if highlight:
            ax, ay = self.agent_pos
            r2 = self.vision_radius ** 2
            for x in range(self.width):
                for y in range(self.height):
                    if (x - ax) ** 2 + (y - ay) ** 2 <= r2:
                        highlight_mask[x, y] = True

        return self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # All spawnable positions (interior cells, excluding border)
        # We track used positions to avoid overlaps
        used = set()

        def rand_pos():
            """Pick a random interior cell not already occupied."""
            while True:
                x = int(self.np_random.integers(1, width - 1))
                y = int(self.np_random.integers(1, height - 1))
                if (x, y) not in used:
                    used.add((x, y))
                    return (x, y)

        # Randomize goal position
        self.goal_pos = rand_pos()
        self.put_obj(Goal(), *self.goal_pos)

        # Randomize agent (runner) position
        agent_pos = rand_pos()
        self.agent_pos = agent_pos
        self.agent_dir = int(self.np_random.integers(0, 4))

        # Randomize seeker positions and place them in the grid
        self.seeker_positions = []
        self.seeker_objects = []
        for _ in range(self.n_seekers):
            pos = rand_pos()
            seeker = Seeker()
            self.seeker_objects.append(seeker)
            self.seeker_positions.append(list(pos))
            self.grid.set(*pos, seeker)


    def gen_obs(self):
        return {
            'agent': np.array(self.agent_pos, dtype=np.int32),
            'goal': np.array(self.goal_pos, dtype=np.int32),
            'seekers': np.array(self.seeker_positions, dtype=np.int32),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed)
        return obs, info

    def _move_seekers(self):
        """Move each seeker one step along the shortest (8-directional) path to
        the agent, updating both the position list and the grid in-place."""
        agent = np.array(self.agent_pos)
        for idx, pos in enumerate(self.seeker_positions):
            seeker_pos = np.array(pos)
            diff = agent - seeker_pos

            # Sign gives the unit step in each axis (-1, 0, or +1)
            step = np.sign(diff)
            new_pos = seeker_pos + step

            new_x = int(np.clip(new_pos[0], 0, self.width - 1))
            new_y = int(np.clip(new_pos[1], 0, self.height - 1))
            old_x, old_y = int(seeker_pos[0]), int(seeker_pos[1])

            # Nothing to do if position hasn't changed
            if (old_x, old_y) == (new_x, new_y):
                continue

            # Only move to passable cells; explicitly skip the goal cell so
            # seekers never overwrite it (Goal.can_overlap() is True in MiniGrid)
            cell = self.grid.get(new_x, new_y)
            target_is_goal = (new_x, new_y) == tuple(self.goal_pos)
            if not target_is_goal and (cell is None or cell.can_overlap()):
                # Update grid: restore old cell (put back goal if seeker was on it),
                # then place seeker in new position
                old_cell_was_goal = (old_x, old_y) == tuple(self.goal_pos)
                self.grid.set(old_x, old_y, Goal() if old_cell_was_goal else None)
                self.grid.set(new_x, new_y, self.seeker_objects[idx])
                self.seeker_positions[idx] = [new_x, new_y]

    def _seeker_caught_agent(self):
        """Return True if any seeker occupies the same cell as the agent."""
        for pos in self.seeker_positions:
            if tuple(pos) == tuple(self.agent_pos):
                return True
        return False

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = int(action.item())

        # Get movement direction
        direction = self.action_map[action]

        # Calculate new position
        new_pos = (
            self.agent_pos[0] + direction[0],
            self.agent_pos[1] + direction[1]
        )

        # Clip to grid boundaries
        new_pos = (
            int(np.clip(new_pos[0], 0, self.width - 1)),
            int(np.clip(new_pos[1], 0, self.height - 1))
        )

        prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))

        # Check if cell is passable (seekers have can_overlap=True, so agent can step on them)
        cell = self.grid.get(*new_pos)
        if cell is None or cell.can_overlap():
            self.agent_pos = new_pos
        
        # Increment step counter
        self.step_count += 1

        # Move seekers toward the agent after the agent moves
        self._move_seekers()

        # Check if a seeker caught the agent
        if self._seeker_caught_agent():
            observation = self.gen_obs()
            return observation, -100.0, True, False, {}

        # Check if reached goal
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 100.0
            terminated = True
        else:
            curr_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
            dist_reward = (prev_dist - curr_dist) * 0.5

            # Proximity bonus using inverse distance so it spikes sharply when
            # adjacent to the goal (~8.0 at dist=1), overwhelming any seeker-fear
            # and preventing oscillation. Falls to ~0 beyond 5 cells.
            if curr_dist < 5.0:
                proximity_bonus = min(8.0, 8.0 / max(curr_dist, 0.5)) - 1.6
            else:
                proximity_bonus = 0.0

            reward = -0.1 + dist_reward + proximity_bonus
            terminated = False

        # Check if max steps reached
        truncated = self.step_count >= self.max_steps

        observation = self.gen_obs()
        return observation, reward, terminated, truncated, {}