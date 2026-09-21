# agent.py
import heapq
import math
import random
from collections import deque

# directions (y goes up)
HEADINGS = ['N', 'E', 'S', 'W']
DELTA = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}

# absolute moves (the search agent uses these)
MOVES = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}


def left_of(h):
    return HEADINGS[(HEADINGS.index(h) - 1) % 4]


def right_of(h):
    return HEADINGS[(HEADINGS.index(h) + 1) % 4]


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


# simple reflex agent (no __init__, no memory)
class SimpleReflexAgent:
    """Only looks at the current percept."""

    def sense_and_act(self, percept):
        if percept['food_here']:  # food here -> suck
            return 'suck'
        if percept['wall_ahead']:  # wall ahead -> turn left
            return 'turn_left'
        return 'forward'  # otherwise keep going


# model-based agent (remembers where it's been)
class ModelBasedAgent:
    """Tracks its own x, y since it can't see the real coordinates."""

    def __init__(self):
        self.x, self.y = 0, 0  # my own position, start is (0, 0)
        self.heading = 'N'  # my own heading
        self.visited = {(0, 0)}  # cells I've been on
        self.walls = set()  # walls I've seen
        self.last_action = None  # what I did last step

    # helpers
    def _cell(self, heading):
        dx, dy = DELTA[heading]
        return (self.x + dx, self.y + dy)

    def _fresh(self, cell):
        """Not visited and not a wall."""
        return cell not in self.visited and cell not in self.walls

    # update memory
    def _update_state(self, percept):
        # move/turn based on what I did last step
        if self.last_action == 'turn_left':
            self.heading = left_of(self.heading)
        elif self.last_action == 'turn_right':
            self.heading = right_of(self.heading)
        elif self.last_action == 'forward':
            # only goes forward when the way is clear, so it worked
            self.x, self.y = self._cell(self.heading)
        self.visited.add((self.x, self.y))

        # note the wall if there is one
        if percept['wall_ahead']:
            self.walls.add(self._cell(self.heading))

    # rules
    def sense_and_act(self, percept):
        self._update_state(percept)  # memory first

        if percept['food_here']:  # eat if there's food
            action = 'suck'
        else:
            action = self._choose_move(percept)
        self.last_action = action  # remember for next step
        return action

    def _choose_move(self, percept):
        ahead = self._cell(self.heading)
        left = self._cell(left_of(self.heading))
        right = self._cell(right_of(self.heading))
        behind = self._cell(left_of(left_of(self.heading)))
        ahead_free = not percept['wall_ahead']

        if ahead_free and self._fresh(ahead):  # new cell ahead
            return 'forward'
        if self._fresh(left):  # new cell on the left
            return 'turn_left'
        if self._fresh(right):  # new cell on the right
            return 'turn_right'
        if self._fresh(behind):  # new cell behind
            return 'turn_left'
        return self._head_to_nearest_frontier(ahead_free)  # all nearby cells done, go find a new one

    def _head_to_nearest_frontier(self, ahead_free):
        """BFS through visited cells to the nearest cell with an unexplored neighbour."""
        start = (self.x, self.y)
        queue = [(start, None)]  # (cell, first step from start)
        seen = {start}
        while queue:
            (cx, cy), first = queue.pop(0)
            if first is not None and any(
                    self._fresh((cx + DELTA[h][0], cy + DELTA[h][1])) for h in HEADINGS):
                return self._turn_or_step(first, ahead_free)
            for h in HEADINGS:
                nxt = (cx + DELTA[h][0], cy + DELTA[h][1])
                if nxt in self.visited and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, first or h))
        return 'stop'  # nothing left

    def _turn_or_step(self, direction, ahead_free):
        if direction == self.heading:
            return 'forward' if ahead_free else 'turn_left'
        if direction == right_of(self.heading):
            return 'turn_right'
        return 'turn_left'  # left, or behind (turn twice)


# goal-based agent: plans a path to the food first, then follows it
class SearchAgent:
    def __init__(self):
        self.plan = []  # actions still left to do
        self.active_algo = 'BFS'  # 'BFS', 'DFS', 'UCS' or 'AStar'
        self.heuristic_type = 'manhattan'  # used by A*: 'manhattan' or 'euclidean'
        self.expanded = 0  # nodes expanded so far (to compare the algorithms)

    def sense_and_act(self, percept):
        if not self.plan:  # no plan yet, make one
            start = tuple(percept['agent_pos'])
            walls, grid_size = percept['walls'], percept['grid_size']
            # closest food first
            foods = sorted((tuple(f) for f in percept['all_food']),
                           key=lambda f: (self.manhattan_distance(start, f), f))
            for goal_pos in foods:
                if self.active_algo == 'BFS':
                    path = self.bfs_search(start, goal_pos, walls, grid_size)
                elif self.active_algo == 'DFS':
                    path = self.dfs_search(start, goal_pos, walls, grid_size)
                elif self.active_algo == 'UCS':
                    path = self.ucs_search(start, goal_pos, walls, grid_size)
                elif self.active_algo == 'AStar':
                    path = self.astar_search(start, goal_pos, walls, grid_size, self.heuristic_type)
                else:
                    raise ValueError("unknown algorithm: " + str(self.active_algo))
                if path is not None:
                    self.plan = path + ['suck']  # walk there, then eat
                    break
        if not self.plan:
            return 'stop'  # no reachable food left
        return self.plan.pop(0)

    # heuristics: h(n) = estimated cost from pos to goal
    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    # next cells the agent can move to (inside the grid, not a wall)
    def _neighbors(self, pos, walls, grid_size):
        width, height = grid_size
        for action, (dx, dy) in MOVES.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in walls:
                yield action, nxt

    # BFS: queue (FIFO), shallowest node first
    def bfs_search(self, start, goal, walls, grid_size):
        start, goal = tuple(start), tuple(goal)
        walls = {tuple(w) for w in walls}
        frontier = deque([(start, [])])  # (cell, actions so far)
        reached = {start}
        while frontier:
            pos, path = frontier.popleft()
            if pos == goal:
                return path
            self.expanded += 1
            for action, nxt in self._neighbors(pos, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return None  # goal can't be reached

    # DFS: stack (LIFO), deepest node first
    def dfs_search(self, start, goal, walls, grid_size):
        start, goal = tuple(start), tuple(goal)
        walls = {tuple(w) for w in walls}
        frontier = [(start, [])]
        reached = set()
        while frontier:
            pos, path = frontier.pop()
            if pos == goal:
                return path
            if pos in reached:
                continue
            reached.add(pos)
            self.expanded += 1
            for action, nxt in self._neighbors(pos, walls, grid_size):
                if nxt not in reached:
                    frontier.append((nxt, path + [action]))
        return None

    # UCS: priority queue ordered by total path cost g(n)
    def ucs_search(self, start, goal, walls, grid_size, step_cost=None):
        start, goal = tuple(start), tuple(goal)
        walls = {tuple(w) for w in walls}
        frontier = [(0, 0, start, [])]  # (g, tie-breaker, cell, actions so far)
        reached = set()
        count = 1
        while frontier:
            g, _, pos, path = heapq.heappop(frontier)
            if pos == goal:
                return path
            if pos in reached:
                continue
            reached.add(pos)
            self.expanded += 1
            for action, nxt in self._neighbors(pos, walls, grid_size):
                if nxt not in reached:
                    cost = 1 if step_cost is None else step_cost(pos, nxt)  # every move costs 1 by default
                    heapq.heappush(frontier, (g + cost, count, nxt, path + [action]))
                    count += 1
        return None

    # A*: priority queue ordered by f(n) = g(n) + h(n)
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)
        walls = {tuple(w) for w in walls}
        if heuristic_type == 'manhattan':
            h = self.manhattan_distance
        elif heuristic_type == 'euclidean':
            h = self.euclidean_distance
        else:
            raise ValueError("unknown heuristic: " + str(heuristic_type))

        # (f_cost, g_cost, current_pos, path_taken)
        frontier = [(h(start_pos, goal_pos), 0, start_pos, [])]
        reached_states = set()
        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)
            if current_pos == goal_pos:
                return path_taken
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)
            self.expanded += 1
            for action, nxt in self._neighbors(current_pos, walls, grid_size):
                if nxt not in reached_states:
                    g_new = g_cost + 1
                    f_new = g_new + h(nxt, goal_pos)
                    heapq.heappush(frontier, (f_new, g_new, nxt, path_taken + [action]))
        return None


if __name__ == "__main__":
    # testing checkpoint: start (0, 0), goal (3, 4)
    agent = SearchAgent()
    print("Manhattan:", agent.manhattan_distance((0, 0), (3, 4)))  # should be 7
    print("Euclidean:", agent.euclidean_distance((0, 0), (3, 4)))  # should be 5.0