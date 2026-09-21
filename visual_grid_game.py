# visual_grid_game.py  (SE3062 Practical 02 version)
import copy
import random
import tkinter as tk

# directions (y goes up)
HEADINGS = ['N', 'E', 'S', 'W']
DELTA = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}


def left_of(h):
    return HEADINGS[(HEADINGS.index(h) - 1) % 4]


def right_of(h):
    return HEADINGS[(HEADINGS.index(h) + 1) % 4]


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None, num_traps=5,
                 max_steps=60):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        self.facing = 'E'  # starts facing east
        self.max_steps = max_steps

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Generate adversarial opponents
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        # Hidden hazard: toxic traps (avoid start (0,0), walls, food and opponents)
        self.toxic_traps = set()
        while len(self.toxic_traps) < num_traps:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            trap_pos = (tx, ty)
            if (trap_pos != (0, 0)
                    and trap_pos not in self.walls
                    and trap_pos not in self.food_positions
                    and [tx, ty] not in self.opponents
                    and trap_pos not in self.toxic_traps):
                self.toxic_traps.add(trap_pos)

        self.score = 0
        self.steps = 0
        self.collision = False
        self.stopped = False
        self.visited_cells = {(0, 0)}  # just for drawing the trail

    # helpers
    def _cell_ahead(self):
        dx, dy = DELTA[self.facing]
        return (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

    def _is_blocked(self, cell):
        x, y = cell
        outside = not (0 <= x < self.width and 0 <= y < self.height)
        return outside or cell in self.walls

    # agent only gets these two, no coordinates
    def get_percept(self) -> dict:
        return {
            'wall_ahead': self._is_blocked(self._cell_ahead()),
            'food_here': tuple(self.agent_pos) in self.food_positions,
        }

    def execute_action(self, action: str):
        self.steps += 1

        if action == 'turn_left':
            self.facing = left_of(self.facing)
        elif action == 'turn_right':
            self.facing = right_of(self.facing)
        elif action == 'forward':
            ahead = self._cell_ahead()
            if self._is_blocked(ahead):
                self.score -= 5  # bumped a wall
            else:
                self.agent_pos = list(ahead)
                self.visited_cells.add(ahead)
        elif action == 'suck':
            tuple_pos = tuple(self.agent_pos)
            if tuple_pos in self.food_positions:
                self.food_positions.remove(tuple_pos)
                self.score += 20
        elif action == 'stop':
            self.stopped = True

        # Toxic trap penalty
        if tuple(self.agent_pos) in self.toxic_traps:
            self.score -= 15

        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return (len(self.food_positions) == 0 or self.steps >= self.max_steps
                or self.collision or self.stopped)


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


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None, num_traps=0,
                 max_steps=600, delay_ms=60):
        self.root = root
        self.root.title("SE3062 - Practical 02: Reflex vs Model-Based Agent")
        self.delay_ms = delay_ms

        # same layout every run so both agents get the same world
        self.start_env = VisualGridHuntGame(width=width, height=height, num_food=num_food,
                                            num_opponents=num_opponents, custom_walls=walls,
                                            num_traps=num_traps, max_steps=max_steps)
        self.env = copy.deepcopy(self.start_env)

        # Dynamically calculate cell size so the total canvas fits nicely within a 600x600 window ceiling
        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_reflex = tk.Button(root, text="Run Simple Reflex Agent",
                                    command=lambda: self.run_agent(SimpleReflexAgent(), "Simple Reflex"),
                                    font=("Arial", 12), bg="#000066", fg="white")
        self.btn_reflex.pack(pady=5)

        self.btn_model = tk.Button(root, text="Run Model-Based Agent",
                                   command=lambda: self.run_agent(ModelBasedAgent(), "Model-Based"),
                                   font=("Arial", 12), bg="#166534", fg="white")
        self.btn_model.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                # Only draw text if cell is large enough
                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        # trail of visited cells
        for vx, vy in self.env.visited_cells:
            cx = vx * self.cell_size + self.cell_size / 2
            cy = (self.env.height - 1 - vy) * self.cell_size + self.cell_size / 2
            r = self.cell_size * 0.1
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#93c5fd", outline="")

        for tx, ty in self.env.toxic_traps:
            x1 = tx * self.cell_size
            y1 = (self.env.height - 1 - ty) * self.cell_size
            cx = x1 + self.cell_size / 2
            cy = y1 + self.cell_size / 2
            r = self.cell_size * 0.35
            # Purple diamond
            self.canvas.create_polygon(cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy,
                                       fill="#7e22ce", outline="#581c87")

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066",
                                outline="#1e3a8a")

        # arrow for the direction it's facing
        dx, dy = DELTA[self.env.facing]
        cx = ax * self.cell_size + self.cell_size / 2
        cy = (self.env.height - 1 - ay) * self.cell_size + self.cell_size / 2
        self.canvas.create_line(cx, cy, cx + dx * self.cell_size * 0.35, cy - dy * self.cell_size * 0.35,
                                fill="white", width=3, arrow=tk.LAST)

    def run_agent(self, agent, name):
        self.env = copy.deepcopy(self.start_env)  # fresh copy each run
        self.btn_reflex.config(state="disabled")
        self.btn_model.config(state="disabled")
        is_memoryless = isinstance(agent, SimpleReflexAgent)
        seen_states = set()

        def finish(text):
            self.label.config(text=text)
            self.btn_reflex.config(state="normal")
            self.btn_model.config(state="normal")

        def step():
            # same state twice = it will loop forever
            if is_memoryless:
                state = (tuple(self.env.agent_pos), self.env.facing, frozenset(self.env.food_positions))
                if state in seen_states:
                    finish(f"{name}: INFINITE LOOP detected at step {self.env.steps}! "
                           f"Food left: {len(self.env.food_positions)} | Score: {self.env.score}")
                    return
                seen_states.add(state)

            if not self.env.is_done():
                percept = self.env.get_percept()
                action = agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"{name} | Score: {self.env.score} | Steps: {self.env.steps} | "
                                       f"Action: {action} | Food left: {len(self.env.food_positions)}")
                self.root.after(self.delay_ms, step)
            else:
                if self.env.collision:
                    end_text = f"Collision! Game Over! Final Score: {self.env.score}"
                elif len(self.env.food_positions) == 0:
                    end_text = f"{name}: all food eaten in {self.env.steps} steps! Final Score: {self.env.score}"
                elif self.env.stopped:
                    end_text = (f"{name}: explored everything and stopped. "
                                f"Food left: {len(self.env.food_positions)} | Final Score: {self.env.score}")
                else:
                    end_text = f"{name}: step limit reached. Final Score: {self.env.score}"
                finish(end_text)

        step()


if __name__ == "__main__":
    root = tk.Tk()
    # 12x12, 15 food, no opponents or traps
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0, num_traps=0)
    root.mainloop()