# visual_grid_game.py
import copy
import random
import tkinter as tk

from agent import (SimpleReflexAgent, ModelBasedAgent, SearchAgent,
                   DELTA, MOVES, left_of, right_of)

# which way the agent faces after an absolute move
FACING = {'Up': 'N', 'Right': 'E', 'Down': 'S', 'Left': 'W'}


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

    def get_percept(self) -> dict:
        return {
            'wall_ahead': self._is_blocked(self._cell_ahead()),
            'food_here': tuple(self.agent_pos) in self.food_positions,
            'agent_pos': list(self.agent_pos),  # start cell for the search
            'grid_size': (self.width, self.height),
            'walls': list(self.walls),
            'all_food': list(self.food_positions),
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
        elif action in MOVES:  # Up / Down / Left / Right (search agent)
            dx, dy = MOVES[action]
            self.facing = FACING[action]
            new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            if self._is_blocked(new_pos):
                self.score -= 5  # bumped a wall
            else:
                self.agent_pos = list(new_pos)
                self.visited_cells.add(new_pos)
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


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None, num_traps=0,
                 max_steps=1000, delay_ms=30):
        self.root = root
        self.root.title("Practical 04: A* Search")
        self.delay_ms = delay_ms

        # same layout every run so every agent gets the same world
        self.agent = None
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

        bar = tk.Frame(root)
        bar.pack(pady=5)
        self.buttons = []
        for text, color, cmd in [
            ("BFS", "#000066", lambda: self.run_search('BFS')),
            ("DFS", "#b45309", lambda: self.run_search('DFS')),
            ("UCS", "#7e22ce", lambda: self.run_search('UCS')),
            ("A*", "#0f766e", lambda: self.run_search('AStar')),
            ("Reflex", "#475569", lambda: self.run_agent(SimpleReflexAgent(), "Simple Reflex")),
            ("Model-Based", "#166534", lambda: self.run_agent(ModelBasedAgent(), "Model-Based")),
        ]:
            btn = tk.Button(bar, text=text, command=cmd, font=("Arial", 12), bg=color, fg="white")
            btn.pack(side="left", padx=4)
            self.buttons.append(btn)

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

        # rest of the plan (search agent only)
        if isinstance(self.agent, SearchAgent):
            px, py = self.env.agent_pos
            for a in self.agent.plan:
                if a in MOVES:
                    px, py = px + MOVES[a][0], py + MOVES[a][1]
                    cx = px * self.cell_size + self.cell_size / 2
                    cy = (self.env.height - 1 - py) * self.cell_size + self.cell_size / 2
                    r = self.cell_size * 0.12
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#fb923c", outline="")

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

    def run_search(self, algo):
        agent = SearchAgent()
        agent.active_algo = algo
        self.run_agent(agent, algo)

    def run_agent(self, agent, name):
        self.env = copy.deepcopy(self.start_env)  # fresh copy each run
        self.agent = agent
        for b in self.buttons:
            b.config(state="disabled")
        is_memoryless = isinstance(agent, SimpleReflexAgent)
        seen_states = set()

        def nodes():  # how many nodes the search has expanded (search agents only)
            return f" | Nodes expanded: {agent.expanded}" if isinstance(agent, SearchAgent) else ""

        def finish(text):
            self.label.config(text=text)
            for b in self.buttons:
                b.config(state="normal")

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
                                       f"Action: {action} | Food left: {len(self.env.food_positions)}" + nodes())
                self.root.after(self.delay_ms, step)
            else:
                if self.env.collision:
                    end_text = f"Collision! Game Over! Final Score: {self.env.score}"
                elif len(self.env.food_positions) == 0:
                    end_text = (f"{name}: all food eaten in {self.env.steps} steps! "
                                f"Final Score: {self.env.score}" + nodes())
                elif self.env.stopped:
                    end_text = (f"{name}: stopped, nothing left to reach. "
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