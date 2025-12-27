# ===============================================================
# SECTION 1 — IMPORTS AND CONFIGURATION
# ===============================================================

import tkinter as tk
from tkinter import messagebox
from collections import deque
import random
import math
import time
import heapq
from tkinter import ttk
import os


GRID_SIZE = 20
TILE_SIZE = 24
INITIAL_BATTERY = 500

COSTS = {
    "flat": 5,
    "sandy": 10,
    "rocky": 1000,
    "recharge station": 0,
    "start": 0,
    "goal": 0,
    "cliff": math.inf,  # Impassable
    "trap": 20,         # Costly to enter
    "hazardous": 25     # Costly to enter
}
COLORS = {
    "flat": "light green",
    "sandy": "khaki",
    "rocky": "gray",
    "recharge station": "blue",
    "start": "green",
    "goal": "red",
    "path": "cyan",
    "current": "orange",
    "cliff": "black",
    "trap": "purple",
    "hazardous": "yellow"
}
HEURISTIC_COLORS = {
    "Manhattan": "blue",
    "Euclidean": "purple",
    "Squared Manhattan": "orange",
    "Logarithmic": "darkgreen"
}

# Cliffs and rocks are impassable.
# Traps and Hazardous terrain are *passable* but dangerous.
IMPASSABLE = ["rocky", "cliff"]

# ===============================================================
# SECTION 2 — HEURISTIC FUNCTIONS FOR A*
# ==================

def manhattan_heuristic(a, b):
    # Ideal for 4-directional movement
    return 5 * (abs(a[0] - b[0]) + abs(a[1] - b[1]))

def euclidean_heuristic(a, b):
    return 5 * math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def squared_manhattan_heuristic(a, b):
    """Quadratic version of Manhattan — strongly goal-focused."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return 5 * ((dx + dy) ** 2)

def log_heuristic(a, b):
    """Logarithmic growth — encourages exploration."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return 5 * math.log2(1 + dx + dy)

HEURISTICS = {
    "Manhattan": manhattan_heuristic,
    "Euclidean": euclidean_heuristic,
    "Squared Manhattan": squared_manhattan_heuristic,
    "Logarithmic": log_heuristic
}

# ===============================================================
# SECTION 3 — ROVER AGENT CLASS
# ===============================================================

class RoverAgent:
    def __init__(self, start, goal, grid_size, terrain_map):
        self.location = start
        self.start = start
        self.goal = goal
        self.grid_size = grid_size
        self.terrain_map = terrain_map
        self.battery = INITIAL_BATTERY
        self.path = deque()
        #self.nodes_expanded = 0
        self.path_cost = 0
        self.messages = deque(maxlen=5)
        
        ##
        self.last_safe_location = start
        self.known_hazards = set() 
        # --- END NEW ---

    def find_all_recharge_stations(self):
        """Returns a list of all recharge station coordinates."""
        return [
            (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
            if self.terrain_map[r][c] == "recharge station"
        ]

    def update_location(self, new_loc):
        self.last_safe_location = self.location ##
        self.location = new_loc
        terrain = self.terrain_map[new_loc[0]][new_loc[1]]
        cost = COSTS.get(terrain, 5)
        self.battery -= cost
        self.path_cost += cost

    def recharge(self):
        self.battery = min(INITIAL_BATTERY, self.battery + 50)
        self.add_message(f"RECHARGE: Battery at {round(self.battery)}/{INITIAL_BATTERY}")

    def get_euclidean_distance(self, loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def decide_action(self, heuristic_func):
        """Smarter logic to handle low initial battery."""
        # --- NEW FOR PART 2: HAZARD CHECK ---
        current_terrain = self.terrain_map[self.location[0]][self.location[1]]
        if current_terrain in ["trap", "hazardous"]:
            self.add_message(f"DANGER: Hit {current_terrain}! Backtracking to {self.last_safe_location}.")
            
            # Add to memory so A* avoids it in the future
            self.known_hazards.add(self.location)
            
            # Clear any existing plan
            self.path.clear()
            
            # Create a new, 1-step path back to safety
            self.path.append(self.last_safe_location)
            
            # Execute the move
            return "MOVE"
        # --- END NEW ---

        
        is_at_station = self.terrain_map[self.location[0]][self.location[1]] == "recharge station"
        if is_at_station and self.battery < INITIAL_BATTERY:
            return "RECHARGE"
        if (0.20 * INITIAL_BATTERY <= self.battery <= 0.25 * INITIAL_BATTERY):
            all_stations = self.find_all_recharge_stations()
            best_station = None
            min_cost = float('inf')

            for station in all_stations:
                path, _, cost = self.a_star_search(self.location, station, heuristic_func)
                if path and cost < min_cost:
                    min_cost = cost
                    best_station = station
            
            if best_station and self.plan_path(best_station, heuristic_func):
                 self.add_message("LOW BATT: Overriding mission to recharge.")
                 return "MOVE"

        if not self.path:
            if self.location == self.goal:
                self.add_message("MISSION COMPLETE: Goal Reached!")
                return "STOP"
            if self.plan_path(self.goal, heuristic_func):
                self.add_message("STRATEGY: Path to goal is affordable. Proceeding.")
                return "MOVE"
            else:
                self.add_message("STRATEGY: Goal is too far. Finding a charging stop.")
                reachable_stations = []
                for station in self.find_all_recharge_stations():
                    path, _, cost = self.a_star_search(self.location, station, heuristic_func)
                    if path and self.battery >= cost:
                        dist_to_goal = self.get_euclidean_distance(station, self.goal)
                        reachable_stations.append((dist_to_goal, station))
                
                if reachable_stations:
                    best_station = min(reachable_stations, key=lambda x: x[0])[1]
                    self.add_message(f"STRATEGY: Best reachable stop is {best_station}.")
                    if self.plan_path(best_station, heuristic_func):
                        return "MOVE"

                self.add_message("FATAL: Goal and all stations are unreachable. Stopping.")
                return "STOP"
        
        return "MOVE"

    def plan_path(self, target_loc, heuristic_func):
        """Plans a path and returns True only if it's found and affordable."""
        if target_loc is None: return False
            
        path, expanded, cost_of_path = self.a_star_search(self.location, target_loc, heuristic_func)
        #self.nodes_expanded += expanded
        
        if not path:
            self.path.clear()
            return False

        if self.battery < cost_of_path:
            self.path.clear()
            self.add_message(f"INFO: Path to {target_loc} is unaffordable (costs {cost_of_path}).")
            return False
            
        self.path = deque(path[1:])
        self.add_message(f"PLANNER: New path to {target_loc} (Cost: {cost_of_path}).")
        return True
            
    def add_message(self, msg):
        self.messages.append(f"{time.strftime('%H:%M:%S')} - {msg}")

    def a_star_search(self, start, goal, heuristic_func):
        if self.terrain_map[start[0]][start[1]] in IMPASSABLE or self.terrain_map[goal[0]][goal[1]] in IMPASSABLE:
            return None, 0, 0
        g_score = {start: 0}
        f_score = {start: heuristic_func(start, goal)}
        open_list = [(f_score[start], start)]
        came_from = {}
        nodes_expanded = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while open_list:
            _, current = heapq.heappop(open_list)
            nodes_expanded += 1
            if current == goal:
                path = deque([current])
                temp = current
                while temp in came_from:
                    temp = came_from[temp]
                    path.appendleft(temp)
                return list(path), nodes_expanded, g_score[goal]
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                terrain = self.terrain_map[neighbor[0]][neighbor[1]]
                if terrain in IMPASSABLE:
                    continue
                if neighbor in self.known_hazards:
                    continue
                move_cost = COSTS[terrain]
                tentative_g_score = g_score[current] + move_cost
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    new_f_score = tentative_g_score + heuristic_func(neighbor, goal)
                    heapq.heappush(open_list, (new_f_score, neighbor))
        return None, nodes_expanded, 0
    
# ===============================================================
# SECTION 4 — GUI IMPLEMENTATION (Tkinter)
# ===============================================================


class RoverGUI:
    def __init__(self, master):
        self.master = master
        master.title("CSE518 Planetary Rover")
        self.grid_size = GRID_SIZE
        self.start = (0, 0)
        self.goal = (self.grid_size - 1, self.grid_size - 1)
        self.terrain_map = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.agent = None
        self.is_paused = False
        self.is_running = False
        self.speed_delay = 200 
        self.setup_layout()
        self.step_mode = False


        self.randomize_grid()
        self.update_log("INFO: Application loaded. Press 'Start Simulation'.")
    def setup_layout(self):
        canvas_frame = tk.Frame(self.master)
        canvas_frame.grid(row=0, column=0, padx=10, pady=10)

        # --- Scrollable control panel for smaller screens ---
        controls_container = tk.Frame(self.master)
        controls_container.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        controls_canvas = tk.Canvas(controls_container, height=600, width=300)
        controls_scrollbar = tk.Scrollbar(controls_container, orient="vertical", command=controls_canvas.yview)
        controls_scrollbar.pack(side="right", fill="y")
        controls_canvas.pack(side="left", fill="both", expand=True)

        controls_frame = tk.Frame(controls_canvas)
        controls_canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        def update_scroll_region(event):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        controls_frame.bind("<Configure>", update_scroll_region)

        def _on_mousewheel(event):
            controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        controls_canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Windows/macOS
        controls_canvas.bind_all("<Button-4>", lambda e: controls_canvas.yview_scroll(-1, "units"))  # Linux scroll up
        controls_canvas.bind_all("<Button-5>", lambda e: controls_canvas.yview_scroll(1, "units"))   # Linux scroll down

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.grid_size * TILE_SIZE,
            height=self.grid_size * TILE_SIZE,
            bg="white"
        )
        self.canvas.pack()

        tk.Label(controls_frame, text="Controls", font=('Arial', 12, 'bold')).pack(pady=(0, 5), anchor='w')
        self.heuristic_var = tk.StringVar(self.master, "Manhattan")
        tk.OptionMenu(controls_frame, self.heuristic_var, *HEURISTICS.keys()).pack(pady=5, fill='x')
        tk.Button(controls_frame, text="Randomize Grid", command=self.randomize_grid).pack(pady=5, fill='x')
        tk.Button(controls_frame, text="Reset Agent", command=self.reset_agent).pack(pady=5, fill='x')
        tk.Button(controls_frame, text="Compare Heuristics", command=self.compare_heuristics, bg="lightblue").pack(pady=5, fill='x')

      
        self.step_button = tk.Button(controls_frame, text="Start Simulation", command=self.step_simulation, bg='light blue')
        self.step_button.pack(pady=10, fill='x')

        stats_frame = tk.LabelFrame(controls_frame, text="Agent Status", padx=10, pady=10)
        stats_frame.pack(pady=10, fill='x')
        self.loc_label = tk.Label(stats_frame, text="Location: (0, 0)")
        self.loc_label.pack(anchor='w')
        self.battery_label = tk.Label(stats_frame, text=f"Battery: {INITIAL_BATTERY}/{INITIAL_BATTERY}")
        self.battery_label.pack(anchor='w')

        # --- Battery Progress Bar ---
        self.battery_bar = ttk.Progressbar(stats_frame, length=200, mode='determinate', maximum=INITIAL_BATTERY)
        self.battery_bar.pack(anchor='w', pady=(3, 6))
        self.battery_bar['value'] = INITIAL_BATTERY
        #self.nodes_label = tk.Label(stats_frame, text="Nodes Expanded: 0")
        #self.nodes_label.pack(anchor='w')
        self.cost_label = tk.Label(stats_frame, text="Path Cost: 0")
        self.cost_label.pack(anchor='w')

        # --- Simulation Control Buttons ---
        sim_control_frame = tk.LabelFrame(controls_frame, text="Simulation Control", padx=10, pady=10)
        sim_control_frame.pack(pady=10, fill='x')
        tk.Button(sim_control_frame, text="Pause", command=self.pause_simulation, bg="light gray").pack(pady=2, fill='x')
        tk.Button(sim_control_frame, text="Resume", command=self.resume_simulation, bg="light gray").pack(pady=2, fill='x')
        tk.Button(sim_control_frame, text="Step Once", command=self.step_once, bg="light gray").pack(pady=2, fill='x')

        # --- Agent Log ---
        log_frame = tk.LabelFrame(controls_frame, text="Agent Log", padx=10, pady=10)
        log_frame.pack(pady=10, fill='x')
        self.log_text = tk.StringVar()
        tk.Label(log_frame, textvariable=self.log_text, justify=tk.LEFT, font=('Courier', 9), wraplength=250).pack()

        # --- Animation Speed Control ---
        speed_frame = tk.LabelFrame(controls_frame, text="Animation Speed", padx=10, pady=10)
        speed_frame.pack(pady=10, fill='x')
        tk.Label(speed_frame, text="Adjust Rover Speed (ms delay):").pack(anchor='w')
        self.speed_slider = tk.Scale(speed_frame, from_=50, to=1000, orient='horizontal', command=self.update_speed)
        self.speed_slider.set(200)
        self.speed_slider.pack(fill='x', pady=5)

        # --- Legend ---
        legend_frame = tk.LabelFrame(controls_frame, text="Legend", padx=10, pady=10)
        legend_frame.pack(pady=10, fill='x')
        legend_items = {
            "Flat": COLORS["flat"], "Sandy": COLORS["sandy"], "Rocky": COLORS["rocky"],
            "Cliff": COLORS["cliff"], "Trap": COLORS["trap"], "Hazardous": COLORS["hazardous"],
            "Recharge Station": COLORS["recharge station"], "Path": COLORS["path"],
            "Rover": COLORS["current"], "Start": COLORS["start"], "Goal": COLORS["goal"]
        }
        for text, color in legend_items.items():
            f = tk.Frame(legend_frame)
            f.pack(anchor="w", pady=1)
            tk.Label(f, bg=color, width=2, height=1, relief="solid", bd=1).pack(side="left", padx=(0, 5))
            tk.Label(f, text=text).pack(side="left")


    def randomize_grid(self):
        self.canvas.delete("all")
        self.terrain_map = [[None] * self.grid_size for _ in range(self.grid_size)]
        
        self.terrain_map[self.start[0]][self.start[1]] = "start"
        self.terrain_map[self.goal[0]][self.goal[1]] = "goal"

        for _ in range(5): 
            while True:
                r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                if self.terrain_map[r][c] is None:
                    self.terrain_map[r][c] = "recharge station"
                    break
        
        terrain_types = ["flat"] * 6 + ["sandy"] * 3 + ["rocky"] * 1 + ["cliff"] * 1 + ["trap"] * 1 + ["hazardous"] * 1
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.terrain_map[r][c] is None:
                    self.terrain_map[r][c] = random.choice(terrain_types)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self.draw_cell(r, c)
        
        self.reset_agent() 
        self.update_log("INFO: New grid randomized. Agent reset.")

    def draw_cell(self, r, c, type_override=None):
        x1, y1 = c * TILE_SIZE, r * TILE_SIZE
        x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE
        terrain_type = type_override or self.terrain_map[r][c]
        color = COLORS.get(terrain_type, "white")
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        
        label_info = {"start": "S", "goal": "G", "recharge station": "R", "cliff": "C", "trap": "T", "hazardous": "H"}
        if terrain_type in label_info:
            self.canvas.create_text(x1 + TILE_SIZE/2, y1 + TILE_SIZE/2, text=label_info[terrain_type], fill="white", font=("Arial", 10, "bold"))
        
    def draw_path(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.terrain_map[r][c] not in ["start", "goal", "recharge station"]:
                    self.draw_cell(r, c)
        
        if self.agent.path:
            for r, c in self.agent.path:
                self.draw_cell(r, c, "path")
        
        r, c = self.agent.location
        self.draw_cell(r, c, "current")

    def reset_agent(self):
        self.agent = RoverAgent(self.start, self.goal, self.grid_size, self.terrain_map)
        self.draw_info()
        self.draw_path()
        self.update_log("INFO: Agent state reset to Start.")
        self.step_button.config(text="Start Simulation", state=tk.NORMAL)

    def update_log(self, message):
        if self.agent:
            self.agent.add_message(message)
            self.log_text.set("\n".join(self.agent.messages))
        
    def draw_info(self):
        if self.agent:
            battery = round(self.agent.battery)
            self.loc_label.config(text=f"Location: {self.agent.location}")
            self.battery_label.config(text=f"Battery: {battery}/{INITIAL_BATTERY}")
            #self.nodes_label.config(text=f"Nodes Expanded: {self.agent.nodes_expanded}")
            self.cost_label.config(text=f"Path Cost: {self.agent.path_cost}")
            self.battery_bar['value'] = battery

            
            if battery > 50:
                self.battery_bar.config(style="green.Horizontal.TProgressbar")
            elif battery > 20:
                self.battery_bar.config(style="yellow.Horizontal.TProgressbar")
            else:
                self.battery_bar.config(style="red.Horizontal.TProgressbar")
    def pause_simulation(self):
        """Pause the simulation."""
        if not self.is_running:
            self.update_log("WARN: Simulation not running.")
            return
        self.is_paused = True
        self.update_log("INFO: Simulation paused.")
        self.step_button.config(text="Paused")

    def resume_simulation(self):
        """Resume the simulation."""
        if not self.is_running:
            self.update_log("WARN: Simulation not started yet.")
            return
        if not self.is_paused:
            self.update_log("WARN: Simulation already running.")
            return
        self.is_paused = False
        self.update_log("INFO: Simulation resumed.")
        self.step_button.config(text="Running...")
        self.master.after(self.speed_delay, self.step_simulation)

    def step_once(self):
        """Advance the simulation by exactly one step."""
        if not self.is_running:
            self.is_running = True
            self.update_log("INFO: Step-by-step simulation started.")

        self.step_mode = True   # activate step mode
        self.is_paused = False  # allow one iteration
        self.step_simulation()



    def step_simulation(self):
        """Main loop controlling rover movement and decision-making."""
        if self.is_paused:
            return

        if not self.is_running:
            self.is_running = True
            self.update_log("INFO: Simulation started.")
            self.step_button.config(text="Running...")

        if self.agent.location == self.agent.goal:
            messagebox.showinfo("Simulation Complete", "Goal reached! Mission successful.")
            self.is_running = False
            self.step_button.config(text="COMPLETE", state=tk.DISABLED)
            return



        heuristic_func = HEURISTICS[self.heuristic_var.get()]
        action = self.agent.decide_action(heuristic_func)
        
        if action == "MOVE":
            if self.agent.path:
                self.agent.update_location(self.agent.path.popleft())
            else:
                self.update_log("WARN: Path empty. Re-planning.")
        elif action == "RECHARGE":
            self.agent.recharge()
        elif action == "STOP":
            self.update_log("INFO: Agent has stopped.")
            self.step_button.config(text="STOPPED", state=tk.DISABLED)
            if self.agent.location != self.agent.goal:
                 messagebox.showwarning("Simulation Stopped", "Agent stopped. Goal may be unreachable or battery is too low.")
            return

        self.draw_path()
        self.draw_info()
        
        if not self.is_paused and self.is_running and action != "STOP":
            if self.step_mode:
                self.is_paused = True
                self.step_mode = False
                self.update_log("INFO: Step executed. Simulation paused.")
                self.step_button.config(text="Paused")
            else:
                # Normal continuous running
                self.master.after(self.speed_delay, self.step_simulation)

    def update_speed(self, val):
        self.speed_delay = int(val)
    # To avoid flooding logs on every slider move
        if int(val) % 100 == 0:
            self.update_log(f"INFO: Speed delay set to {self.speed_delay} ms.")
        # ===============================================================
        # ===============================================================
    # MULTI-HEURISTIC OVERLAY FEATURE
    # ===============================================================
    def compare_heuristics(self):
        """Run all 4 heuristics on the same grid and overlay paths in different colors."""
        self.update_log("INFO: Comparing all heuristics on current map...")

        # Remove old overlays only (keep terrain)
        self.canvas.delete("path_overlay")

        # Redraw terrain grid (so paths draw cleanly on top)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self.draw_cell(r, c)

        colors_used = []
        results = []

        # Run A* for each heuristic
        temp_agent = RoverAgent(self.start, self.goal, self.grid_size, self.terrain_map)

        for h_name, h_func in HEURISTICS.items():
            path, expanded, cost = temp_agent.a_star_search(self.start, self.goal, h_func)
            if not path:
                self.update_log(f"WARN: {h_name} could not find a path.")
                continue

            color = HEURISTIC_COLORS[h_name]
            colors_used.append((h_name, color))
            results.append((h_name, cost, len(path)))

            # Draw the path in its unique color
            width_val = 2 + list(HEURISTICS.keys()).index(h_name)
            for i in range(len(path) - 1):
                (r1, c1) = path[i]
                (r2, c2) = path[i + 1]
                x1, y1 = c1 * TILE_SIZE + TILE_SIZE / 2, r1 * TILE_SIZE + TILE_SIZE / 2
                x2, y2 = c2 * TILE_SIZE + TILE_SIZE / 2, r2 * TILE_SIZE + TILE_SIZE / 2
                # Apply small pixel offset for overlapping visibility
                offsets = {
                    "Manhattan": (-4, -4),
                    "Euclidean": (4, -4),
                    "Squared Manhattan": (-4, 4),
                    "Logarithmic": (4, 4)
                }

                dx, dy = offsets[h_name]
                width_val = 3  # uniform line width for all

                # Draw each segment with spatial offset so overlaps are distinct
                for i in range(len(path) - 1):
                    (r1, c1) = path[i]
                    (r2, c2) = path[i + 1]

                    x1 = c1 * TILE_SIZE + TILE_SIZE / 2 + dx
                    y1 = r1 * TILE_SIZE + TILE_SIZE / 2 + dy
                    x2 = c2 * TILE_SIZE + TILE_SIZE / 2 + dx
                    y2 = r2 * TILE_SIZE + TILE_SIZE / 2 + dy

                    self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill=color,
                        width=width_val,
                        tags="path_overlay"
                    )
        # --- Legend ---
        legend_x = self.grid_size * TILE_SIZE - 160
        legend_y = 10
        box_height = 30 + 20 * len(colors_used)

        self.canvas.create_rectangle(
            legend_x - 10, legend_y - 5, legend_x + 140, legend_y + box_height,
            fill="white", outline="black", tags="path_overlay"
        )
        self.canvas.create_text(
            legend_x + 60, legend_y + 5,
            text="Heuristic Legend", font=("Arial", 10, "bold"), tags="path_overlay"
        )

        y_offset = 25
        for h_name, color in colors_used:
            self.canvas.create_rectangle(
                legend_x, legend_y + y_offset - 8, legend_x + 20, legend_y + y_offset + 8,
                fill=color, outline="black", tags="path_overlay"
            )
            self.canvas.create_text(
                legend_x + 80, legend_y + y_offset,
                text=h_name, anchor="w", font=("Arial", 9), tags="path_overlay"
            )
            y_offset += 20

        # --- Log Results ---
        for h_name, cost, length in results:
            self.update_log(f"{h_name}: Path Length = {length}, Cost = {cost:.1f}")

        self.update_log("INFO: Multi-heuristic comparison complete.")

        # Bring start/goal/recharge markers to top
        self.canvas.tag_raise("all")
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.terrain_map[r][c] in ["start", "goal", "recharge station"]:
                    self.draw_cell(r, c)


    
# ===============================================================
# SECTION 5 — MAIN EXECUTION
# ===============================================================


if __name__ == "__main__":
    root = tk.Tk()
    app = RoverGUI(root)
    root.mainloop()