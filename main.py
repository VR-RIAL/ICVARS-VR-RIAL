import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import json
import numpy as np
import sys
import os
from collections import deque
import heapq
from ortools.sat.python import cp_model
import paho.mqtt.client as mqtt
import ssl
import json
from dataclasses import dataclass
from layout_validator import LayoutValidator, ValidationResult
from room_allocator import SmartRoomAllocator, allocate_room_sizes, RoomAllocation
from geometry_aware_optimiser import GeometryAwareOptimizer, AxisAwareCompactor, AxisBiasedGapFiller
from vastu_relaxation_manager import RetryableFloorPlanGenerator, create_retryable_generator

NUM_CANDIDATES = 40
MAX_PLACEMENT_ATTEMPTS = 25
GRID_SIZE = 0.5
COMPACTION_STEPS = 150
GAP_FILLING_ITERATIONS = 50

VALIDATION_CONFIG = {
    'min_room_area': 15.0,
    'max_room_area': 2500.0,
    'max_aspect_ratio': 4,
    'area_cv_threshold': 1.5,
    'min_dimension': 4
}

W_AESTHETIC = 6.0
W_VASTU = 4.0

PLOT_WIDTH, PLOT_HEIGHT = 40,40
EPSILON, DOOR_WIDTH = 0.12, 3.0

DISALLOWED_ADJACENCY_RULES = {
    tuple(sorted(("Bathroom", "Kitchen"))),
    tuple(sorted(("Bathroom", "Dining Room"))),
    tuple(sorted(("Bathroom", "Pooja"))),
    tuple(sorted(("Bathroom", "Bathroom"))),
    tuple(sorted(("Bathroom", "Balcony"))),
    tuple(sorted(('Bedroom', 'Kitchen'))),
    tuple(sorted(('Master Bedroom', 'Kitchen'))),
    tuple(sorted(('Bedroom', 'Bedroom'))),
    tuple(sorted(('Master Bedroom', 'Bedroom'))),
    tuple(sorted(('Master Bedroom', 'Master Bedroom')))
}

DEFAULT_ROOM_DIMS = { "Pooja": (6, 6), "Kitchen": (10, 8), "Master Bedroom": (14, 12), "Bedroom": (12, 10), "Living Room": (20, 16), "Dining Room": (12, 10), "Bathroom": (6, 8), "Store": (6, 6), "Storage": (6, 6), "Guest Room": (10, 10), "Study": (8, 8), "Patio": (12, 8), "Balcony": (10, 4), "Hallway": (18, 5) }
VASTU_PREFS = { "Pooja": ['N', 'E'], "Kitchen": ['S', 'E'], "Master Bedroom": ['S', 'W', 'N'], "Store": ["S", "W"], "Bathroom": ["N", "W", "E"], "Guest Room": ["N", "W", "E"], "Living Room": ["N", "E", "Center"], "Dining Room": ["W", "E", "N"], "Bedroom": ["W", "S"], "Study": ["E", 'W'], "Patio": ["N", 'E'], "Balcony": ["N", "E"], "Hallway": ["Center", "N", "E", "S", "W"] }
PLACEMENT_PRIORITY = [ "Living Room", "Pooja", "Kitchen", "Dining Room", "Master Bedroom", "Bedroom", "Guest Room", "Bathroom", "Store", "Study", "Patio", "Balcony", "Hallway" ]
ROOM_COLORS = { "Pooja": 'gold', "Kitchen": 'orangered', "Master Bedroom": 'darkgreen', "Bedroom": 'mediumseagreen', "Living Room": 'skyblue', "Dining Room": 'lightblue', "Bathroom": 'lightcoral', "Store": 'gray', "Guest Room": 'plum', "Study": 'khaki', "Patio": 'lightyellow', "Balcony": 'lightgray', "Hallway": 'beige', "DEFAULT": 'silver' }
ZONE_ADJACENCY = { 'N': {'NE', 'NW'}, 'S': {'SE', 'SW'}, 'E': {'NE', 'SE'}, 'W': {'NW', 'SW'}, 'NE': {'N', 'E'}, 'NW': {'N', 'W'}, 'SE': {'S', 'E'}, 'SW': {'S', 'W'}, 'Center': {'N', 'S', 'E', 'W'}}

def get_zone(x, y, center_x, center_y):
    dx, dy = x - center_x, y - center_y
    if abs(dx) < PLOT_WIDTH * 0.2 and abs(dy) < PLOT_HEIGHT * 0.2: return 'Center'
    angle = math.degrees(math.atan2(dy, dx))
    if -22.5 <= angle < 22.5: return 'E'
    elif 22.5 <= angle < 67.5: return 'NE'
    elif 67.5 <= angle < 112.5: return 'N'
    elif 112.5 <= angle < 157.5: return 'NW'
    elif 157.5 <= angle or angle < -157.5: return 'W'
    elif -157.5 <= angle < -112.5: return 'SW'
    elif -112.5 <= angle < -67.5: return 'S'
    else: return 'SE'

def get_adjacencies(room_list, door_based=False):
    adj = {r.name: [] for r in room_list}
    for i in range(len(room_list)):
        for j in range(i + 1, len(room_list)):
            r1, r2 = room_list[i], room_list[j]
            if r1.is_adjacent_to(r2):
                adj[r1.name].append(r2.name)
                adj[r2.name].append(r1.name)
    return adj

def get_adjacencies_with_offset(room_list, door_based=False):
    adj = {r.name: [] for r in room_list}
    for i in range(len(room_list)):
        for j in range(len(room_list)):
            r1, r2 = room_list[i], room_list[j]
            adjacent,wall,offset=r1.is_adjacent_to_with_offset(r2)
            if adjacent:
                adj[r1.name].append((r2.name, wall, offset))
    converted = {}
    directions = ['top', 'right', 'bottom', 'left']
    for room, connections in adj.items():
        new_format = {d: [] for d in directions}

        for connected_room, direction, offset in connections:
            new_format[direction].append((connected_room, offset))

        for d in directions:
            new_format[d] = tuple(new_format[d])

        converted[room] = new_format
    return converted

class Room:
    def __init__(self, name, type, width, height):
        self.name, self.type, self.width, self.height = name, type, width, height
        self.x, self.y, self.placed = None, None, False
        self.vastu_prefs = VASTU_PREFS.get(self.type, [])
        self.has_attached_bath = False
        self.original_width = width
        self.original_height = height
        self.min_width = width * 0.8
        self.min_height = height * 0.8
        self.max_width = width * 1.5
        self.max_height = height * 1.5

    def set_position(self, x, y):
        self.x, self.y, self.placed = x, y, True

    def set_dimensions(self, width, height):
        self.width = max(self.min_width, min(self.max_width, width))
        self.height = max(self.min_height, min(self.max_height, height))

    @property
    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2) if self.placed else (None, None)

    def collides_with(self, other_room, tolerance=EPSILON):
        if not self.placed or not other_room.placed: return False
        return (self.x < other_room.x + other_room.width - tolerance and
                self.x + self.width > other_room.x + tolerance and
                self.y < other_room.y + other_room.height - tolerance and
                self.y + self.height > other_room.y + tolerance)

    def is_adjacent_to(self, other_room):
        if not self.placed or not other_room.placed: return False
        x_adj = abs(self.x + self.width - other_room.x) < EPSILON or abs(other_room.x + other_room.width - self.x) < EPSILON
        y_overlap = max(self.y, other_room.y) < min(self.y + self.height, other_room.y + other_room.height)
        y_adj = abs(self.y + self.height - other_room.y) < EPSILON or abs(other_room.y + other_room.height - self.y) < EPSILON
        x_overlap = max(self.x, other_room.x) < min(self.x + self.width, other_room.x + other_room.width)
        return (x_adj and y_overlap) or (y_adj and x_overlap)

    def is_adjacent_to_with_offset(self, other_room):
        if not self.placed or not other_room.placed:
            return False, None, None
        if abs(self.x + self.width - other_room.x) < EPSILON:
            if max(self.y, other_room.y) < min(self.y + self.height, other_room.y + other_room.height):
                return True, 'right', other_room.y+other_room.height/2 - self.y-self.height/2
            else:
                return False, None, None
        elif abs(other_room.x + other_room.width - self.x) < EPSILON:
            if max(self.y, other_room.y) < min(self.y + self.height, other_room.y + other_room.height):
                return True, 'left',  self.y+self.height/2-other_room.y-other_room.height/2
            else:
                return False, None, None

        elif abs(self.y + self.height - other_room.y) < EPSILON:
            if max(self.x, other_room.x) < min(self.x + self.width, other_room.x + other_room.width):
                return True, 'top', self.x+self.width/2-other_room.x-other_room.width/2
            else:
                return False, None, None
        elif abs(other_room.y + other_room.height - self.y) < EPSILON:
            if max(self.x, other_room.x) < min(self.x + self.width, other_room.x + other_room.width):
                return True, 'bottom', other_room.x+other_room.width/2 - self.x-self.width/2
            else:
                return False, None, None
        return False, None, None

    def __repr__(self): return f"Room({self.name}, Placed: {self.placed})"

class FloorPlanGenerator:
    def __init__(self, requirements_data):
        self.rooms = self._load_and_prepare_rooms(requirements_data)
        self.placed_rooms = []
        self.hallway_count = 0
        self.placement_attempts = {}
        self.plot_bounds = self._get_plot_bounds(requirements_data)

    def _get_plot_bounds(self, requirements_data):
        plot_width = requirements_data.get("house_dimensions", {}).get("width", PLOT_WIDTH)
        plot_height = requirements_data.get("house_dimensions", {}).get("length", PLOT_HEIGHT)
        return {"min_x": 0, "max_x": plot_width, "min_y": 0, "max_y": plot_height}

    def _load_and_prepare_rooms(self, requirements_data):

        allocations, metrics = allocate_room_sizes(requirements_data, DEFAULT_ROOM_DIMS)

        if metrics.get('status') == 'IMPOSSIBLE':
            print(f"\n FATAL: Cannot fit rooms into plot!")
            print(f"   {metrics.get('error')}")
            import sys
            sys.exit(1)

        user_rooms = []

        for alloc in allocations:
            user_rooms.append({
                "name": alloc.name,
                "type": alloc.type,
                "width": alloc.allocated_width,
                "height": alloc.allocated_height,
                "is_user_specified": alloc.is_user_specified
            })

        et = {r['type'] for r in user_rooms}
        flexible_scaling = metrics.get('flexible_scaling_factor', 1.0)

        if "Living Room" not in et:
            w, h = DEFAULT_ROOM_DIMS["Living Room"]
            user_rooms.append({
                "name": "Living Room",
                "type": "Living Room",
                "width": w * flexible_scaling,
                "height": h * flexible_scaling,
                "is_user_specified": False
            })
        if "Kitchen" not in et:
            w, h = DEFAULT_ROOM_DIMS["Kitchen"]
            user_rooms.append({
                "name": "Kitchen",
                "type": "Kitchen",
                "width": w * flexible_scaling,
                "height": h * flexible_scaling,
                "is_user_specified": False
            })
        if "Dining Room" not in et:
            w, h = DEFAULT_ROOM_DIMS["Dining Room"]
            user_rooms.append({
                "name": "Dining Room",
                "type": "Dining Room",
                "width": w * flexible_scaling,
                "height": h * flexible_scaling,
                "is_user_specified": False
            })

        nbd = sum(1 for r in user_rooms if "Bedroom" in r['type'])
        nba = sum(1 for r in user_rooms if "Bathroom" in r['type'])
        if nbd > 0 and nba == 0:
            w, h = DEFAULT_ROOM_DIMS["Bathroom"]
            for i in range(max(1, (nbd + 1) // 2)):
                user_rooms.append({
                    "name": f"Bathroom {i+1}",
                    "type": "Bathroom",
                    "width": w * flexible_scaling,
                    "height": h * flexible_scaling,
                    "is_user_specified": False
                })

        bf = False
        for r in user_rooms:
            if r['type'] == "Bedroom" and not bf:
                r['type'] = "Master Bedroom"
                r['name'] = r['name'].replace("Bedroom", "Master Bedroom")
                bf = True

        fr = []
        for r in user_rooms:
            room = Room(r['name'], r['type'], r['width'], r['height'])

            if r.get('is_user_specified', False):
                room.min_width = r['width']
                room.max_width = r['width']
                room.min_height = r['height']
                room.max_height = r['height']

            fr.append(room)

        fr.sort(key=lambda r: PLACEMENT_PRIORITY.index(r.type) if r.type in PLACEMENT_PRIORITY else len(PLACEMENT_PRIORITY))
        return fr

    def generate_layout(self):
        unplaced = deque(self.rooms)
        anchor = next((r for r in unplaced if r.type == "Living Room"), unplaced[0])
        anchor.set_position(0, 0); self.placed_rooms.append(anchor); unplaced.remove(anchor)
        self.placement_attempts = {r.name: 0 for r in self.rooms}
        stuck = []
        while unplaced:
            room = unplaced.popleft()
            self.placement_attempts[room.name] += 1
            if self.placement_attempts[room.name] > MAX_PLACEMENT_ATTEMPTS: stuck.append(room); continue
            if self._attempt_connected_placement(room): continue
            else:
                colliding, anc = self._find_collision_details(room)
                if not colliding or not anc: unplaced.append(room); continue

                original_pos = (colliding.x, colliding.y)

                self.placed_rooms.remove(colliding); colliding.placed = False
                hallway = self._create_and_place_hallway(anc, colliding, original_pos)
                if not hallway: self.placed_rooms.append(colliding); colliding.placed = True
                unplaced.appendleft(room); unplaced.appendleft(colliding)
        if stuck:
            print(f"Attempting to force-place {len(stuck)} stuck room(s)...")
            for r in stuck: self._force_place_room(r)

        print("\n" + "="*70)
        print("GEOMETRY-AWARE OPTIMIZATION PIPELINE")
        print("="*70)

        optimizer = GeometryAwareOptimizer(
            plot_width=self.plot_bounds["max_x"],
            plot_height=self.plot_bounds["max_y"],
            aspect_ratio_weight=100.0,
            movement_weight=1.0
        )
        optimizer.optimize_layout(self, time_limit=30)

        compactor = AxisAwareCompactor(
            plot_width=self.plot_bounds["max_x"],
            plot_height=self.plot_bounds["max_y"],
            grid_size=GRID_SIZE
        )
        compactor.compact(self, max_iterations=COMPACTION_STEPS)

        gap_filler = AxisBiasedGapFiller(
            plot_width=self.plot_bounds["max_x"],
            plot_height=self.plot_bounds["max_y"],
            grid_size=GRID_SIZE
        )

        print("="*70 + "\n")

    def _get_layout_center(self):
        if not self.placed_rooms: return 0, 0
        min_x = min(r.x for r in self.placed_rooms); max_x = max(r.x + r.width for r in self.placed_rooms)
        min_y = min(r.y for r in self.placed_rooms); max_y = max(r.y + r.height for r in self.placed_rooms)
        return (min_x + max_x) / 2, (min_y + max_y) / 2

    def _find_main_component(self):
        if not self.placed_rooms: return []
        adj = get_adjacencies(self.placed_rooms)
        visited, all_components = set(), []
        for room in self.placed_rooms:
            if room.name not in visited:
                comp, q = [], deque([room.name])
                visited.add(room.name); comp.append(room.name)
                while q:
                    cn = q.popleft()
                    for nn in adj.get(cn, []):
                        if nn not in visited: visited.add(nn); comp.append(nn); q.append(nn)
                all_components.append(comp)
        if not all_components: return self.placed_rooms
        main_comp_names = max(all_components, key=len)
        return [r for r in self.placed_rooms if r.name in main_comp_names]

    def _attempt_connected_placement(self, room):
        layout_cx, layout_cy = self._get_layout_center()
        ideal_spots, compromise_spots = [], []
        main_component_anchors = self._find_main_component()

        if "Bedroom" in room.type:

            main_component_anchors = [r for r in main_component_anchors if "Bedroom" not in r.type]

        if room.type == "Bathroom" and not any(br.has_attached_bath for br in self.placed_rooms if "Bedroom" in br.type):
            mb = next((r for r in self.placed_rooms if r.type == "Master Bedroom"), None)
            if mb: main_component_anchors.insert(0, mb)

        for anchor in random.sample(main_component_anchors, len(main_component_anchors)) if main_component_anchors else []:
            for direction in ['N', 'S', 'E', 'W']:
                if direction in ['N', 'S']:
                    static_y = anchor.y + anchor.height if direction == 'N' else anchor.y - room.height
                    for x in np.linspace(anchor.x - room.width + DOOR_WIDTH, anchor.x + anchor.width - DOOR_WIDTH, 5):
                        self._classify_spot(room, x, static_y, layout_cx, layout_cy, ideal_spots, compromise_spots)
                else:
                    static_x = anchor.x + anchor.width if direction == 'E' else anchor.x - room.width
                    for y in np.linspace(anchor.y - room.height + DOOR_WIDTH, anchor.y + anchor.height - DOOR_WIDTH, 5):
                        self._classify_spot(room, static_x, y, layout_cx, layout_cy, ideal_spots, compromise_spots)
        if ideal_spots:
            x, y = random.choice(ideal_spots);
            return self._check_and_place(room, x, y)
        if compromise_spots:
            x, y = random.choice(compromise_spots);
            return self._check_and_place(room, x, y)
        return False

    def _classify_spot(self, room, x, y, layout_cx, layout_cy, ideal_spots, compromise_spots):
        temp_room = Room(room.name, room.type, room.width, room.height)
        snapped_x, snapped_y = round(x / GRID_SIZE) * GRID_SIZE, round(y / GRID_SIZE) * GRID_SIZE
        temp_room.set_position(snapped_x, snapped_y)
        if any(temp_room.collides_with(pr) for pr in self.placed_rooms): return
        neighbors = [pr for pr in self.placed_rooms if temp_room.is_adjacent_to(pr)]
        if not neighbors: return

        for p_room in neighbors:
            pair = tuple(sorted((room.type, p_room.type)))
            if pair in DISALLOWED_ADJACENCY_RULES: return

            if "Bedroom" in room.type and "Bedroom" in p_room.type: return
            if room.type == "Bathroom" and "Bedroom" in p_room.type and p_room.has_attached_bath: return
            if "Bedroom" in room.type and p_room.type == "Bathroom" and room.has_attached_bath: return
        if room.type == "Balcony" and all(p_room.type == "Balcony" for p_room in neighbors): return
        zone = get_zone(temp_room.center[0], temp_room.center[1], layout_cx, layout_cy)
        if zone in room.vastu_prefs: ideal_spots.append((snapped_x, snapped_y))
        else: compromise_spots.append((snapped_x, snapped_y))

    def _force_place_room(self, room):

        preferred_anchors = []
        other_anchors = []

        for r in self.placed_rooms:
            if r.type in ["Living Room", "Dining Room", "Hallway"]:
                preferred_anchors.append(r)
            elif "Bedroom" not in r.type:
                other_anchors.append(r)

        for anchor_list in [preferred_anchors, other_anchors]:
            for r in anchor_list:
                for direction in ['N', 'E', 'S', 'W']:
                    x, y = self._get_target_coords_for_door(room, r, direction)
                    ideal, compromise = [], []
                    self._classify_spot(room, x, y, 0, 0, ideal, compromise)
                    if ideal or compromise:
                        spot = ideal[0] if ideal else compromise[0]
                        if self._check_and_place(room, spot[0], spot[1]):
                            print(f"INFO: Force-placed '{room.name}'.");
                            return
        print(f"ERROR: Could not force-place '{room.name}'.")

    def _find_collision_details(self, room):
        main_comp = self._find_main_component()
        for anchor in main_comp:
            if anchor is room: continue
            for direction in ['N', 'S', 'E', 'W']:
                x, y = self._get_target_coords_for_door(room, anchor, direction)
                room.set_position(x, y)
                for pr in self.placed_rooms:
                    if pr is room: continue
                    if room.collides_with(pr): room.placed = False; return pr, anchor
                room.placed = False
        if main_comp and self.placed_rooms:
            colliding = next((r for r in reversed(self.placed_rooms) if r is not room), None)
            anchor = next((r for r in reversed(main_comp) if r is not room), None)
            if colliding and anchor: return colliding, anchor
        return None, None

    def _create_and_place_hallway(self, anchor, original, original_pos):
        self.hallway_count += 1

        hallway_name = f"Hallway {self.hallway_count}"
        self.placement_attempts[hallway_name] = 0

        original_x, original_y = original_pos
        original_center = (original_x + original.width / 2, original_y + original.height / 2)

        potential_hallways = []

        for direction in ['N', 'S', 'E', 'W']:
            if direction in ['N', 'S']:
                width, height = original.width, 5
                static_y = anchor.y + anchor.height if direction == 'N' else anchor.y - height
                for x in np.linspace(anchor.x, anchor.x + anchor.width - width, 3):
                    potential_hallways.append(Room(hallway_name, "Hallway", width, height))
                    potential_hallways[-1].set_position(x, static_y)
            else:
                width, height = 5, original.height
                static_x = anchor.x + anchor.width if direction == 'E' else anchor.x - width
                for y in np.linspace(anchor.y, anchor.y + anchor.height - height, 3):
                    potential_hallways.append(Room(hallway_name, "Hallway", width, height))
                    potential_hallways[-1].set_position(static_x, y)

        potential_hallways.sort(key=lambda r: math.hypot(r.center[0] - original_center[0], r.center[1] - original_center[1]))

        for hallway in potential_hallways:

            if not any(hallway.collides_with(pr) for pr in self.placed_rooms):

                temp_placed = self.placed_rooms[:] + [hallway]
                temp_room = Room(original.name, original.type, original.width, original.height)
                temp_room.set_position(original_x, original_y)

                is_reconnectable = not any(temp_room.collides_with(pr) for pr in temp_placed)

                if is_reconnectable:

                    self.placed_rooms.append(hallway)
                    return hallway

        hx, hy = self._get_target_coords_for_door(Room("", "", original.width, original.height), anchor, 'N' if abs(original_center[1] - anchor.center[1]) > abs(original_center[0] - anchor.center[0]) else 'E')
        hallway = Room(hallway_name, "Hallway", DEFAULT_ROOM_DIMS['Hallway'][0], DEFAULT_ROOM_DIMS['Hallway'][1])
        if self._check_and_place(hallway, hx, hy):
            return hallway

        return None

    def _check_and_place(self, room, x, y):
        temp_room = Room(room.name, room.type, room.width, room.height); temp_room.set_position(x, y)
        if any(temp_room.collides_with(pr) for pr in self.placed_rooms): return False
        room.set_position(x,y); self.placed_rooms.append(room)

        if room.type == "Bathroom":
            for pr in self.placed_rooms:
                if "Bedroom" in pr.type and room.is_adjacent_to(pr) and not pr.has_attached_bath:
                    pr.has_attached_bath = True
                    room.has_attached_bath = True
                    break
        elif "Bedroom" in room.type:
            for pr in self.placed_rooms:
                if pr.type == "Bathroom" and room.is_adjacent_to(pr) and not room.has_attached_bath:
                    room.has_attached_bath = True
                    pr.has_attached_bath = True
                    break
        return True

    def _get_target_coords_for_door(self, room, anchor, direction):
        ax, ay, aw, ah, rw, rh = anchor.x, anchor.y, anchor.width, anchor.height, room.width, room.height
        if direction == 'N': return (ax + (aw - rw) / 2, ay + ah)
        if direction == 'S': return (ax + (aw - rw) / 2, ay - rh)
        if direction == 'E': return (ax + aw, ay + (ah - rh) / 2)
        if direction == 'W': return (ax - rw, ay + (ah - rh) / 2)
        return (ax, ay)

    def _try_move(self, room, dx, dy):
        ox, oy = room.x, room.y
        room.set_position(ox + dx, oy + dy)
        if any(room.collides_with(o) for o in self.placed_rooms if o is not room):
            room.set_position(ox, oy); return False
        if not any(room.is_adjacent_to(o) for o in self.placed_rooms if o is not room):
            room.set_position(ox, oy); return False
        one_bath_exists = any(br.has_attached_bath for br in self.placed_rooms if "Bedroom" in br.type and br is not room)
        for o in self.placed_rooms:
            if o is room or not room.is_adjacent_to(o): continue
            pair = tuple(sorted((room.type, o.type)))
            if pair in DISALLOWED_ADJACENCY_RULES: room.set_position(ox, oy); return False
            if "Bedroom" in room.type and "Bedroom" in o.type: room.set_position(ox, oy); return False
            if room.type == "Bathroom" and "Bedroom" in o.type and (o.has_attached_bath): room.set_position(ox, oy); return False
            if "Bedroom" in room.type and o.type == "Bathroom" and (room.has_attached_bath): room.set_position(ox, oy); return False
        return True

    def draw(self, output_path, score_card=None):
        fig, ax = plt.subplots(figsize=(18, 18))
        placed = [r for r in self.placed_rooms if r.placed]
        if not placed: ax.text(0.5, 0.5, "Layout Generation Failed.", ha='center', va='center')
        else:
            min_x, max_x = min(r.x for r in placed), max(r.x + r.width for r in placed)
            min_y, max_y = min(r.y for r in placed), max(r.y + r.height for r in placed)
            cx, cy = self._get_layout_center()
            ax.set_xlim(min_x - 15, max_x + 15); ax.set_ylim(min_y - 15, max_y + 15)

            plot_rect = patches.Rectangle((self.plot_bounds["min_x"], self.plot_bounds["min_y"]),
                                         self.plot_bounds["max_x"] - self.plot_bounds["min_x"],
                                         self.plot_bounds["max_y"] - self.plot_bounds["min_y"],
                                         linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(plot_rect)
            ax.text(self.plot_bounds["max_x"]/2, self.plot_bounds["max_y"] + 2,
                   f"Plot Boundary ({self.plot_bounds['max_x']}x{self.plot_bounds['max_y']})",
                   ha='center', fontsize=12, color='red', weight='bold')

            for room in placed:

                color = ROOM_COLORS.get(room.type, ROOM_COLORS["DEFAULT"])
                rect = patches.Rectangle((room.x, room.y), room.width, room.height, linewidth=2, edgecolor='black', facecolor=color, alpha=0.8)
                ax.add_patch(rect)
                zone = get_zone(room.center[0], room.center[1], cx, cy)
                ax.text(room.center[0], room.center[1], f"{room.name}\n({room.width:.1f}x{room.height:.1f})\nZone: {zone}", ha='center', va='center', fontsize=10, weight='bold')
            drawn_doors_pairs=self._draw_doors(ax, placed)
        handles, ut = [], set()
        for r in self.rooms:
            if r.type not in ut: handles.append(patches.Patch(color=ROOM_COLORS.get(r.type, 'silver'), label=r.type)); ut.add(r.type)
        if score_card:
            st = "--- Layout Score Card ---\n" + "\n".join([f"{k}: {v:.2f}" for k, v in score_card.items()])
            fig.text(0.87, 0.6, st, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), title="Room Types")
        plt.title('Vastu Layout (AI-Scored & Optimized with Gap Filling)', fontsize=18)
        ax.set_aspect('equal', adjustable='box'); plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(output_path, dpi=200)
        plt.close()
        adjw=get_adjacencies_with_offset(placed)

        json_path = os.path.splitext(output_path)[0] + '.json'
        rooms_data = {"rooms": [],"doors":[]}
        def _convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: _convert_tuples(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert_tuples(i) for i in obj]
            return obj

        for r in placed:
            adjacency = adjw.get(r.name, {})
            adjacency = _convert_tuples(adjacency)
            rooms_data["rooms"].append({
                'name': r.name,
                'type': r.type,
                'width': float(r.height),
                'length': float(r.width),
                'x': float(r.x+r.width/2),
                'y': float(-r.y-r.height/2),
                'adjacency': adjacency
            })

        for d in drawn_doors_pairs:
            rooms_data["doors"].append({
                'x': float(d[0]),
                'y': float(-d[1]),
                'room1': d[2],
                'room2': d[3],
                'orientation': d[4]
            })
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(rooms_data, jf, indent=2)
        print(f"Saved room positions to {json_path}")

    def _draw_doors(self, ax, placed_rooms):
        adj = get_adjacencies(placed_rooms, door_based=True)
        room_map = {r.name: r for r in placed_rooms}
        drawn_doors_pairs = set()
        rooms_with_at_least_one_door = set()

        for r1n, neighbors in adj.items():
            if 'Bathroom' in room_map[r1n].type:
                for r2n in neighbors:
                    if 'Bedroom' in room_map[r2n].type and r1n not in rooms_with_at_least_one_door:
                        self._plot_door(ax, r1n, r2n, room_map, drawn_doors_pairs)
                        rooms_with_at_least_one_door.add(r1n)
                        break

        living_room_name = next((r.name for r in placed_rooms if 'Living Room' in r.type), None)
        if living_room_name:

            non_bedroom_neighbors = [r2n for r2n in adj.get(living_room_name, [])
                                     if 'Bedroom' not in room_map[r2n].type and r2n not in rooms_with_at_least_one_door]
            bedroom_neighbors = [r2n for r2n in adj.get(living_room_name, [])
                                 if 'Bedroom' in room_map[r2n].type and r2n not in rooms_with_at_least_one_door]

            for r2n in non_bedroom_neighbors + bedroom_neighbors:
                if r2n not in rooms_with_at_least_one_door:
                    self._plot_door(ax, living_room_name, r2n, room_map, drawn_doors_pairs)
                    rooms_with_at_least_one_door.add(r2n)

        for r1n in [r.name for r in placed_rooms if 'Hallway' in r.type]:
            for r2n in adj.get(r1n, []):
                if r2n not in rooms_with_at_least_one_door:
                    self._plot_door(ax, r1n, r2n, room_map, drawn_doors_pairs)
                    rooms_with_at_least_one_door.add(r2n)

        for r1n, neighbors in adj.items():
            for r2n in neighbors:

                if 'Bedroom' in room_map[r1n].type and 'Bedroom' in room_map[r2n].type:
                    continue

                if (r1n not in rooms_with_at_least_one_door or r2n not in rooms_with_at_least_one_door) and \
                    not ('Bathroom' in room_map[r1n].type and r1n in rooms_with_at_least_one_door) and \
                    not ('Bathroom' in room_map[r2n].type and r2n in rooms_with_at_least_one_door):

                    self._plot_door(ax, r1n, r2n, room_map, drawn_doors_pairs)
                    rooms_with_at_least_one_door.add(r1n)
                    rooms_with_at_least_one_door.add(r2n)
        return drawn_doors_pairs

    def _plot_door(self, ax, r1n, r2n, room_map, drawn_doors_pairs):
        if tuple(sorted((r1n, r2n))) in drawn_doors_pairs: return
        r1, r2 = room_map.get(r1n), room_map.get(r2n)
        if not r1 or not r2: return
        if abs(r1.x + r1.width - r2.x) < EPSILON or abs(r2.x + r2.width - r1.x) < EPSILON:
            omin, omax = max(r1.y, r2.y), min(r1.y + r1.height, r2.y + r2.height)
            if omax - omin >= DOOR_WIDTH:
                dy, dx = (omin + omax) / 2, max(r1.x, r2.x)
                ax.plot([dx, dx], [dy - DOOR_WIDTH/2, dy + DOOR_WIDTH/2], color='white', lw=6, solid_capstyle='round')
                drawn_doors_pairs.add(tuple((dx, dy, r1n, r2n,"vertical")))
        elif abs(r1.y + r1.height - r2.y) < EPSILON or abs(r2.y + r2.height - r1.y) < EPSILON:
            omin, omax = max(r1.x, r2.x), min(r1.x + r1.width, r2.x + r2.width)
            if omax - omin >= DOOR_WIDTH:
                dx, dy = (omin + omax) / 2, max(r1.y, r2.y)
                ax.plot([dx - DOOR_WIDTH/2, dx + DOOR_WIDTH/2], [dy, dy], color='white', lw=6, solid_capstyle='round')
                drawn_doors_pairs.add(tuple((dx, dy, r1n, r2n, "horizontal")))

    def _is_fully_accessible_from_anchor(self):
        if not self.placed_rooms: return False
        anchor = next((r for r in self.placed_rooms if r.type == "Living Room"), self.placed_rooms[0])
        adj = get_adjacencies(self.placed_rooms)
        q, visited = deque([anchor.name]), {anchor.name}
        while q:
            cn = q.popleft()
            for nn in adj.get(cn, []):
                if nn not in visited: visited.add(nn); q.append(nn)
        return len(visited) == len(self.placed_rooms)

class LayoutScorer:
    def __init__(self, placed, all_rooms):
        self.placed, self.all_rooms = placed, all_rooms
        self.cx, self.cy = self._get_layout_center()

    def _get_layout_center(self):
        if not self.placed: return 0, 0
        min_x = min(r.x for r in self.placed); max_x = max(r.x + r.width for r in self.placed)
        min_y = min(r.y for r in self.placed); max_y = max(r.y + r.height for r in self.placed)
        return (min_x + max_x) / 2, (min_y + max_y) / 2

    def _calculate_vastu_score(self):
        score, count = 0, 0
        for r in self.placed:
            if r.type == "Hallway": continue
            zone = get_zone(r.center[0], r.center[1], self.cx, self.cy)
            prefs = r.vastu_prefs
            if zone in prefs: score += 100
            elif any(z in ZONE_ADJACENCY.get(zone, set()) for z in prefs): score += 80
            else: score += 10
            count += 1
        return (score / count) if count > 0 else 0

    def _calculate_aesthetic_score(self):
        if not self.placed: return 0
        min_x, max_x = min(r.x for r in self.placed), max(r.x + r.width for r in self.placed)
        min_y, max_y = min(r.y for r in self.placed), max(r.y + r.height for r in self.placed)
        bbox_area = (max_x - min_x) * (max_y - min_y)
        room_area = sum(r.width * r.height for r in self.placed)
        return (room_area / bbox_area) * 100 if bbox_area > 0 else 0

    def get_score_card(self):
        card = { "Vastu Score": self._calculate_vastu_score(), "Aesthetic Score": self._calculate_aesthetic_score() }
        card["Total Score"] = (W_VASTU * card["Vastu Score"]) + (W_AESTHETIC * card["Aesthetic Score"])
        return card

def sendto_viewer(image_path):
    broker_url = "cda8dea3eb4c4036a2389d3011a09c5f.s1.eu.hivemq.cloud"
    broker_port = 8884
    username = "hivemq.webclient.1728110744101"
    password = "%yk*1$<vM5PGzl9wUYZ7"
    topic = "/test_topic"

    client = mqtt.Client(transport="websockets")

    client.username_pw_set(username, password)

    client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.tls_insecure_set(True)

    client.connect(broker_url, broker_port)
    client.loop_start()

    try:

        json_path = os.path.splitext(image_path)[0] + '.json'
        if not os.path.exists(json_path):
            print(f"JSON file not found at {json_path}. Nothing to publish.")
        else:
            with open(json_path, 'r', encoding='utf-8') as jf:
                payload = jf.read()

            result = client.publish(topic, payload)

            try:
                result.wait_for_publish()
            except Exception:
                pass
            print(f"Published JSON file: '{json_path}' to topic: '{topic}'")

    except Exception as e:
        print(f"Error while publishing JSON: {e}")

    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vastu_generator.py <request_output_directory>")
        sys.exit(1)

    req_dir = sys.argv[1]
    input_json_path = os.path.join(req_dir, "requirements.json")
    output_image_path = os.path.join(req_dir, "Vastu_Layout_Output_v4.png")
    validation_report_path = os.path.join(req_dir, "validation_report.json")

    if not os.path.exists(input_json_path):
        print(f"FATAL: Input file not found at {input_json_path}")
        sys.exit(1)

    print(f"Loading requirements from: {input_json_path}")
    reqs = json.load(open(input_json_path, 'r'))

    validator = LayoutValidator(**VALIDATION_CONFIG)

    print("\n" + "="*60)
    print("VALIDATING INPUT REQUIREMENTS")
    print("="*60)
    requirements_validation = validator.validate_requirements(reqs)
    print(requirements_validation)

    if not requirements_validation.is_valid:
        print("\n FATAL: Input requirements are invalid!")
        print("Please fix the room specifications and try again.")
        sys.exit(1)

    if requirements_validation.warnings:
        print("\n  Proceeding with warnings. Layout generation may fail.")
        print("Continue anyway? (y/n): ", end='')
        choice = input().strip().lower()
        if choice != 'y':
            print("Aborted by user.")
            sys.exit(0)

    best_layout, best_score, best_score_card = None, -float('inf'), None
    best_validation = None
    valid_candidates = 0

    print(f"\nGenerating and evaluating {NUM_CANDIDATES} candidate layouts...")
    print("Note: Each candidate will retry with relaxed Vastu preferences if CP-SAT fails\n")

    for i in range(NUM_CANDIDATES):
        print(f"\n{'#'*70}")
        print(f"CANDIDATE {i+1}/{NUM_CANDIDATES}")
        print(f"{'#'*70}")

        retryable = create_retryable_generator(FloorPlanGenerator, reqs, VASTU_PREFS)
        gen, retry_stats = retryable.generate_with_retries()

        if gen is None:
            print(f"\n   Candidate {i+1} failed after {retry_stats['configurations_tried']} attempts")
            print(f"     CP-SAT failures: {retry_stats['cp_sat_failures']}")
            continue

        layout = gen.placed_rooms

        print(f"\n   Generation Statistics:")
        print(f"     Total attempts: {retry_stats['configurations_tried']}")
        print(f"     Final relaxation level: {retry_stats['final_relaxation_level']}")
        print(f"     CP-SAT successes: {retry_stats['cp_sat_successes']}")
        print(f"     CP-SAT failures: {retry_stats['cp_sat_failures']}")

        if not all(r.placed for r in gen.rooms):
            print(f"   Candidate {i+1} incomplete")
            continue

        if not gen._is_fully_accessible_from_anchor():
            print(f"   Candidate {i+1} fragmented")
            continue

        validation_result = validator.validate(layout, exclude_hallways=True)

        print(f"\n   DEBUG - Layout Details:")
        print(f"     Total rooms: {len(layout)}")
        print(f"     Placed rooms: {sum(1 for r in layout if r.placed)}")
        if layout:
            areas = [r.width * r.height for r in layout if r.placed]
            print(f"     Room areas: min={min(areas):.1f}, max={max(areas):.1f}, avg={sum(areas)/len(areas):.1f}")
            print(f"     Total area: {sum(areas):.1f}")

        scorer = LayoutScorer(layout, gen.rooms)
        card = scorer.get_score_card()
        score = card["Total Score"]

        vastu_penalty = retry_stats['final_relaxation_level'] * 50
        adjusted_score = score - vastu_penalty

        status = "VALID" if validation_result.is_valid else "INVALID"
        print(f"\n  - Candidate {i+1}/{NUM_CANDIDATES}")
        print(f"    Score: {score:.2f} (adjusted: {adjusted_score:.2f} after -{vastu_penalty} Vastu penalty)")
        print(f"    Status: {status}")
        print(f"    Relaxation level: {retry_stats['final_relaxation_level']}/4")

        if not validation_result.is_valid:
            print(f"\n   VALIDATION ERRORS:")
            for error in validation_result.errors[:3]:
                print(f"      {error}")
        if validation_result.warnings:
            print(f"\n    VALIDATION WARNINGS ({len(validation_result.warnings)}):")
            for warning in validation_result.warnings[:2]:
                print(f"      {warning}")

        if validation_result.is_valid and retry_stats['success']:
            valid_candidates += 1
            if adjusted_score > best_score:
                best_score, best_layout, best_score_card = adjusted_score, layout, card
                best_validation = validation_result
                best_gen = gen
                best_stats = retry_stats
                print(f"    *** New best VALID layout! ***")
                print(f"        Vastu: {card['Vastu Score']:.2f}, Aesthetic: {card['Aesthetic Score']:.2f}")
        else:
            if not validation_result.is_valid:
                print(f"     Rejected: validation failures")
            elif not retry_stats['success']:
                print(f"     Rejected: CP-SAT never succeeded")

    print(f"\n{valid_candidates}/{NUM_CANDIDATES} candidates passed validation")

    if best_layout:
        print("\n--- Generation Complete ---")
        print(f"Best layout found with a total score of: {best_score:.2f}")
        print(f"Final Vastu relaxation level: {best_stats['final_relaxation_level']}/4")
        print(f"Total configurations attempted: {best_stats['configurations_tried']}")

        print(best_validation)

        final_gen = FloorPlanGenerator(reqs)
        final_gen.placed_rooms = best_layout
        final_gen.plot_bounds = final_gen._get_plot_bounds(reqs)
        final_gen.draw(output_image_path, best_score_card)
        print(f"Image saved to {output_image_path}")

        validator.generate_validation_report(best_layout, validation_report_path)

        log_file = "results_log.txt"
        summary = f"{output_image_path} | Total: {best_score_card['Total Score']:.2f} | Vastu: {best_score_card['Vastu Score']:.2f} | Aesthetic: {best_score_card['Aesthetic Score']:.2f} | Valid: \n"

        print("Do you want to visualise the layout now? (y/n): ", end='')
        choice = input().strip().lower()
        if choice == 'y':
            sendto_viewer(output_image_path)

        try:
            with open(log_file, 'a') as f:
                f.write(summary)
            print(f"Results appended to {log_file}")
            print(summary)
        except Exception as e:
            print(f"Error: Could not write to log file {log_file}. {e}")
    else:
        print("FATAL: Could not generate any VALID, fully connected layouts after all attempts.")
        print("Try:")
        print("  1. Increasing NUM_CANDIDATES")
        print("  2. Relaxing validation thresholds in VALIDATION_CONFIG")
        print("  3. Adjusting room size requirements")