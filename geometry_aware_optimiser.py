from ortools.sat.python import cp_model
import math

class GeometryAwareOptimizer:

    def __init__(self,
                 plot_width: float,
                 plot_height: float,
                 aspect_ratio_weight: float = 1.0,
                 movement_weight: float = 1.0,
                 directional_preference_weight: float = 50.0):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.plot_aspect = plot_width / plot_height if plot_height > 0 else 1.0
        self.aspect_ratio_weight = aspect_ratio_weight
        self.movement_weight = movement_weight
        self.directional_preference_weight = directional_preference_weight

        self.is_vertical = plot_height > plot_width
        self.is_horizontal = plot_width > plot_height
        self.is_square = abs(plot_width - plot_height) < 1.0

        if self.is_vertical:
            self.grid_x = 1.0
            self.grid_y = 0.5
        elif self.is_horizontal:
            self.grid_x = 0.5
            self.grid_y = 1.0
        else:
            self.grid_x = self.grid_y = 0.5

    def optimize_layout(self, generator, time_limit: int = 30, max_coord: int = 300):
        grid_res = getattr(generator, 'GRID_SIZE', 0.5)
        scale = int(1.0 / grid_res)

        rooms = [(r.name, r.x, r.y, r.width, r.height,
                r.min_width, r.max_width, r.min_height, r.max_height)
                for r in generator.placed_rooms if r.placed]

        if not rooms:
            return False

        model = cp_model.CpModel()
        x, y, width, height = {}, {}, {}, {}

        max_x_limit = int(self.plot_width * scale)
        max_y_limit = int(self.plot_height * scale)

        for name, ox, oy, ow, oh, min_w, max_w, min_h, max_h in rooms:
            x[name] = model.NewIntVar(0, max_x_limit, f"x_{name}")
            y[name] = model.NewIntVar(0, max_y_limit, f"y_{name}")

            width[name] = model.NewIntVar(int(min_w * scale), int(max_w * scale), f"w_{name}")
            height[name] = model.NewIntVar(int(min_h * scale), int(max_h * scale), f"h_{name}")

            model.Add(x[name] + width[name] <= max_x_limit)
            model.Add(y[name] + height[name] <= max_y_limit)

        x_intervals, y_intervals = [], []
        for name, _, _, _, _, _, _, _, _ in rooms:
            x_end = model.NewIntVar(0, max_x_limit, f"x_end_{name}")
            y_end = model.NewIntVar(0, max_y_limit, f"y_end_{name}")
            model.Add(x_end == x[name] + width[name])
            model.Add(y_end == y[name] + height[name])
            x_intervals.append(model.NewIntervalVar(x[name], width[name], x_end, f"x_int_{name}"))
            y_intervals.append(model.NewIntervalVar(y[name], height[name], y_end, f"y_int_{name}"))

        model.AddNoOverlap2D(x_intervals, y_intervals)

        min_x_bb = model.NewIntVar(0, max_x_limit, "min_x_bb")
        max_x_bb = model.NewIntVar(0, max_x_limit, "max_x_bb")
        min_y_bb = model.NewIntVar(0, max_y_limit, "min_y_bb")
        max_y_bb = model.NewIntVar(0, max_y_limit, "max_y_bb")

        model.AddMinEquality(min_x_bb, [x[n] for n in x])
        model.AddMaxEquality(max_x_bb, [x[n] + width[n] for n in x])
        model.AddMinEquality(min_y_bb, [y[n] for n in y])
        model.AddMaxEquality(max_y_bb, [y[n] + height[n] for n in y])

        bb_w = model.NewIntVar(0, max_x_limit, "bb_w")
        bb_h = model.NewIntVar(0, max_y_limit, "bb_h")
        model.Add(bb_w == max_x_bb - min_x_bb)
        model.Add(bb_h == max_y_bb - min_y_bb)

        area = model.NewIntVar(0, max_x_limit * max_y_limit, "area")
        model.AddMultiplicationEquality(area, [bb_w, bb_h])

        move_costs = []
        for name, ox, oy, _, _, _, _, _, _ in rooms:
            dx = model.NewIntVar(0, max_x_limit, f"dx_{name}")
            dy = model.NewIntVar(0, max_y_limit, f"dy_{name}")

            model.AddAbsEquality(dx, x[name] - int(ox * scale))
            model.AddAbsEquality(dy, y[name] - int(oy * scale))
            move_costs.append(dx)
            move_costs.append(dy)

        model.Minimize(area + (sum(move_costs) * 50))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for room in generator.placed_rooms:
                if room.name in x:

                    nx = round((solver.Value(x[room.name]) / scale) * 2) / 2
                    ny = round((solver.Value(y[room.name]) / scale) * 2) / 2
                    nw = round((solver.Value(width[room.name]) / scale) * 2) / 2
                    nh = round((solver.Value(height[room.name]) / scale) * 2) / 2
                    room.set_position(nx, ny)
                    room.set_dimensions(nw, nh)

            generator._cp_sat_success = True
            return True

        return False

class AxisAwareCompactor:

    def __init__(self, plot_width: float, plot_height: float, grid_size: float = 0.5):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.grid_size = grid_size

        self.is_vertical = plot_height > plot_width
        self.primary_axis = 'y' if self.is_vertical else 'x'
        self.secondary_axis = 'x' if self.is_vertical else 'y'

    def compact(self, generator, max_iterations: int = 150):
        print(f"\nAxis-aware compaction (plot: {self.plot_width}x{self.plot_height}, primary: {self.primary_axis})...")

        for i in range(max_iterations):
            layout_cx, layout_cy = generator._get_layout_center()
            rooms = sorted(
                generator.placed_rooms,
                key=lambda r: math.hypot(r.center[0] - layout_cx, r.center[1] - layout_cy),
                reverse=True
            )

            moved = False

            for room in rooms:
                dx = layout_cx - room.center[0]
                dy = layout_cy - room.center[1]

                if self.primary_axis == 'y':

                    if self._try_move(generator, room, 0, self.grid_size if dy > 0 else -self.grid_size):
                        moved = True
                        continue
                    if self._try_move(generator, room, self.grid_size if dx > 0 else -self.grid_size, 0):
                        moved = True
                else:

                    if self._try_move(generator, room, self.grid_size if dx > 0 else -self.grid_size, 0):
                        moved = True
                        continue
                    if self._try_move(generator, room, 0, self.grid_size if dy > 0 else -self.grid_size):
                        moved = True

            if not moved:
                print(f"  Compaction converged after {i+1} steps")
                break

    def _try_move(self, generator, room, dx, dy):
        ox, oy = room.x, room.y
        room.set_position(ox + dx, oy + dy)

        if any(room.collides_with(o) for o in generator.placed_rooms if o is not room):
            room.set_position(ox, oy)
            return False

        if not any(room.is_adjacent_to(o) for o in generator.placed_rooms if o is not room):
            room.set_position(ox, oy)
            return False

        return True

class AxisBiasedGapFiller:

    def __init__(self, plot_width: float, plot_height: float, grid_size: float = 0.5):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.grid_size = grid_size

    def fill_gaps(self, generator, max_iterations: int = 50):
        step_sizes = [self.grid_size * 4, self.grid_size * 2, self.grid_size]

        for step_size in step_sizes:
            for iteration in range(max_iterations):

                placed = [r for r in generator.placed_rooms if r.placed]
                if not placed:
                    continue

                bbox_width = max(r.x + r.width for r in placed) - min(r.x for r in placed)
                bbox_height = max(r.y + r.height for r in placed) - min(r.y for r in placed)

                if bbox_width < bbox_height:

                    priority_dirs = ['right', 'left', 'down', 'up']
                else:

                    priority_dirs = ['down', 'up', 'right', 'left']

                expanded_any = False

                for room in generator.placed_rooms:
                    if room.type == "Hallway":
                        continue

                    for direction in priority_dirs:
                        if self._try_expand(generator, room, direction, step_size):
                            expanded_any = True
                            break

                if not expanded_any:
                    break

        print(f"✓ Gap filling complete (axis-biased)")

    def _try_expand(self, generator, room, direction, step_size):
        orig_x, orig_y = room.x, room.y
        orig_width, orig_height = room.width, room.height

        if direction == 'right':
            new_width = room.width + step_size
            if room.x + new_width > self.plot_width or new_width > room.max_width:
                return False
            room.width = new_width
        elif direction == 'left':
            new_x = room.x - step_size
            if new_x < 0:
                return False
            new_width = room.width + step_size
            if new_width > room.max_width:
                return False
            room.x = new_x
            room.width = new_width
        elif direction == 'down':
            new_height = room.height + step_size
            if room.y + new_height > self.plot_height or new_height > room.max_height:
                return False
            room.height = new_height
        elif direction == 'up':
            new_y = room.y - step_size
            if new_y < 0:
                return False
            new_height = room.height + step_size
            if new_height > room.max_height:
                return False
            room.y = new_y
            room.height = new_height

        for other in generator.placed_rooms:
            if other is room:
                continue
            if room.collides_with(other, tolerance=0.05):
                room.x, room.y = orig_x, orig_y
                room.width, room.height = orig_width, orig_height
                return False

        has_adjacent = any(room.is_adjacent_to(other) for other in generator.placed_rooms if other is not room)
        if not has_adjacent and room.type != "Living Room":
            room.x, room.y = orig_x, orig_y
            room.width, room.height = orig_width, orig_height
            return False

        return True