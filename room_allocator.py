import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class RoomAllocation:
    name: str
    type: str
    original_width: float
    original_height: float
    allocated_width: float
    allocated_height: float
    is_user_specified: bool
    scaling_factor: float

    @property
    def original_area(self):
        return self.original_width * self.original_height

    @property
    def allocated_area(self):
        return self.allocated_width * self.allocated_height

class SmartRoomAllocator:

    def __init__(self,
                 plot_width: float,
                 plot_height: float,
                 target_utilization: float = 0.95,
                 min_utilization: float = 0.30,
                 max_utilization: float = 1.0):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.plot_area = plot_width * plot_height
        self.target_utilization = target_utilization
        self.min_utilization = min_utilization
        self.max_utilization = max_utilization

    def allocate(self, rooms_data: List[Dict], default_dims: Dict) -> Tuple[List[RoomAllocation], Dict]:
        allocations = []
        metrics = {}

        all_rooms_data = list(rooms_data)
        existing_types = {r.get("type", "").replace('_', ' ').title() for r in rooms_data}

        if any("Living" in t for t in existing_types):
            existing_types.add("Living Room")

        essential_rooms = []
        if "Living Room" not in existing_types:
            essential_rooms.append({"type": "Living Room", "count": 1, "width": None, "height": None})
        if "Kitchen" not in existing_types:
            essential_rooms.append({"type": "Kitchen", "count": 1, "width": None, "height": None})
        if "Dining Room" not in existing_types:
            essential_rooms.append({"type": "Dining Room", "count": 1, "width": None, "height": None})

        bedroom_count = sum(r.get("count", 1) for r in all_rooms_data
                           if "Bedroom" in r.get("type", "").replace('_', ' ').title())
        bathroom_count = sum(r.get("count", 1) for r in all_rooms_data
                            if "Bathroom" in r.get("type", "").replace('_', ' ').title())

        if bedroom_count > 0 and bathroom_count == 0:
            needed_bathrooms = max(1, (bedroom_count + 1) // 2)
            essential_rooms.append({"type": "Bathroom", "count": needed_bathrooms, "width": None, "height": None})

        all_rooms_data.extend(essential_rooms)

        user_specified = []
        flexible = []

        for room_spec in all_rooms_data:
            room_type = room_spec.get("type", "").replace('_', ' ').title()
            if "Living" in room_type:
                room_type = "Living Room"

            count = room_spec.get("count", 1)
            width = room_spec.get("width")
            height = room_spec.get("length")

            for i in range(count):
                name = f"{room_spec.get('name', room_type)} {i+1}" if count > 1 else room_spec.get('name', room_type)

                if width is not None and height is not None:

                    user_specified.append({
                        'name': name,
                        'type': room_type,
                        'width': width,
                        'height': height,
                        'is_user_specified': True
                    })
                else:

                    default_w, default_h = default_dims.get(room_type, (10, 10))
                    flexible.append({
                        'name': name,
                        'type': room_type,
                        'width': default_w,
                        'height': default_h,
                        'is_user_specified': False
                    })

        locked_area = sum(r['width'] * r['height'] for r in user_specified)
        metrics['locked_area'] = locked_area
        metrics['locked_room_count'] = len(user_specified)

        target_total_area = self.plot_area * self.target_utilization
        available_for_flexible = target_total_area - locked_area
        metrics['target_total_area'] = target_total_area
        metrics['available_for_flexible'] = available_for_flexible

        current_flexible_area = sum(r['width'] * r['height'] for r in flexible)
        metrics['current_flexible_area'] = current_flexible_area

        if available_for_flexible <= 0:

            metrics['status'] = 'IMPOSSIBLE'
            metrics['error'] = f"User-specified rooms require {locked_area:.0f} sq units, but target area is only {target_total_area:.0f} sq units"
            return [], metrics

        max_growth_buffer = 1.2
        raw_scaling = math.sqrt(available_for_flexible / current_flexible_area) if current_flexible_area > 0 else 1.0
        scaling_factor = min(raw_scaling, max_growth_buffer)
        metrics['flexible_scaling_factor'] = scaling_factor

        total_area_with_scaling = locked_area + (current_flexible_area * scaling_factor * scaling_factor)
        actual_utilization = total_area_with_scaling / self.plot_area
        metrics['actual_utilization'] = actual_utilization

        if actual_utilization < self.min_utilization:
            metrics['warning'] = f"Low utilization: {actual_utilization*100:.1f}%. Rooms will be very small or sparse."
        elif actual_utilization > self.max_utilization:
            metrics['warning'] = f"High utilization: {actual_utilization*100:.1f}%. Layout may be cramped."

        for room in user_specified:
            allocations.append(RoomAllocation(
                name=room['name'],
                type=room['type'],
                original_width=room['width'],
                original_height=room['height'],
                allocated_width=room['width'],
                allocated_height=room['height'],
                is_user_specified=True,
                scaling_factor=1.0
            ))

        for room in flexible:

            new_width = round((room['width'] * scaling_factor) * 2) / 2
            new_height = round((room['height'] * scaling_factor) * 2) / 2

            allocations.append(RoomAllocation(
                name=room['name'],
                type=room['type'],
                original_width=room['width'],
                original_height=room['height'],
                allocated_width=new_width,
                allocated_height=new_height,
                is_user_specified=False,
                scaling_factor=scaling_factor
            ))

        metrics['status'] = 'SUCCESS'
        metrics['total_allocated_area'] = sum(a.allocated_area for a in allocations)
        metrics['room_count'] = len(allocations)

        return allocations, metrics

    def print_allocation_summary(self, allocations: List[RoomAllocation], metrics: Dict):
        print("\n" + "="*70)
        print("SMART ROOM ALLOCATION SUMMARY")
        print("="*70)

        if metrics.get('status') == 'IMPOSSIBLE':
            print(f"\nERROR: {metrics.get('error')}")
            return

        print(f"\nPlot: {self.plot_width}x{self.plot_height} = {self.plot_area:.0f} sq units")
        print(f"Target Utilization: {self.target_utilization*100:.0f}%")
        print(f"Actual Utilization: {metrics['actual_utilization']*100:.1f}%")

        if 'warning' in metrics:
            print(f"\nWARNING: {metrics['warning']}")

        print(f"\nALLOCATION BREAKDOWN:")
        print(f"  Locked (user-specified): {metrics['locked_room_count']} rooms, {metrics['locked_area']:.0f} sq units")
        print(f"  Flexible (default): {metrics['room_count'] - metrics['locked_room_count']} rooms")
        print(f"  Flexible scaling factor: {metrics['flexible_scaling_factor']:.2f}x")
        print(f"  Total allocated: {metrics['total_allocated_area']:.0f} sq units")

        user_spec = [a for a in allocations if a.is_user_specified]
        flex = [a for a in allocations if not a.is_user_specified]

        if user_spec:
            print(f"\nUSER-SPECIFIED ROOMS (unchanged):")
            for alloc in user_spec:
                print(f"  {alloc.name}: {alloc.allocated_width:.1f}x{alloc.allocated_height:.1f} = {alloc.allocated_area:.0f} sq units")

        if flex:
            print(f"\nFLEXIBLE ROOMS (scaled {metrics['flexible_scaling_factor']:.2f}x):")
            for alloc in flex:
                print(f"  {alloc.name}: {alloc.original_width:.1f}x{alloc.original_height:.1f} -> "
                      f"{alloc.allocated_width:.1f}x{alloc.allocated_height:.1f} "
                      f"({alloc.original_area:.0f} -> {alloc.allocated_area:.0f} sq units)")

        print("="*70 + "\n")

def allocate_room_sizes(requirements_data: Dict, default_dims: Dict) -> Tuple[List[RoomAllocation], Dict]:
    plot_width = requirements_data.get("house_dimensions", {}).get("width", 40)
    plot_height = requirements_data.get("house_dimensions", {}).get("length", 40)

    allocator = SmartRoomAllocator(plot_width, plot_height)
    allocations, metrics = allocator.allocate(
        requirements_data.get("rooms", []),
        default_dims
    )

    allocator.print_allocation_summary(allocations, metrics)

    return allocations, metrics