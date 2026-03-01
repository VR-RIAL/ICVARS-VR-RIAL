import hashlib
import json
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

class VastuRelaxationManager:

    def __init__(self, original_vastu_prefs: Dict[str, List[str]]):
        self.original_prefs = {k: list(v) for k, v in original_vastu_prefs.items()}
        self.current_prefs = {k: list(v) for k, v in original_vastu_prefs.items()}
        self.relaxation_level = 0
        self.tried_configurations = set()
        self.max_relaxation_levels = 4

        self.zone_expansion_order = {
            0: [],
            1: self._get_adjacent_zones,
            2: self._get_cardinal_zones,
            3: self._get_all_zones,
        }

    def _get_adjacent_zones(self, preferred_zones: List[str]) -> List[str]:
        ZONE_ADJACENCY = {
            'N': ['NE', 'NW', 'Center'],
            'S': ['SE', 'SW', 'Center'],
            'E': ['NE', 'SE', 'Center'],
            'W': ['NW', 'SW', 'Center'],
            'NE': ['N', 'E'],
            'NW': ['N', 'W'],
            'SE': ['S', 'E'],
            'SW': ['S', 'W'],
            'Center': ['N', 'S', 'E', 'W']
        }

        expanded = set(preferred_zones)
        for zone in preferred_zones:
            expanded.update(ZONE_ADJACENCY.get(zone, []))

        return list(expanded)

    def _get_cardinal_zones(self, preferred_zones: List[str]) -> List[str]:
        return ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'Center']

    def _get_all_zones(self, preferred_zones: List[str]) -> List[str]:
        return ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'Center']

    def get_current_preferences(self) -> Dict[str, List[str]]:
        return self.current_prefs.copy()

    def relax_preferences(self) -> bool:
        if self.relaxation_level >= self.max_relaxation_levels:
            return False

        self.relaxation_level += 1
        expansion_func = self.zone_expansion_order.get(self.relaxation_level)

        if expansion_func:

            for room_type, original_zones in self.original_prefs.items():
                if callable(expansion_func):
                    self.current_prefs[room_type] = expansion_func(original_zones)
                else:
                    self.current_prefs[room_type] = original_zones

            print(f"\nRelaxing Vastu preferences (Level {self.relaxation_level}/{self.max_relaxation_levels})")
            self._print_relaxation_summary()
            return True

        return False

    def _print_relaxation_summary(self):
        changes = []
        for room_type, new_prefs in self.current_prefs.items():
            old_prefs = self.original_prefs[room_type]
            if set(new_prefs) != set(old_prefs):
                added = set(new_prefs) - set(old_prefs)
                if added:
                    changes.append(f"  {room_type}: {old_prefs} -> added {sorted(added)}")

        if changes:
            print("  Changes:")
            for change in changes[:5]:
                print(change)
            if len(changes) > 5:
                print(f"  ... and {len(changes) - 5} more")

    def record_attempt(self, room_placements: List[Tuple[str, int, int]]) -> bool:

        config_str = json.dumps(sorted(room_placements), sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        if config_hash in self.tried_configurations:
            return False

        self.tried_configurations.add(config_hash)
        return True

    def reset_to_level(self, level: int):
        self.relaxation_level = min(level, self.max_relaxation_levels)
        expansion_func = self.zone_expansion_order.get(self.relaxation_level)

        if expansion_func:
            for room_type, original_zones in self.original_prefs.items():
                if callable(expansion_func):
                    self.current_prefs[room_type] = expansion_func(original_zones)
                else:
                    self.current_prefs[room_type] = original_zones

    def get_stats(self) -> Dict:
        return {
            'relaxation_level': self.relaxation_level,
            'max_levels': self.max_relaxation_levels,
            'configurations_tried': len(self.tried_configurations),
            'can_relax_more': self.relaxation_level < self.max_relaxation_levels
        }

class RetryableFloorPlanGenerator:

    def __init__(self, base_generator_class, requirements_data, original_vastu_prefs):
        self.base_class = base_generator_class
        self.requirements = requirements_data
        self.vastu_manager = VastuRelaxationManager(original_vastu_prefs)
        self.max_retries_per_level = 3
        self.max_total_retries = 15

    def generate_with_retries(self) -> Tuple[Optional[object], Dict]:
        total_attempts = 0
        stats = {
            'attempts_per_level': defaultdict(int),
            'cp_sat_failures': 0,
            'cp_sat_successes': 0,
            'final_relaxation_level': 0,
            'success': False
        }

        while total_attempts < self.max_total_retries:

            current_prefs = self.vastu_manager.get_current_preferences()
            relaxation_level = self.vastu_manager.relaxation_level

            print(f"\n{'='*70}")
            print(f"ATTEMPT {total_attempts + 1}/{self.max_total_retries} (Relaxation Level: {relaxation_level})")
            print(f"{'='*70}")

            for retry in range(self.max_retries_per_level):
                total_attempts += 1
                stats['attempts_per_level'][relaxation_level] += 1

                generator = self.base_class(self.requirements)

                for room in generator.rooms:
                    if room.type in current_prefs:
                        room.vastu_prefs = current_prefs[room.type]

                print(f"\n  Generating layout (retry {retry + 1}/{self.max_retries_per_level} at level {relaxation_level})...")
                layout = generator.generate_layout()

                if not all(r.placed for r in generator.rooms):
                    print(f"  Layout generation incomplete, retrying...")
                    continue

                placements = [(r.name, int(r.x), int(r.y)) for r in generator.placed_rooms]
                is_new_config = self.vastu_manager.record_attempt(placements)

                if not is_new_config:
                    print(f"  Configuration already tried, generating new variant...")
                    continue

                if hasattr(generator, '_cp_sat_success') and generator._cp_sat_success:
                    print(f"  CP-SAT optimization succeeded!")
                    stats['cp_sat_successes'] += 1
                    stats['final_relaxation_level'] = relaxation_level
                    stats['success'] = True
                    stats.update(self.vastu_manager.get_stats())

                    return generator, stats
                else:
                    print(f"  CP-SAT optimization failed")
                    stats['cp_sat_failures'] += 1

            if not self.vastu_manager.relax_preferences():
                print(f"\nAll relaxation levels exhausted!")
                break

        print(f"\nFailed to generate valid layout after {total_attempts} attempts")
        stats.update(self.vastu_manager.get_stats())
        return None, stats

def create_retryable_generator(generator_class, requirements_data, vastu_prefs):
    return RetryableFloorPlanGenerator(generator_class, requirements_data, vastu_prefs)