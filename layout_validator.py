import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

    def __str__(self):
        status = "VALID" if self.is_valid else "INVALID"
        output = [f"\n{'='*60}", f"Layout Validation: {status}", f"{'='*60}"]

        if self.errors:
            output.append("\nERRORS:")
            for err in self.errors:
                output.append(f"  • {err}")

        if self.warnings:
            output.append("\nWARNINGS:")
            for warn in self.warnings:
                output.append(f"  • {warn}")

        output.append("\nMETRICS:")
        for key, value in self.metrics.items():
            output.append(f"  • {key}: {value:.2f}")

        output.append(f"{'='*60}\n")
        return "\n".join(output)

class LayoutValidator:

    def __init__(self,
                 min_room_area: float = 36.0,
                 max_room_area: float = 400.0,
                 max_aspect_ratio: float = 3.0,
                 outlier_iqr_multiplier: float = 1.5,
                 area_cv_threshold: float = 0.8,
                 min_dimension: float = 4.0):
        self.min_room_area = min_room_area
        self.max_room_area = max_room_area
        self.max_aspect_ratio = max_aspect_ratio
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.area_cv_threshold = area_cv_threshold
        self.min_dimension = min_dimension

    def validate_requirements(self, requirements_data: dict) -> ValidationResult:
        errors = []
        warnings = []
        metrics = {}

        if "rooms" not in requirements_data:
            return ValidationResult(True, [], [], {})

        rooms_data = requirements_data["rooms"]
        total_rooms = sum(r.get("count", 1) for r in rooms_data)
        metrics['specified_room_count'] = total_rooms

        for room_spec in rooms_data:
            room_type = room_spec.get("type", "Unknown")
            width = room_spec.get("width")
            length = room_spec.get("length")
            count = room_spec.get("count", 1)

            if width is None or length is None:
                continue

            if width < self.min_dimension or length < self.min_dimension:
                errors.append(
                    f"{room_type}: Dimension too small ({width}x{length}). "
                    f"Minimum dimension is {self.min_dimension}"
                )

            area = width * length
            if area < self.min_room_area:
                errors.append(
                    f"{room_type}: Area {area:.1f} sq units is below minimum {self.min_room_area}"
                )
            if area > self.max_room_area:
                errors.append(
                    f"{room_type}: Area {area:.1f} sq units exceeds maximum {self.max_room_area}"
                )

            ratio = max(width, length) / min(width, length)
            if ratio > self.max_aspect_ratio:
                errors.append(
                    f"{room_type}: Aspect ratio {ratio:.1f}:1 (dimensions {width}x{length}) "
                    f"exceeds maximum {self.max_aspect_ratio}:1. Room is too disproportionate!"
                )

        if "house_dimensions" in requirements_data:
            plot_width = requirements_data["house_dimensions"].get("width")
            plot_height = requirements_data["house_dimensions"].get("length")

            if plot_width and plot_height:
                plot_area = plot_width * plot_height

                total_area = 0
                for room_spec in rooms_data:
                    w = room_spec.get("width")
                    h = room_spec.get("length")
                    if w and h:
                        total_area += w * h * room_spec.get("count", 1)

                if total_area > 0:
                    utilization = total_area / plot_area
                    metrics['plot_area'] = plot_area
                    metrics['total_room_area'] = total_area
                    metrics['estimated_utilization'] = utilization

                    if utilization > 0.85:
                        warnings.append(
                            f"Plot may be too small: rooms need {total_area:.0f} sq units, "
                            f"plot is {plot_area:.0f} sq units ({utilization*100:.0f}% utilization)"
                        )

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def validate(self, rooms: List, exclude_hallways: bool = True) -> ValidationResult:
        errors = []
        warnings = []
        metrics = {}

        analysis_rooms = [r for r in rooms if r.placed]
        if exclude_hallways:
            analysis_rooms = [r for r in analysis_rooms if r.type != "Hallway"]

        if not analysis_rooms:
            return ValidationResult(
                is_valid=False,
                errors=["No rooms to validate"],
                warnings=[],
                metrics={}
            )

        areas = [r.width * r.height for r in analysis_rooms]
        metrics['total_rooms'] = len(analysis_rooms)
        metrics['mean_area'] = np.mean(areas)
        metrics['median_area'] = np.median(areas)
        metrics['min_area'] = np.min(areas)
        metrics['max_area'] = np.max(areas)
        metrics['std_area'] = np.std(areas)

        min_area_violations = []
        for room in analysis_rooms:
            area = room.width * room.height
            if area < self.min_room_area:
                min_area_violations.append(
                    f"{room.name}: {area:.2f} sq units (min: {self.min_room_area})"
                )

        if min_area_violations:
            errors.append(f"Rooms below minimum area ({len(min_area_violations)}):")
            errors.extend([f"  - {v}" for v in min_area_violations])

        max_area_violations = []
        for room in analysis_rooms:
            area = room.width * room.height
            if area > self.max_room_area:
                max_area_violations.append(
                    f"{room.name}: {area:.2f} sq units (max: {self.max_room_area})"
                )

        if max_area_violations:
            errors.append(f"Rooms above maximum area ({len(max_area_violations)}):")
            errors.extend([f"  - {v}" for v in max_area_violations])

        aspect_ratio_violations = []
        for room in analysis_rooms:
            ratio = max(room.width, room.height) / min(room.width, room.height)
            if ratio > self.max_aspect_ratio:
                aspect_ratio_violations.append(
                    f"{room.name}: {ratio:.2f}:1 (max: {self.max_aspect_ratio}:1) [{room.width:.1f}x{room.height:.1f}]"
                )

        if aspect_ratio_violations:
            warnings.append(f"Disproportionate rooms ({len(aspect_ratio_violations)}):")
            warnings.extend([f"  - {v}" for v in aspect_ratio_violations])

        mean_median_diff = abs(metrics['mean_area'] - metrics['median_area'])
        mean_median_ratio = mean_median_diff / metrics['median_area']
        metrics['mean_median_diff_pct'] = mean_median_ratio * 100

        if mean_median_ratio > 0.3:
            warnings.append(
                f"Large mean-median divergence: {mean_median_ratio*100:.1f}% "
                f"(mean: {metrics['mean_area']:.1f}, median: {metrics['median_area']:.1f})"
            )

        cv = metrics['std_area'] / metrics['mean_area'] if metrics['mean_area'] > 0 else 0
        metrics['coefficient_of_variation'] = cv

        if cv > self.area_cv_threshold:
            warnings.append(
                f"High area variability: CV = {cv:.2f} (threshold: {self.area_cv_threshold})"
            )

        q1 = np.percentile(areas, 25)
        q3 = np.percentile(areas, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.outlier_iqr_multiplier * iqr
        upper_bound = q3 + self.outlier_iqr_multiplier * iqr

        metrics['q1'] = q1
        metrics['q3'] = q3
        metrics['iqr'] = iqr
        metrics['outlier_lower_bound'] = lower_bound
        metrics['outlier_upper_bound'] = upper_bound

        outliers = []
        for room in analysis_rooms:
            area = room.width * room.height
            if area < lower_bound or area > upper_bound:
                outlier_type = "small" if area < lower_bound else "large"
                outliers.append(
                    f"{room.name}: {area:.2f} sq units ({outlier_type} outlier)"
                )

        if outliers:
            warnings.append(f"Statistical outliers detected ({len(outliers)}):")
            warnings.extend([f"  - {o}" for o in outliers])

        metrics['outlier_count'] = len(outliers)

        if metrics['max_area'] / metrics['min_area'] > 10:
            warnings.append(
                f"Large size disparity: largest room is {metrics['max_area']/metrics['min_area']:.1f}x "
                f"larger than smallest"
            )

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def generate_validation_report(self, rooms: List, output_path: Optional[str] = None) -> ValidationResult:
        result = self.validate(rooms)

        if output_path:
            report = {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'metrics': result.metrics,
                'validation_config': {
                    'min_room_area': self.min_room_area,
                    'max_room_area': self.max_room_area,
                    'max_aspect_ratio': self.max_aspect_ratio,
                    'outlier_iqr_multiplier': self.outlier_iqr_multiplier,
                    'area_cv_threshold': self.area_cv_threshold
                }
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"Validation report saved to: {output_path}")

        return result

def validate_layout_from_json(json_path: str, validator: Optional[LayoutValidator] = None) -> ValidationResult:
    if validator is None:
        validator = LayoutValidator()

    with open(json_path, 'r') as f:
        data = json.load(f)

    class MockRoom:
        def __init__(self, room_data):
            self.name = room_data['name']
            self.type = room_data['type']
            self.width = room_data['width']
            self.height = room_data['length']
            self.placed = True

    rooms = [MockRoom(r) for r in data['rooms']]
    return validator.validate(rooms)