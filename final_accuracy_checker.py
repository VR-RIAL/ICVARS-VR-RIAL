import json

def get_room_count(rooms_dict, room_type):
    total_count = 0
    for room_name, room_details in rooms_dict.items():
        if room_type in room_name.lower():
            if 'count' in room_details:
                total_count += room_details['count']
            elif isinstance(room_details.get('dimensions'), list):
                total_count += len(room_details['dimensions'])
            else:

                total_count += 1
    return total_count

def compare_requirements(requirements_file, image_analysis_file):
    with open(requirements_file, 'r') as f:
        requirements = json.load(f)
    with open(image_analysis_file, 'r') as f:
        image_analysis = json.load(f)

    accuracy_score = 0
    total_requirements = 0
    mismatched_features = []

    if 'house_dimensions' in requirements and 'house_dimensions' in image_analysis:
        total_requirements += 1
        req_dims = requirements['house_dimensions']
        img_dims = image_analysis['house_dimensions']

        if 'width' in req_dims and 'length' in req_dims:
            if img_dims.get('width', 0) <= req_dims['width'] and \
               img_dims.get('length', 0) <= req_dims['length']:
                accuracy_score += 1
            else:
                mismatched_features.append({
                    'feature': 'house_dimensions',
                    'required': f"Less than or equal to {req_dims['width']}x{req_dims['length']}",
                    'found': f"{img_dims.get('width', 0)}x{img_dims.get('length', 0)}"
                })

    if 'rooms' in requirements and 'rooms' in image_analysis:
        required_rooms_list = requirements['rooms']
        found_rooms_dict = image_analysis['rooms']

        for req_room in required_rooms_list:
            req_type = req_room.get('type')
            req_count = req_room.get('count', 0)

            if not req_type:
                continue

            if req_type == 'bedroom':
                total_requirements += 4
            else:
                total_requirements += 1

            found_count = get_room_count(found_rooms_dict, req_type)

            if found_count >= req_count:
                if req_type == 'bedroom':
                    accuracy_score += 4
                else:
                    accuracy_score += 1
            else:
                mismatched_features.append({
                    'feature': f'{req_type} count',
                    'required': req_count,
                    'found': found_count
                })

            if 'features' in req_room:
                for feature in req_room['features']:
                    total_requirements += 1
                    if feature in str(image_analysis):
                        accuracy_score += 1
                    else:
                        mismatched_features.append({
                            'feature': f'{req_type} feature: {feature}',
                            'required': True,
                            'found': False
                        })

    if 'house_dimensions' in requirements and 'features' in requirements['house_dimensions']:
        for feature in requirements['house_dimensions']['features']:
            total_requirements += 1
            if feature in str(image_analysis):
                accuracy_score += 1
            else:
                mismatched_features.append({
                    'feature': f'house feature: {feature}',
                    'required': True,
                    'found': False
                })

    accuracy_percentage = (accuracy_score / total_requirements) * 100 if total_requirements > 0 else 0

    return {
        'accuracy_percentage': accuracy_percentage,
        'mismatched_features': mismatched_features
    }

if __name__ == '__main__':
    results = compare_requirements('Inputs/{number}/requirements.json', 'Inputs/{number}/image_analysis.json')
    print(f"Accuracy: {results['accuracy_percentage']:.2f}%")
    if results['mismatched_features']:
        print("\nMismatched Features:")
        for mismatch in results['mismatched_features']:
            print(f"  - Feature: {mismatch['feature']}")
            print(f"    Required: {mismatch['required']}")
            print(f"    Found: {mismatch['found']}")