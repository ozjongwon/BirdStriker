# python3 create_class_names.py --data_dir CUB_200_2011
import os
import json
import argparse

def create_class_names_json(data_dir):
    """Create a JSON file mapping class indices to class names."""
    # Load class names
    classes_file = os.path.join(data_dir, 'classes.txt')
    class_names = {}

    with open(classes_file, 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split(' ', 1)
            # Convert to 0-based indexing
            class_idx = int(class_id) - 1
            class_names[str(class_idx)] = class_name

    # Save to JSON
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)

    print(f"Created class_names.json with {len(class_names)} classes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to CUB_200_2011 dataset')

    args = parser.parse_args()
    create_class_names_json(args.data_dir)
