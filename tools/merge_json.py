import os
import json


def merge_json_files(folder_path, output_json):
    merged_data = {}

    # Walk through folder and subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                # Read JSON file with error handling
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                        # Use the file name without extension as the key
                        key = os.path.splitext(file)[0]

                        # Convert total_params to millions and average_time to milliseconds
                        if 'total_params' in data:
                            data['total_params'] = data['total_params'] / 1e6  # Convert to millions
                        if 'average_time' in data:
                            data['average_time'] = data['average_time'] * 1e3  # Convert to milliseconds

                        merged_data[key] = data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")

    # Write merged data to the specified output JSON file
    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=4)


# Example usage
output_json = r'D:\Code\pycode\Data_All\Database_of_CV\Experiment\analysis\show\merge_.json'
merge_json_files(r'D:\Code\pycode\Data_All\Database_of_CV\Experiment\analysis\show', output_json)