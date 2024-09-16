import json
import pandas as pd

# Load JSONL files into pandas DataFrames
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Load the first JSONL file
file_path_1 = '/fsx_0/user/jiuhai/data/eval/cv-bench/llava-cv-bench-3D.jsonl'
data1 = load_jsonl(file_path_1)

# Load the second JSONL file
file_path_2 = '/data/home/jiuhai/llama3-mlp3x/scripts/v1_5/eval/playground/data/eval/cv-bench/test.jsonl'
data2 = load_jsonl(file_path_2)


# Initialize accuracy counter
accuracy = 0
# Create a dictionary from the second DataFrame for faster lookups
data2_dict = {entry['question_id']: entry for entry in data2}

# Iterate through the first DataFrame and compare with the second
for a, b in zip(data1, data2):
    assert a['question_id'] == b['question_id']
    if a['answer'] == b['text'] or a['answer'] in b['text'] or b['text'] in a['answer']:
        accuracy += 1
    else:
        print(a['answer'], b['text'])


# Calculate accuracy
total_entries = len(data1)
accuracy_percentage = (accuracy / total_entries) * 100

print(f"Accuracy: {accuracy_percentage:.2f}%")

# If you want to save the results to a file
results = {
    "total_entries": total_entries,
    "matches": accuracy,
    "accuracy_percentage": accuracy_percentage
}

with open('results.json', 'w') as results_file:
    json.dump(results, results_file)

print(f"Results saved to results.json")






import json
import argparse

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def main(file_path_1, file_path_2):
    # Load the first JSON file
    data1 = load_json(file_path_1)

    # Load the second JSON file
    data2 = load_json(file_path_2)

    # Initialize accuracy counter
    accuracy = 0

    # Ensure data is a list of dictionaries
    if not (isinstance(data1, list) and all(isinstance(d, dict) for d in data1)):
        raise ValueError("Data1 is not a list of dictionaries")
    if not (isinstance(data2, list) and all(isinstance(d, dict) for d in data2)):
        raise ValueError("Data2 is not a list of dictionaries")

    # Compare entries and calculate accuracy
    for a, b in zip(data1, data2):
        assert a['question_id'] == b['question_id'], f"Mismatched question_id: {a['question_id']} != {b['question_id']}"
        question_id = a['question_id']
        text = a['text']
        
        if b['answer'] == text:
            accuracy += 1

    # Calculate accuracy
    total_entries = len(data1)
    accuracy_percentage = (accuracy / total_entries) * 100

    print(f"Accuracy: {accuracy_percentage:.2f}%")

    # If you want to save the results to a file
    results = {
        "total_entries": total_entries,
        "matches": accuracy,
        "accuracy_percentage": accuracy_percentage
    }

    with open('results.json', 'w') as results_file:
        json.dump(results, results_file)

    print(f"Results saved to results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two JSON files and calculate accuracy.")
    parser.add_argument("file_path_1", type=str, help="Path to the first JSON file.")
    parser.add_argument("file_path_2", type=str, help="Path to the second JSON file.")
    args = parser.parse_args()
    
    main(args.file_path_1, args.file_path_2)
