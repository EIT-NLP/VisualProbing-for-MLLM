import pandas as pd
import json
import re
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Process JSON file path and optionally save results.')
    parser.add_argument('--json_path', type=str, help='Path to the JSONL file', required=True)
    parser.add_argument('--result_path', type=str, help='Optional path to save the result file')
    return parser.parse_args()

# Load the JSON file into a DataFrame
def load_data(json_path):
    data = []
    with open(json_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return pd.DataFrame(data)

def filter_text(df):
    def apply_filter(row):
        if row['gpt'] == "Could not determine":
            # 使用 'text' 列进行匹配
            return " ".join(re.findall(r'[a-zA-Z]', row['text']))
        else:
            # 使用 'gpt' 列进行匹配
            return " ".join(re.findall(r'[a-zA-Z]', row['gpt']))
    
    df["filtered_text"] = df.apply(apply_filter, axis=1)
    return df

# Define a function to calculate accuracy for a given source
def calculate_accuracy(df, source):
    source_df = df[df['source'] == source]
    count = 0
    for i in range(len(source_df)):
        if source_df['filtered_text'].iloc[i] in source_df['answer'].iloc[i] and source_df['filtered_text'].iloc[i] != '':
            count += 1
    accuracy = count / len(source_df)  # Assuming 'result' is 1 for correct and 0 for incorrect
    return accuracy

def main():
    args = parse_args()
    df = load_data(args.json_path)
    df = filter_text(df)
    
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(df, 'COCO')
    accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    # Prepare the result string
    result_string = (
        f"CV-Bench Accuracy: {combined_accuracy:.4f}\n\n"
        f"Type Accuracies:\n"
        f"2D Accuracy: {accuracy_2d:.4f}\n"
        f"3D Accuracy: {accuracy_3d:.4f}\n\n"
        f"Source Accuracies:\n"
        f"ADE20K Accuracy: {accuracy_2d_ade:.4f}\n"
        f"COCO Accuracy: {accuracy_2d_coco:.4f}\n"
        f"Omni3D Accuracy: {accuracy_3d_omni:.4f}\n"
    )

    # Check if result_path is provided
    if args.result_path:
        # Save the result to the specified result_path
        print(result_string)
        with open(args.result_path, 'w') as result_file:
            result_file.write(result_string)
    else:
        # Print to the console
        print(result_string)

if __name__ == '__main__':
    main()