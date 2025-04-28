import os
import json
import string
import pandas as pd

DATASET_SPLITS = {
    "mvbench_action_localization", 
    "mvbench_egocentric_navigation",
    "mvbench_moving_direction",
    "mvbench_moving_count",
    "temporalbench_subset",
    "vinoground_video",
    "vinoground_text"
}

def normalize_text(s):
    return ''.join(ch for ch in str(s).lower().strip() if ch not in string.punctuation)

def answers_equal(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return normalize_text(a) == normalize_text(b)

def get_accuracy(csv_file):
    df = pd.read_csv(csv_file)
    df['correct'] = df.apply(lambda row: answers_equal(row['model_answer'], row['groundtruth']), axis=1)
    return df['correct'].mean()

def main(directory):
    results = {}
    for model_name in os.listdir(directory):
        if model_name.startswith('bad'):
            continue
        model_path = os.path.join(directory, model_name)
        if os.path.isdir(model_path):
            accuracies = {}
            for file in os.listdir(model_path):
                if file.endswith('.csv'):
                    csv_path = os.path.join(model_path, file)
                    accuracy = get_accuracy(csv_path)
                    for d in DATASET_SPLITS:
                        if d in file:  
                            dataset = d
                    accuracies[dataset] = accuracy
            if accuracies:
                overall_avg = sum(accuracies.values()) / len(accuracies)
                accuracies['overall_average'] = overall_avg
                results[model_name] = accuracies

    with open('model_accuracies.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Create a DataFrame for CSV output
    # Collect all dataset names
    datasets = set()
    for model_acc in results.values():
        datasets.update(model_acc.keys())
    datasets = sorted(datasets)

    # Create a DataFrame with models as rows and datasets as columns
    df = pd.DataFrame(index=results.keys(), columns=datasets)

    for model, accs in results.items():
        for dataset, acc in accs.items():
            df.at[model, dataset] = acc

    df.to_csv('model_accuracies.csv')

    return results

if __name__ == '__main__':
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(directory)