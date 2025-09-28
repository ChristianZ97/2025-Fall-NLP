import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import re

def evaluate_model(model_path: str, questions_df: pd.DataFrame):
    """
    Evaluates a single Word2Vec model on the analogy task.
    Returns a dictionary of accuracies for each category and sub-category.
    """
    print(f"\n--- Evaluating model: {os.path.basename(model_path)} ---")
    model = Word2Vec.load(model_path)
    
    preds = []
    golds = []

    for analogy in tqdm(questions_df["Question"], desc="Processing analogies"):
        word_a, word_b, word_c, word_d = analogy.split()
        golds.append(word_d)
        
        vocab = model.wv.key_to_index
        if not all(word in vocab for word in [word_a, word_b, word_c]):
            preds.append(None)
            continue

        try:
            res_seq = model.wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=1)
            pred = res_seq[0][0]
            preds.append(pred)
        except Exception:
            preds.append(None)

    accuracies = {}
    golds_np, preds_np = np.array(golds), np.array(preds)

    def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
        valid_indices = pred != None
        if np.sum(valid_indices) == 0:
            return 0.0
        return np.mean(gold[valid_indices] == pred[valid_indices])

    for category in questions_df["Category"].unique():
        mask = questions_df["Category"] == category
        golds_cat, preds_cat = golds_np[mask], preds_np[mask]
        acc_cat = calculate_accuracy(golds_cat, preds_cat)
        accuracies[category] = acc_cat * 100

    for sub_category in questions_df["SubCategory"].unique():
        mask = questions_df["SubCategory"] == sub_category
        golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
        acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
        accuracies[sub_category] = acc_subcat * 100
        
    return accuracies


def main():
    """
    Main function to find models, evaluate them, and print a results table.
    """
    # --- Configuration ---
    questions_file = "questions-words.csv"
    model_pattern = r"word2vec_(\d+\.?\d*)\.model"
    
    # --- Find Models (Corrected Logic) ---
    model_paths = {}
    # First, get a list of all files in the directory
    all_files = sorted(os.listdir('.'))
    
    # Iterate through files and find the main model files
    for filename in all_files:
        # We are looking for files that END with ".model", not ones that have ".model" in the middle
        if filename.endswith(".model"):
            match = re.match(model_pattern, filename)
            if match:
                # Ensure it's a file and not a directory
                if os.path.isfile(filename):
                    ratio = float(match.group(1))
                    model_paths[ratio] = filename
    
    if not model_paths:
        print("No models found matching the pattern 'word2vec_*.model'. Exiting.")
        return
        
    print(f"Found {len(model_paths)} models to evaluate: {list(model_paths.values())}")
    
    # --- Load Data and Evaluate ---
    questions_df = pd.read_csv(questions_file)
    all_results = {}
    for ratio, path in sorted(model_paths.items()):
        all_results[f"{ratio*100:.1f}% Sample"] = evaluate_model(path, questions_df)

    # --- Print Markdown Table ---
    print("\n\n--- Final Results Table ---")
    
    categories_to_show = [
        "Semantic", "Syntactic", "family", "capital-world", 
        "gram3-comparative", "gram7-past-tense", "gram8-plural"
    ]
    
    header = "| Category / Sub-Category     | " + " | ".join(all_results.keys()) + " |"
    separator = "| --------------------------- |" + " :--------: |" * len(all_results)
    print(header)
    print(separator)

    for category in categories_to_show:
        row = f"| {category.ljust(27)} |"
        for model_name in all_results.keys():
            accuracy = all_results[model_name].get(category, 0.0)
            row += f" {accuracy:.2f}%    |"
        print(row)

if __name__ == "__main__":
    main()
