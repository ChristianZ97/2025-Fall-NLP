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
            res_seq = model.wv.most_similar(
                positive=[word_b, word_c], negative=[word_a], topn=1
            )
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

    # Your original evaluation logic, unchanged
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
    Main function to find models, evaluate them, and print a fully aligned results table.
    """
    questions_file = "questions-words.csv"
    model_pattern = r"word2vec_(\d+\.?\d*)\.model"

    model_paths = {}
    all_files = sorted(os.listdir("."))
    for filename in all_files:
        if filename.endswith(".model"):
            match = re.match(model_pattern, filename)
            if match and os.path.isfile(filename):
                ratio = float(match.group(1))
                model_paths[ratio] = filename

    if not model_paths:
        print("No models found matching the pattern 'word2vec_*.model'. Exiting.")
        return

    print(f"Found {len(model_paths)} models to evaluate: {list(model_paths.values())}")

    questions_df = pd.read_csv(questions_file)
    all_results = {}
    model_names = []
    for ratio, path in sorted(model_paths.items()):
        model_name = f"{ratio*100:.1f}% Sample"
        model_names.append(model_name)
        all_results[model_name] = evaluate_model(path, questions_df)

    # --- Print Fully Aligned Markdown Table ---
    print("\n\n--- Final Results Table ---")

    # Dynamically get all categories to display
    categories_to_show = list(questions_df["Category"].unique()) + list(
        questions_df["SubCategory"].unique()
    )

    # Calculate column widths for perfect alignment
    # First column width is the max length of category names
    first_col_width = max(len(cat) for cat in categories_to_show)
    # Other column widths are max of model name length or "xx.xx %"
    other_col_widths = [max(len(name), 8) for name in model_names]

    # Header
    header = f"| {'Category / Sub-Category'.ljust(first_col_width)} |"
    for i, name in enumerate(model_names):
        header += f" {name.center(other_col_widths[i])} |"
    print(header)

    # Separator
    separator = f"|{'-' * (first_col_width + 2)}|"
    for width in other_col_widths:
        separator += f"{'-' * (width + 2)}|"
    print(separator)

    # Body
    for category in categories_to_show:
        row = f"| {category.ljust(first_col_width)} |"
        for i, model_name in enumerate(model_names):
            accuracy = all_results[model_name].get(category, 0.0)
            acc_str = f"{accuracy:.2f}%"
            row += f" {acc_str.center(other_col_widths[i])} |"
        print(row)


if __name__ == "__main__":
    main()
