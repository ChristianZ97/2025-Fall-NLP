# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image:
# https://github.com/kaggle/docker-python

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Inspect available input files under /kaggle/input
# ---------------------------------------------------------------------

# for dirname, _, filenames in os.walk("/kaggle/input"):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# At this point you should see (for this competition):
# /kaggle/input/WattBot2025/test_Q.csv
# /kaggle/input/WattBot2025/train_QA.csv
# /kaggle/input/WattBot2025/metadata.csv

# ---------------------------------------------------------------------
# 2. Read competition data
# ---------------------------------------------------------------------

DATA_DIR = Path("./data/WattBot2025")  # adjust if the folder name differs

train_qa = pd.read_csv(DATA_DIR / "train_QA.csv")
test_q = pd.read_csv(DATA_DIR / "test_Q.csv")

print("train_QA columns:", list(train_qa.columns))
print("test_Q columns:", list(test_q.columns))

# According to the competition description, train_QA.csv and the
# submission file must share the same columns:
# id, question, answer, answer_value, answer_unit,
# ref_id, ref_url, supporting_materials, explanation

expected_columns = list(train_qa.columns)

# ---------------------------------------------------------------------
# 3. Define a dummy baseline that always uses the official fallback
# ---------------------------------------------------------------------

FALLBACK_ANSWER = "Unable to answer with confidence based on the provided documents."


def make_dummy_row(row):
    """
    Given a row from test_Q.csv, build a submission row that matches
    the WattBot schema. This dummy version always predicts the
    standardized fallback answer and marks all other fields as is_blank.
    """
    data = {}
    for col in expected_columns:
        if col == "id":
            # Copy the question ID from test_Q
            data[col] = row["id"]
        elif col == "question":
            # Copy the original question text from test_Q
            data[col] = row["question"]
        elif col == "answer":
            # Natural-language answer: use the official fallback phrase
            data[col] = FALLBACK_ANSWER
        elif col in [
            "answer_value",
            "answer_unit",
            "ref_id",
            "ref_url",
            "supporting_materials",
            "explanation",
        ]:
            # For unanswered questions, these must be set to "is_blank"
            data[col] = "is_blank"
        else:
            # Any unexpected extra columns are also set to is_blank
            data[col] = "is_blank"
    return pd.Series(data)


# Apply the dummy function to every test question
submission = test_q.apply(make_dummy_row, axis=1)

# Ensure column order matches train_QA exactly
submission = submission[expected_columns]

print("Submission shape:", submission.shape)
print(submission.head())

# ---------------------------------------------------------------------
# 4. Write submission to /kaggle/working so that Kaggle can pick it up
# ---------------------------------------------------------------------

OUT_PATH = Path("./submission.csv")
submission.to_csv(OUT_PATH, index=False)

print("Saved submission to", OUT_PATH)
