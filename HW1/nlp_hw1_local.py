# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re



# Now you need to do sampling because the corpus is too big.
# You can further perform analysis with a greater sampling ratio.

import random

wiki_txt_path = "wiki_texts_combined.txt"
# wiki_texts_combined.txt is a text file separated by linebreaks (\n).
# Each row in wiki_texts_combined.txt indicates a Wikipedia article.


sample_ratio_list = [0.05, 0.1, 0.2]
for sample_ratio in sample_ratio_list:
    output_path = f"wiki_texts_sampled_{sample_ratio}.txt"

    with open(wiki_txt_path, "r", encoding="utf-8") as f:
        with open(output_path, "w", encoding="utf-8") as output_file:
        # TODO4: Sample `20%` Wikipedia articles
        # Write your code here
######################################################################################################
            total_lines = sum(1 for _ in f)
            sample_size = int(total_lines * sample_ratio)
            
            f.seek(0)
            selected_lines = set(random.sample(range(total_lines), sample_size))
            for line_num, line in enumerate(f):
                if line_num in selected_lines:
                    output_file.write(line)
######################################################################################################



# TODO5: Train your own word embeddings with the sampled articles
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
# Hint: You should perform some pre-processing before training.
######################################################################################################
import nltk
import logging
import multiprocessing
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess

nltk.download('punkt_tab')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class PreProcess:
    def __init__(self, filename):
        self.filename = filename
        print(f"Loading data from: {filename}\n")

    def __iter__(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    sentences = self.sent_preproc(line)
                    for sentence in sentences:
                        if len(sentence) >= 3:
                            yield sentence

                if line_num % 100000 == 0:
                    print(f"Processed {line_num} articles")

    def sent_preproc(self, text):
        raw_sentences = nltk.sent_tokenize(text)    
        processed_sentences = []

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence = sentence.lower()
            sentence = remove_stopwords(sentence)
            tokens = simple_preprocess(sentence, min_len=3, max_len=15)

            if tokens:
                processed_sentences.append(tokens)
        
        return processed_sentences


my_model_list = []
for sample_ratio in sample_ratio_list:
    print(f"\n{'='*60}")
    print(f"Training Word2Vec model with {sample_ratio*100}% data sample")
    print(f"{'='*60}")
    
    sentences = PreProcess(f"wiki_texts_sampled_{sample_ratio}.txt")
    
    my_model = Word2Vec(
        sentences=sentences, 
        vector_size=100, 
        window=8, 
        min_count=5, 
        workers=multiprocessing.cpu_count(), 
        sg=1, 
        epochs=3, 
        compute_loss=False
    )
    
    model_filename = f"word2vec_{sample_ratio}.model"
    my_model.save(model_filename)
    print(f"Model saved as: {model_filename}")
    
    my_model_list.append(my_model)

print(f"\n Training completed! Total models: {len(my_model_list)}")
######################################################################################################
data = pd.read_csv("questions-words.csv")



for my_model in my_model_list:

    # Do predictions and preserve the gold answers (word_D)
    preds = []
    golds = []

    for analogy in tqdm(data["Question"]):
          # TODO6: Write your code here to use your trained word embeddings for getting predictions of the analogy task.
          # You should also preserve the gold answers during iterations for evaluations later.
          """ Hints
          # Unpack the analogy (e.g., "man", "woman", "king", "queen")
          # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
          # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
          # Mikolov et al., 2013: big - biggest and small - smallest
          # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
          """

######################################################################################################
          word_a, word_b, word_c, word_d = analogy.split()
          golds.append(word_d)

          try:
            res_seq = my_model.wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=1)
            pred = res_seq[0][0]
            preds.append(pred)

          except:
            preds.append(None)

    def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(gold == pred)

    golds_np, preds_np = np.array(golds), np.array(preds)

    for category in data["Category"].unique():
        mask = data["Category"] == category
        golds_cat, preds_cat = golds_np[mask], preds_np[mask]
        acc_cat = calculate_accuracy(golds_cat, preds_cat)
        print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

    for sub_category in data["SubCategory"].unique():
        mask = data["SubCategory"] == sub_category
        golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
        acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
        print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")
######################################################################################################



# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO7: Plot t-SNE for the words in the SUB_CATEGORY `: family`

######################################################################################################
for my_model in my_model_list:

    family_words = []
    family_vectors = []
    for analogy, subcategory in zip(data["Question"], data["SubCategory"]):
        if subcategory == SUB_CATEGORY:
            words = analogy.split()

            for word in words:
                if word in my_model.wv.key_to_index and word not in family_words:
                    family_words.append(word)
                    family_vectors.append(my_model.wv[word])

    family_vectors = np.array(family_vectors)


    # tsne
    tsne = TSNE(n_components=2, perplexity=min(30, len(family_vectors)-1), random_state=42)
    vectors_2d = tsne.fit_transform(family_vectors)


    # fig
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.7, c='blue')
    for i, word in enumerate(family_words):
        plt.annotate(word,
                    xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    fontsize=10,
                    ha='left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
######################################################################################################

    plt.title("Word Relationships from Google Analogy Task")
    plt.show()
    plt.savefig("word_relationships_word2vec.png", bbox_inches="tight")

