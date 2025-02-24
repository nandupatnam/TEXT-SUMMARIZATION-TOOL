import nltk
import re
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from flask import Flask, render_template, request

app = Flask(__name__)

def read_article(text):
    article = text.split(". ")
    sentences = []

    for sentence in article:
        cleaned_sentence = re.sub("[^a-zA-Z]", " ", sentence)
        sentences.append(cleaned_sentence.split())

    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = set()

    sent1 = [w.lower() for w in sent1 if w.lower() not in stopwords]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stopwords]

    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1
    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words):  # Correct indentation here!
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(text, top_n=5):
    stop_words = set(stopwords.words('english'))
    sentences = read_article(text)

    if not sentences:
        return "No valid sentences found in the article."

    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summarize_text = [" ".join(ranked_sentences[i][1]) for i in range(min(top_n, len(ranked_sentences)))]

    return ". ".join(summarize_text)

@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    summary = ""
    if request.method == "POST":
        text = request.form["text"]
        original_text = text
        try:
            summary = generate_summary(text, 2)
        except Exception as e:
            summary = f"Error during summarization: {e}"

    return render_template("index.html", original_text=original_text, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)