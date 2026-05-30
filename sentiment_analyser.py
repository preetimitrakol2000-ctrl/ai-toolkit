"""
Sentiment Analyzer — Naive Bayes Text Classifier
==================================================
Classifies text as POSITIVE or NEGATIVE sentiment.
Built with a Bag-of-Words model and log-probability
arithmetic — no NLTK, no sklearn for the core logic.

Author: Your Name
"""

import re
import math
from collections import defaultdict
from typing import Literal


# ── Simple tokenizer ──────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


STOPWORDS = {
    "i", "me", "my", "the", "a", "an", "is", "it", "in", "on",
    "at", "to", "for", "of", "and", "or", "but", "was", "be",
    "this", "that", "with", "as", "by", "from", "are", "were",
}


# ── Naive Bayes Classifier ────────────────────────────────────────────────────
class NaiveBayesSentiment:
    """
    Multinomial Naive Bayes for binary sentiment classification.

    Uses Laplace (add-1) smoothing to handle unseen words.
    Log-probabilities prevent numerical underflow with long texts.
    """

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords
        self.class_log_prior: dict[str, float] = {}
        self.word_log_likelihood: dict[str, dict[str, float]] = {}
        self.vocab: set[str] = set()
        self.classes: list[str] = []

    def _preprocess(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayesSentiment":
        """
        Train on a list of texts and their corresponding labels.
        """
        self.classes = list(set(labels))
        n_total = len(texts)

        # Separate texts by class
        class_texts: dict[str, list[str]] = defaultdict(list)
        for text, label in zip(texts, labels):
            class_texts[label].extend(self._preprocess(text))

        # Build vocabulary
        for words in class_texts.values():
            self.vocab.update(words)

        vocab_size = len(self.vocab)

        # Log prior: P(class) = count(class) / total
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1

        self.class_log_prior = {
            cls: math.log(class_counts[cls] / n_total)
            for cls in self.classes
        }

        # Log likelihood with Laplace smoothing
        self.word_log_likelihood = {}
        for cls in self.classes:
            word_counts: dict[str, int] = defaultdict(int)
            for word in class_texts[cls]:
                word_counts[word] += 1

            total_words = sum(word_counts.values())
            self.word_log_likelihood[cls] = {
                word: math.log((word_counts[word] + 1) / (total_words + vocab_size))
                for word in self.vocab
            }
            # Smoothed probability for unknown words
            self.word_log_likelihood[cls]["<UNK>"] = math.log(
                1 / (total_words + vocab_size)
            )

        return self

    def predict(self, text: str) -> str:
        """Predict the sentiment label for a single text string."""
        tokens = self._preprocess(text)
        scores: dict[str, float] = {}

        for cls in self.classes:
            score = self.class_log_prior[cls]
            for token in tokens:
                if token in self.word_log_likelihood[cls]:
                    score += self.word_log_likelihood[cls][token]
                else:
                    score += self.word_log_likelihood[cls]["<UNK>"]
            scores[cls] = score

        return max(scores, key=lambda c: scores[c])

    def predict_batch(self, texts: list[str]) -> list[str]:
        return [self.predict(t) for t in texts]

    def accuracy(self, texts: list[str], labels: list[str]) -> float:
        preds = self.predict_batch(texts)
        return sum(p == l for p, l in zip(preds, labels)) / len(labels)


# ── Training Data ─────────────────────────────────────────────────────────────
TRAIN_DATA = [
    # Positive
    ("This movie was absolutely fantastic and inspiring!", "positive"),
    ("I loved every moment of this experience.", "positive"),
    ("The product works great, very happy with my purchase.", "positive"),
    ("Brilliant performance, highly recommend this to everyone.", "positive"),
    ("What an amazing story, deeply touching and well written.", "positive"),
    ("Best restaurant I have ever been to, incredible food.", "positive"),
    ("The customer service was outstanding and very helpful.", "positive"),
    ("So much fun, the entire family enjoyed it.", "positive"),
    ("Exceeded all my expectations, truly superb quality.", "positive"),
    ("Wonderful experience from start to finish.", "positive"),
    ("Beautiful design and smooth performance.", "positive"),
    ("Really enjoyed this, will definitely come back.", "positive"),
    # Negative
    ("Terrible experience, I will never return here.", "negative"),
    ("The product broke after two days, total waste of money.", "negative"),
    ("Awful customer service, waited forever and got no help.", "negative"),
    ("Completely disappointed, nothing worked as expected.", "negative"),
    ("Horrible movie, boring and made no sense.", "negative"),
    ("Very poor quality, not worth the price at all.", "negative"),
    ("I regret buying this, it is useless garbage.", "negative"),
    ("The worst experience of my life, absolutely dreadful.", "negative"),
    ("Broken on arrival and support was rude and unhelpful.", "negative"),
    ("Do not buy this, it is a scam and complete junk.", "negative"),
    ("Disgusting food, cold and tasted terrible.", "negative"),
    ("Wasted my time, nothing was as advertised.", "negative"),
]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    texts, labels = zip(*TRAIN_DATA)

    # Leave-one-out style: train on all, test on a few held-out examples
    model = NaiveBayesSentiment(remove_stopwords=True)
    model.fit(list(texts), list(labels))

    train_acc = model.accuracy(list(texts), list(labels))
    print(f"\n✅ Naive Bayes Sentiment Analyzer")
    print(f"   Vocabulary size : {len(model.vocab)} words")
    print(f"   Training accuracy: {train_acc:.2%}")

    # Interactive-style demo
    test_sentences = [
        "This is the best thing I have ever tried, absolutely love it!",
        "Terrible quality, broke immediately and support was useless.",
        "Pretty good overall, quite happy with the result.",
        "Nothing works, I am deeply disappointed with this product.",
        "Surprisingly delightful experience, would recommend to friends.",
    ]

    print("\n── Predictions ──────────────────────────────────────────")
    for sentence in test_sentences:
        prediction = model.predict(sentence)
        emoji = "😊" if prediction == "positive" else "😞"
        print(f"  {emoji}  [{prediction.upper():8s}]  \"{sentence[:55]}...\"")
    print()
