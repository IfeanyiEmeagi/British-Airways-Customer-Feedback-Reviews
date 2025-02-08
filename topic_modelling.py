import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models

# Download all the necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


# Load the data
def load_reviews(file_path: str):
    """Load the reviews from a JSON file"""

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    reviews = [item["review"] for item in data]  # Extract only the reviews

    return reviews


def preprocess_reviews(reviews):
    """Preprocess the text for topic modelling."""

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    processed_reviews = []

    for review in reviews:
        # 1. Remove alpha-numeric characters and lower the case to normalize the characters
        review = re.sub(r"[^a-zA-Z\s]", "", review).lower()

        # 2. Tokenization
        tokens = nltk.word_tokenize(review)

        # 3. Remove stop words and words shorter than 3 characters
        tokens = [
            token for token in tokens if token not in stop_words and len(token) > 2
        ]

        # 4. Lemmatization - reduce words to their original meaning
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        processed_reviews.append(lemmatized_tokens)

    return processed_reviews


def create_dictionary_and_corpus(processed_reviews):
    """Create a Gensim dictionary and corpus from the processed reviews."""

    dictionary = corpora.Dictionary(processed_reviews)

    # filter out words that appear in less than 2 documents or more than 50% of the documents
    # dictionary.filter_extremes(no_below=2, no_above=0.5)

    corpus = [dictionary.doc2bow(review) for review in processed_reviews]

    return dictionary, corpus


def train_lda_model(dictionary, corpus, num_topics=5):
    """Train the LDA model topic"""

    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto",
        per_word_topics=True,
        random_state=42,
    )
    return lda_model


def display_topics(lda_model, num_words=10):
    """Prints the top words for each topic"""
    print("Discoverd Topics: \n")
    for topic_id, topic_words in lda_model.print_topics(
        num_topics=-1, num_words=num_words
    ):
        print(f"Topic: {topic_id + 1}:")
        print(topic_words)
        print("\n")


def create_pyldavis_visualization(
    lda_model, corpus, dictionary, output_file="lda_visualization.html"
):
    """Generates and saves a pyLDAvis visualization of the topic model."""
    vis_data = pyLDAvis.gensim_models.prepare(
        lda_model, corpus, dictionary, sort_topics=False
    )
    pyLDAvis.save_html(vis_data, output_file)
    print(f"pyLDAvis visualization saved to {output_file}")


# --- Main Execution ---
if __name__ == "__main__":
    json_file = "ba_reviews.json"
    reviews = load_reviews(json_file)

    if not reviews:
        print(
            f"No reviews loaded from {json_file}. Please ensure the file exists and contains review data."
        )
    else:
        processed_reviews = preprocess_reviews(reviews)
        dictionary, corpus = create_dictionary_and_corpus(processed_reviews)

        num_topics = 3  # This number can be adjusted
        lda_model = train_lda_model(dictionary, corpus, num_topics)

        print(f"LDA model trained with {num_topics} topics.\n")
        display_topics(lda_model)

        # Create and save pyLDAvis visualization
        create_pyldavis_visualization(lda_model, corpus, dictionary)

        print("\nTopic modeling and visualization completed!")
