import streamlit as st
import tensorflow as tf  # Import TensorFlow explicitly
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras
from googlesearch import search

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load save recommendation models===================================
embeddings = pickle.load(open('models/embeddings.pkl','rb'))
sentences = pickle.load(open('models/sentences.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))

# load save prediction models============================
# Load the model
loaded_model = keras.models.load_model("models/model.h5")
# Load the configuration of the text vectorizer
with open("models/text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)
# Create a new TextVectorization layer with the saved configuration
loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)
# Load the saved weights into the new TextVectorization layer
with open("models/text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)

# Load the vocabulary
with open("models/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)


# custom functions====================================
def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=6, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list

def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_lookup):
    # Preprocess the abstract using the loaded text vectorizer
    preprocessed_abstract = vectorizer([abstract])

    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_abstract)

    # Convert predictions to human-readable labels
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])

    return predicted_labels

# create app=========================================
st.title('Deep Paper Predictor: Next-Gen Research Assistance')
st.write(" ")

# Sidebar for input
st.sidebar.title('Enter the Detail of Paper Here:')
input_paper = st.sidebar.text_input("Paper title", key="paper_title_input")
new_abstract = st.sidebar.text_area("Paper abstract", key="paper_abstract_input")
if st.sidebar.button("Recommend", key="recommend_button"):
    # recommendation part
    recommend_papers = recommendation(input_paper)

    # Main content area
    st.write("### Recommended Papers")
    for paper in recommend_papers:
        st.write("- ", paper)

    st.write(" ")
    st.write(" ")

    # Google search for recommended papers
    st.write("### You will find the detailed research on following links: ")
    for idx, paper in enumerate(recommend_papers, start=1):
        st.write(" ")
        st.write(f"{idx}. {paper}")
        for url in search(paper, num=3, stop=3):
            st.write(url)
            

    st.write(" ")
    st.write(" ")
    # Prediction part
    st.write("### Predicted Subject Area")
    predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)

    # Concatenate predicted categories into a single string
    predicted_categories_string = ", ".join(predicted_categories)

    # Display predicted categories in a larger box
    st.write("Predicted Categories:")
    st.text(predicted_categories_string)
