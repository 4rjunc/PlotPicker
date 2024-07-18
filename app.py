import streamlit as st
import joblib
from scipy.sparse import load_npz
import os
import numpy as np

# Load the k-NN model
knn_model_path = "./knn_model.pkl"
model_knn = joblib.load(knn_model_path)

# Load the pivot table (as a sparse matrix)
pivot_table_path = "./pivot_table.npz"
pt_matrix = load_npz(pivot_table_path)

# Load the pivot table index
pivot_table_index_path = "./pivot_table_index.pkl"
pivot_table_index = joblib.load(pivot_table_index_path)


def recommend_knn(book):
    try:
        query_index = pivot_table_index.get_loc(book)
        distances, indices = model_knn.kneighbors(pt_matrix[query_index], n_neighbors=6)

        recommendations = []
        for i in range(1, len(distances.flatten())):
            recommendations.append(pivot_table_index[indices.flatten()[i]])

        return recommendations

    except KeyError:
        similar_books = find_similar_books(book)
        return similar_books


def find_similar_books(book):
    similar_books = [b for b in pivot_table_index if book.lower() in b.lower()]
    return similar_books[:10] if similar_books else []


# st.title("📚Book Recommendation System")
#
# book_name = st.text_input("👀 Enter a book name:")
# if st.button("Recommend 🚀"):
#    if book_name:
#        recommendations = recommend_knn(book_name)
#        if recommendations:
#            st.write(f'⭐️ Recommendations for "{book_name}":')
#            for i, rec in enumerate(recommendations, 1):
#                st.write(f"{i}. {rec}")
#        else:
#            st.write("No recommendations found 😔 . Try a different book name.")
#    else:
#        st.write("Please enter a book name 🤩")
#

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="centered",
)

# Custom CSS
st.markdown(
    """
<style>
    body {
        background-color: #f0f4f8;
        color: #1e1e1e;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        border: 2px solid #4CAF50;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
        padding-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App title
st.title("📚 Book Recommendation System")

# Create two columns
col1, col2 = st.columns([3, 1])

with col1:
    book_name = st.text_input(
        "👀 Enter a book name:", placeholder="e.g., To Kill a Mockingbird"
    )

with col2:
    if st.button("Recommend 🚀"):
        if book_name:
            recommendations = recommend_knn(book_name)
            if recommendations:
                st.success(f'⭐️ Recommendations for "{book_name}":')
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.error("No recommendations found 😔. Try a different book name.")
        else:
            st.warning("Please enter a book name 🤩")

# Add a footer
st.markdown("---")
st.markdown("Made with ❤️  Arjun")
