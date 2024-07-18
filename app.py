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

# Font
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&family=Playwrite+PE:wght@100..400&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)
# Custom CSS
st.markdown(
    """
<style>
    *{
  font-family: "Outfit", sans-serif !important;
    }
    body {
        display: flex;
        flex-direction: column;
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
        width: 100%;
        margin-top: 10px;
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
        color: #FFFFFF;
        text-wrap: nowrap;
        text-align: center;
        font-family: 'Arial', sans-serif;
        padding-bottom: 20px;
    }
    .recommendation-container {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-qgowjl>p{
    font-size:20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App title
st.title("📚 Book Recommendation System")

# Create a single column layout
col1 = st.container()

with col1:
    book_name = st.text_input(
        "👀 Enter a book name:", placeholder="e.g., To Kill a Mockingbird"
    )

    if st.button("Recommend 🚀"):
        if book_name:
            recommendations = recommend_knn(book_name)
            if recommendations:
                st.markdown(
                    '<div class="recommendation-container">', unsafe_allow_html=True
                )
                st.success(f'⭐️ Recommendations for "{book_name}":')
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("No recommendations found 😔. Try a different book name.")
        else:
            st.warning("Please enter a book name 🤩")

# Add a footer
st.markdown("---")
st.markdown("Made with ❤️  Arjun")
