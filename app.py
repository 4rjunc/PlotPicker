import streamlit as st
import joblib
from scipy.sparse import load_npz
import os
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Load the k-NN model
knn_model_path = "./knn_model.pkl"
model_knn = joblib.load(knn_model_path)

# Load the pivot table (as a sparse matrix)
pivot_table_path = "./pivot_table.npz"
pt_matrix = load_npz(pivot_table_path)

# Load the pivot table index
pivot_table_index_path = "./pivot_table_index.pkl"
pivot_table_index = joblib.load(pivot_table_index_path)

# load books
books = pd.read_csv("./datasets/Books.csv", low_memory=False)


def get_book_image(image_url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching image: {str(e)}")
        return None


def recommend_knn(book):
    try:
        query_index = pivot_table_index.get_loc(book)
        distances, indices = model_knn.kneighbors(pt_matrix[query_index], n_neighbors=6)
        recommendations = []
        for i in range(1, len(distances.flatten())):
            recommended_book = pivot_table_index[indices.flatten()[i]]
            url = get_book_url(recommended_book)
            image_url = get_book_image_url(
                recommended_book
            )  # You'll need to implement this function
            recommendations.append((recommended_book, url, image_url))
        return recommendations
    except KeyError:
        similar_books = find_similar_books(book)
        if similar_books:
            return [("Similar book suggestions:", "", "")] + [
                (b, get_book_url(b), get_book_image_url(b)) for b in similar_books[:5]
            ]
        else:
            return [
                ("No similar books found. Please try a different book name.", "", "")
            ]


def get_book_image_url(book_title):
    # Implement this function to return the image URL for a given book title
    # You might need to join your pivot_table with the original dataset that contains image URLs
    # For example:
    return books[books["Book-Title"] == book_title]["Image-URL-M"].iloc[0]


def find_similar_books(book):
    similar_books = [b for b in pivot_table_index if book.lower() in b.lower()]
    return similar_books[:10] if similar_books else []


def get_book_url(book_title):
    # Implement this function to return the URL for a given book title
    # You might need to join your pivot_table with the original dataset that contains URLs
    # For example:
    return books[books["Book-Title"] == book_title]["Image-URL-M"].iloc[0]


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
  font-family: "Outfit", sans-serif;
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
            if recommendations[0][0] == "Similar book suggestions:":
                st.warning(
                    f"The book '{book_name}' was not found. Here are some similar suggestions:"
                )
            else:
                st.success(f'⭐️ Recommendations for "{book_name}":')

            for i, (rec, url, img_url) in enumerate(recommendations, 1):
                col1, col2 = st.columns([1, 3])
                with col1:
                    img = get_book_image(img_url)
                    print("image", img)
                    if img:
                        st.image(img, width=100)
                    else:
                        st.write("No image available")
                with col2:
                    if url:
                        st.markdown(f"{i}. [{rec}]({url})")
                    else:
                        st.write(f"{i}. {rec}")

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.error("No recommendations found. Please try a different book name.")
    else:
        st.warning("Please enter a book name 🤩")

st.markdown("---")
st.markdown("Made with ❤️  Arjun")
