import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Custom CSS
def custom_css():
    st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
        }
        .st-da {
            background-color: #000000;
            color: #ffffff;  /* White color for font */
        }
        .st-bb {
            background-color: #000000;
            color: #ffffff;  /* White color for font */
        }
        .css-145kmo2 {
            background-color: #000000;
            color: #ffffff;  /* White color for font */
        }
    </style>
    """, unsafe_allow_html=True)




# Load the dataset
@st.cache_data
def load_data():
    # Replace 'path_to_file' with the actual path of your dataset
    df = pd.read_csv('netflix_titles.csv')
    return df[['title', 'description']].dropna()

# Calculate cosine similarity
@st.cache_data
def calculate_cosine_sim(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Get recommendations
def get_recommendations(title, df, cosine_sim):
    idx = df.index[df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Load data
df = load_data()
cosine_sim = calculate_cosine_sim(df)


# Streamlit app
custom_css()
st.image('Logos-Readability-Netflix-logo.jpg', caption=None, width=None, use_column_width=True)
st.markdown("<h1 style='color: white;'>Netflix Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: white;'>Select a Movie or TV Show that You Like & This App Will Recommend You 10 Similar Titles that You Will Also Probably Love:</p>", unsafe_allow_html=True)


#st.title('<p style='color: white;'>Netflix & Chill?</p>')
#st.markdown('<p style='color: white;'>This is a netflix recommendation system based on a content-based filtering approach, which utilizes the descriptions of Netflix movies and TV shows to find similarities among them. The model uses the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm to convert text descriptions into numerical vectors. Then, it calculates the cosine similarity between these vectors to measure how similar the descriptions are. By comparing these similarity scores, the model recommends titles that have similar content</p>')
selected_title = st.selectbox('', df['title'].tolist())

# Predict button
if st.button('Get Your Next Netflix Watch'):
    st.markdown("<p style='color: white;'>Recommended Titles:</p>", unsafe_allow_html=True)
    recommendations = get_recommendations(selected_title, load_data(), calculate_cosine_sim(load_data()))
    for i, title in enumerate(recommendations, start=1):
        st.markdown(f"<p style='color: white;'>{i}. {title}</p>", unsafe_allow_html=True)

# Display a GIF at the end
st.image("Demi-Lovato-Popcorn-Gif.gif", caption="", use_column_width=True)


