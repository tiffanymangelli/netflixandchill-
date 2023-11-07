# netflix recommendations app-
Netflix Movies and TV Shows Recommender

Contentnt based filtering - TF-IDF:
The recommendation system is based on a content-based filtering approach, which utilizes the descriptions of Netflix movies and TV shows to find similarities among them. The model uses the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm to convert text descriptions into numerical vectors. Then, it calculates the cosine similarity between these vectors to measure how similar the descriptions are. By comparing these similarity scores, the model recommends titles that have similar content.

Streamlit App:
The Streamlit app provides a simple and interactive interface for users to explore recommendations based on their chosen Netflix title. Users can select a movie or TV show from a dropdown menu, and the app will instantly display a list of the top 10 recommended titles with similar content. The app aims to help users discover new content on Netflix that aligns with their preferences.
