import streamlit as st
import pandas as pd
import sys
sys.path.append("../src")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from recommender import build_taste_profile, recommend_from_profile

st.set_page_config(page_title="Music Preference Modeling", page_icon="🎵")

@st.cache_data
def load_data():
    catalog = pd.read_csv("../data/processed/full_catalog.csv")
    my_songs = pd.read_csv("../data/processed/my_matched_songs.csv")
    return catalog, my_songs

catalog, my_songs = load_data()

features = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness",
            "valence", "tempo"]

scaler = StandardScaler().fit(catalog[features])
X = scaler.transform(catalog[features])

st.title("🎵 Music Preference Modeling")
st.write("A content-based recommender built on real listening behavior.")

tab1, tab2, tab3 = st.tabs(["Search a song", "My recommendations", "My taste profile"])

# Tab 1: search a song
# Tab 1: search a song, with language filter
with tab1:
    st.subheader("Find similar songs")
    
    language_options = ["All"] + sorted(catalog["genre_or_language"].dropna().unique().tolist())
    selected_language = st.selectbox("Filter by genre/language (optional):", language_options)
    
    song_name = st.text_input("Enter a song name:")

    if song_name:
        matches = catalog[catalog["track_name"].str.lower() == song_name.lower()]
        if matches.empty:
            st.warning("Song not found in the catalog. Try another name.")
        else:
            idx = matches.index[0]
            song_vector = X[idx].reshape(1, -1)
            similarities = cosine_similarity(song_vector, X).flatten()

            results = catalog.copy()
            results["similarity"] = similarities
            results = results[results.index != idx]

            if selected_language != "All":
                results = results[results["genre_or_language"] == selected_language]

            results = results.sort_values("similarity", ascending=False).head(10)

            st.write(f"Songs similar to **{song_name}**:")
            st.dataframe(results[["track_name", "artists", "genre_or_language", "similarity"]], hide_index=True)

# Tab 2: personalized recommendations
# Tab 2: personalized recommendations, with language filter
with tab2:
    st.subheader("Recommended for you")
    
    rec_language = st.selectbox("Filter recommendations by language (optional):", 
                                   ["All"] + sorted(catalog["genre_or_language"].dropna().unique().tolist()),
                                   key="rec_lang")
    top_n = st.slider("Number of top songs to base recommendations on:", 10, 50, 30)

    profile = build_taste_profile(my_songs, features, top_n=top_n)
    already_have = set(my_songs["track"])

    filtered_catalog = catalog if rec_language == "All" else catalog[catalog["genre_or_language"] == rec_language]

    recommendations = recommend_from_profile(profile, filtered_catalog, features, scaler, already_have, n=15)
    st.dataframe(recommendations[["track_name", "artists", "genre_or_language", "similarity"]], hide_index=True)
    
# Tab 3: taste profile insights
with tab3:
    st.subheader("What your listening history says about your taste")

    profile_display = build_taste_profile(my_songs, features, top_n=30)
    st.write("Average audio features of your top 30 most-loved songs:")
    st.bar_chart(profile_display)

    st.write(f"Total matched songs: **{len(my_songs)}**")
    st.write(f"Catalog size: **{len(catalog):,}** songs")