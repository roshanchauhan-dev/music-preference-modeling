from sklearn.metrics.pairwise import cosine_similarity

def build_taste_profile(user_tracks, features, top_n=30):
    """Average audio features of a user's top N highest-scoring songs."""
    top_loved = user_tracks.sort_values("score", ascending=False).head(top_n)
    return top_loved[features].mean()

def recommend_from_profile(profile, catalog, features, scaler, already_have, n=15):
    """Recommend catalog songs most similar to a taste profile."""
    catalog_scaled = scaler.transform(catalog[features])
    profile_scaled = scaler.transform([profile])
    similarities = cosine_similarity(profile_scaled, catalog_scaled).flatten()
    
    catalog = catalog.copy()
    catalog["similarity"] = similarities
    catalog["track_key_check"] = catalog["track_name"] + " - " + catalog["artists"]
    
    recs = catalog[~catalog["track_key_check"].isin(already_have)]
    return recs.sort_values("similarity", ascending=False).head(n)