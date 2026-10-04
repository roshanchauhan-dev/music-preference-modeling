import pandas as pd 
def clean_audio_dataset(df, name_col, artist_col, id_col=None):
    """
    Generic cleaner for any Spotify-style audio features dataset.
    Standardizes column names to track_name / artists.
    """
    df = df.dropna(subset=[name_col, artist_col, "danceability"])

    if id_col:
        df = df.drop_duplicates(subset=id_col)
    df = df.drop_duplicates(subset=[name_col, artist_col])

    df = df[(df["duration_ms"] >= 60000) & (df["duration_ms"] <= 600000)]
    df = df[df["tempo"] > 0]
    df = df.reset_index(drop=True)
    df = df.rename(columns={name_col: "track_name", artist_col: "artists"})
    return df

def build_catalog(datasets,common_cols):
    """
    Combines a list of cleaned datasets into one catalog, removing duplicates.
    """
    combined = pd.concat([d[common_cols] for d in datasets], ignore_index=True)
    combined = combined.drop_duplicates(subset=["track_name", "artists"])
    combined = combined.reset_index(drop=True)
    combined["track_key"] = combined["track_name"] + " - " + combined["artists"]
    return combined

def add_genre_label(df, source_col, label_type):
    """
    Adds a unified 'genre_or_language' column plus a 'label_type' column
    ('genre' or 'language'), so datasets with different label types
    can be combined without pretending they're the same thing.
    """
    df=df.copy()
    df["genre_or_language"] = df[source_col]
    df["label_type"] = label_type
    return df

def match_my_tracks(tracks, catalog):
    """
    Merges personal taste scores against the audio-features catalog.
    Prints the match rate.
    """
    matched = tracks.merge(catalog, left_on="track", right_on="track_key", how="inner")
    rate = len(matched) / len(tracks) * 100
    print(f"Your songs: {len(tracks)}")
    print(f"Matched: {len(matched)}")
    print(f"Match rate: {rate:.1f}%")
    return matched
