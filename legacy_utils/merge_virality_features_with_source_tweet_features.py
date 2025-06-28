import pandas as pd

# Load the CSV
virality_df = pd.read_csv("outputs/virality_features_v2.csv")
print("Virality Columns:", virality_df.columns.tolist())

# Strip all column names to remove accidental spaces
virality_df.columns = virality_df.columns.str.strip()
print("After strip:", virality_df.columns.tolist())

# Now re-check
if 'sourceTweetID' not in virality_df.columns:
    raise ValueError("Column 'sourceTweetID' not found even after stripping. Check spelling.")

# Do the same for source_tweet_features
features_df = pd.read_csv("outputs/source_tweet_features.csv")
features_df.columns = features_df.columns.str.strip()

# Drop the text column if it exists
if 'text' in features_df.columns:
    features_df = features_df.drop('text', axis=1)
    print("Dropped 'text' column from source_tweet_features")

# Merge
merged_df = pd.merge(virality_df, features_df, on="sourceTweetID", how="inner")
merged_df.to_csv("outputs/combined_virality_modeling_data_v2.csv", index=False)
print(f"Merged dataset saved with {len(merged_df)} rows.")
