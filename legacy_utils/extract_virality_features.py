import pandas as pd

TIMELINE_CSV = "outputs/diffusion_timelines_v2.csv"
LOGISTIC_CSV = "outputs/logistic_fit_results_v2.csv"
OUTPUT_CSV = "outputs/virality_features_v2.csv"

def load_data():
    df = pd.read_csv(TIMELINE_CSV)
    df = df[df["minutes_since_source"] >= 0]
    return df

def compute_features(df):
    features = []
    
    # Define all time windows we want to track
    time_windows = [10, 20, 30, 45, 60, 180, 1440]  # 10min, 20min, 30min, 45min, 1hr, 3hr, 24hr

    for source_id, group in df.groupby("sourceTweetID"):
        group = group.copy()
        total_reactions = len(group)
        
        # Calculate reactions for each time window
        feature_dict = {
            "sourceTweetID": source_id,
            "total_reactions": total_reactions,
        }
        
        # Add reactions count for each time window
        for window in time_windows:
            if window == 1440:
                column_name = "reactions_24hr"
            elif window == 180:
                column_name = "reactions_3hr"
            elif window == 60:
                column_name = "reactions_1hr"
            else:
                column_name = f"reactions_{window}min"
            
            feature_dict[column_name] = (group["minutes_since_source"] <= window).sum()

        lifespan = group["minutes_since_source"].max()
        max_bin = group["time_bin"].value_counts().max()
        num_bins = group["time_bin"].nunique()

        feature_dict.update({
            "lifespan_minutes": lifespan,
            "max_bin_reactions": max_bin,
            "num_time_bins": num_bins,
        })
        
        features.append(feature_dict)

    return pd.DataFrame(features)

def merge_with_logistic(virality_df):
    try:
        logistic_df = pd.read_csv(LOGISTIC_CSV)
        merged = pd.merge(virality_df, logistic_df, on="sourceTweetID", how="left")
        return merged
    except FileNotFoundError:
        print("No logistic fit data found. Skipping merge.")
        return virality_df

def main():
    df = load_data()
    virality_df = compute_features(df)
    merged_df = merge_with_logistic(virality_df)

    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved per-tweet virality features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
