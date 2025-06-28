import os
import json
import pandas as pd
from datetime import datetime
from dateutil import parser as dt_parser

ROOT_DIR = "data/6392078/PHEME_veracity/all-rnr-annotated-threads"
OUTPUT_CSV = "outputs/source_tweet_features.csv"

def parse_time(ts):
    return dt_parser.parse(ts)

def find_tweet_json(thread_dir, tweet_id):
    tweet_id = str(tweet_id)
    for sub in ("source-tweets",):  # only want source tweets
        path = os.path.join(thread_dir, sub, f"{tweet_id}.json")
        if os.path.isfile(path):
            return path
    return None

def extract_features_from_tweet(tweet_json):
    features = {}

    # --- Tweet-level features ---
    text = tweet_json.get("text", "")
    features["text"] = text
    features["text_length"] = len(text)
    features["num_mentions"] = len(tweet_json.get("entities", {}).get("user_mentions", []))
    features["num_hashtags"] = len(tweet_json.get("entities", {}).get("hashtags", []))
    features["num_urls"] = len(tweet_json.get("entities", {}).get("urls", []))
    features["source_platform"] = tweet_json.get("source", "").split(">")[-2].split("<")[0] if "source" in tweet_json else "unknown"

    # --- Time & context features ---
    created_at = tweet_json.get("created_at")
    if created_at:
        dt = parse_time(created_at)
        features["hour_of_day"] = dt.hour
        features["day_of_week"] = dt.weekday()
        features["is_weekend"] = int(dt.weekday() >= 5)
        features["created_at_iso"] = dt.isoformat()
    else:
        features["hour_of_day"] = -1
        features["day_of_week"] = -1
        features["is_weekend"] = -1
        features["created_at_iso"] = None

    # --- User-level features ---
    user = tweet_json.get("user", {})
    features["user_id"] = user.get("id_str", "")
    features["followers_count"] = user.get("followers_count", 0)
    features["friends_count"] = user.get("friends_count", 0)
    features["statuses_count"] = user.get("statuses_count", 0)
    features["favourites_count"] = user.get("favourites_count", 0)
    features["verified"] = int(user.get("verified", False))
    features["default_profile_image"] = int(user.get("default_profile_image", True))
    features["geo_enabled"] = int(user.get("geo_enabled", False))

    # --- User account age ---
    acc_created = user.get("created_at")
    if acc_created:
        acc_dt = parse_time(acc_created)
        features["account_age_days"] = (dt - acc_dt).days
    else:
        features["account_age_days"] = -1

    return features

def process_all_source_tweets():
    rows = []

    for event_name in os.listdir(ROOT_DIR):
        event_path = os.path.join(ROOT_DIR, event_name)
        if not os.path.isdir(event_path):
            continue

        for category in ("non-rumours", "rumours"):
            cat_path = os.path.join(event_path, category)
            if not os.path.isdir(cat_path):
                continue

            for thread_id in os.listdir(cat_path):
                thread_dir = os.path.join(cat_path, thread_id)
                source_id = str(thread_id)

                tweet_path = find_tweet_json(thread_dir, source_id)
                if not tweet_path:
                    continue

                try:
                    with open(tweet_path, encoding="utf8") as f:
                        tweet = json.load(f)
                        features = extract_features_from_tweet(tweet)
                        features["event"] = event_name
                        features["sourceTweetID"] = source_id
                        rows.append(features)
                except Exception as e:
                    continue

    return rows

def main():
    rows = process_all_source_tweets()
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved source tweet features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
