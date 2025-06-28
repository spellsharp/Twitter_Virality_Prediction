import os
import json
import pandas as pd
from datetime import datetime

# Root directory of the dataset
ROOT_DIR = "data/6392078/PHEME_veracity/all-rnr-annotated-threads"
OUTPUT_CSV = "outputs/diffusion_timelines_v2.csv"
TIME_BINS_MINUTES = [10, 20, 30, 45, 60, 180]  # Multiple time bin sizes in minutes

def parse_time(ts):
    return datetime.strptime(ts, "%a %b %d %H:%M:%S %z %Y")

def extract_edges(struct_map, parent=None):
    for tid, children in struct_map.items():
        tid = str(tid)
        if parent is not None:
            yield parent, tid
        if isinstance(children, dict):
            yield from extract_edges(children, parent=tid)

def find_tweet_json(thread_dir, tweet_id):
    tweet_id = str(tweet_id)
    for sub in ("reactions", "source-tweets"):
        path = os.path.join(thread_dir, sub, f"{tweet_id}.json")
        if os.path.isfile(path):
            return path
    return None

def process_event_timelines(event_dir):
    rows = []

    for category in ("non-rumours", "rumours"):
        cat_path = os.path.join(event_dir, category)
        if not os.path.isdir(cat_path):
            continue

        for thread_id in os.listdir(cat_path):
            thread_dir = os.path.join(cat_path, thread_id)
            struct_path = os.path.join(thread_dir, "structure.json")
            if not os.path.isfile(struct_path):
                continue

            try:
                with open(struct_path, encoding="utf8") as f:
                    struct = json.load(f)
            except:
                continue

            source_id = str(thread_id)
            source_path = find_tweet_json(thread_dir, source_id)
            if not source_path:
                continue

            try:
                with open(source_path, encoding="utf8") as f:
                    source_tweet = json.load(f)
                    source_time = parse_time(source_tweet["created_at"])
            except:
                continue

            for _, child_id in extract_edges(struct, parent=source_id):
                child_path = find_tweet_json(thread_dir, child_id)
                if not child_path:
                    continue

                try:
                    with open(child_path, encoding="utf8") as f:
                        child = json.load(f)
                        child_time = parse_time(child["created_at"])
                        minutes = (child_time - source_time).total_seconds() / 60
                        
                        # Create a row for each time bin size
                        for bin_size in TIME_BINS_MINUTES:
                            time_bin = int(minutes // bin_size) * bin_size
                            
                            rows.append({
                                "event": os.path.basename(event_dir),
                                "sourceTweetID": source_id,
                                "replyTweetID": str(child["id_str"]),
                                "created_at": child["created_at"],
                                "minutes_since_source": minutes,
                                "bin_size_minutes": bin_size,
                                "time_bin": time_bin
                            })
                except:
                    continue

    return rows

def main():
    all_rows = []
    for event_name in os.listdir(ROOT_DIR):
        event_path = os.path.join(ROOT_DIR, event_name)
        if os.path.isdir(event_path) and \
           os.path.isdir(os.path.join(event_path, "rumours")) and \
           os.path.isdir(os.path.join(event_path, "non-rumours")):
            print(f"Processing event: {event_name}")
            rows = process_event_timelines(event_path)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved diffusion timeline to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
