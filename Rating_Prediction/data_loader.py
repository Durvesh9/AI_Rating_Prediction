import pandas as pd
import requests
import json
from pathlib import Path

def download_yelp_dataset():
   
    
    # Sample Yelp reviews for testing (if you can't download full dataset)
    sample_reviews = [
        {
            "text": "The food was absolutely amazing! Fresh ingredients, perfect flavors. Highly recommend!",
            "stars": 5
        },
        {
            "text": "Good place, decent food and service. Nothing special but nothing bad either.",
            "stars": 3
        },
        {
            "text": "Terrible experience. Cold food, slow service, rude staff. Will never return.",
            "stars": 1
        },
        {
            "text": "Pretty good! Nice ambiance, friendly staff. Small portions but tasty.",
            "stars": 4
        },
        {
            "text": "Okay restaurant, nothing to write home about. Average quality.",
            "stars": 2
        },
        {
            "text": "Exceeded all expectations! Best meal I've had in months. Worth every penny!",
            "stars": 5
        },
        {
            "text": "Disappointed. Long wait times, overpriced for what you get.",
            "stars": 2
        },
        {
            "text": "Solid choice! Great food, good prices, friendly atmosphere.",
            "stars": 4
        },
        {
            "text": "Worst restaurant ever. Food poisoning twice. Avoid at all costs!",
            "stars": 1
        },
        {
            "text": "Very nice restaurant with excellent service and delicious food.",
            "stars": 5
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_reviews)
    
    # Expand to 250 rows by repeating with variations
    while len(df) < 250:
        df = pd.concat([df, df.sample(n=min(50, 250-len(df)))], ignore_index=True)
    
    # Save
    output_path = Path("Rating Prediction\data\yelp.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(df)} reviews to {output_path}")
    return df

def load_real_yelp_dataset(filepath):
    """
    Load real Yelp dataset from Kaggle download.
    
    Steps:
    1. Download from https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset
    2. Extract the CSV file
    3. Pass the path here
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} reviews from {filepath}")
        print(f"  Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        print("Using sample data instead...")
        return download_yelp_dataset()

if __name__ == "__main__":
    # Create sample dataset
    df = download_yelp_dataset()
    
    print("\nDataset Summary:")
    print(f"  Total reviews: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Rating distribution:\n{df['stars'].value_counts().sort_index()}")