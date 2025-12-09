"""
TASK 1: Rating Prediction via Prompting
This script demonstrates 3 different prompting approaches for rating prediction.
Each approach gets progressively more sophisticated.
"""

import pandas as pd
import json
import time
from typing import Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os


# CONFIG: API KEY + DATA PATH


load_dotenv()

API_KEY =  os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)


DATA_PATH = "Rating_Prediction\data\yelp.csv"   


# PART 1: LOAD & PREPARE DATA

def load_and_sample_data(filepath: str, sample_size: int = 200) -> pd.DataFrame:
    """
    Load Yelp reviews and sample for testing.
    Expected columns: 'text', 'stars' (or 'rating')
    """
    df = pd.read_csv(filepath)

    if "rating" in df.columns:
        df = df.rename(columns={"rating": "stars"})

    # Sample reviews
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {df.columns.tolist()}")
    return df


# PART 2: DEFINE 3 PROMPTING APPROACHES

class RatingPredictor:
    """Unified interface for all 3 prompting approaches"""

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    # APPROACH 1: Simple Direct Prompt
    def prompt_v1_simple(self, review_text: str) -> Dict:
        """
        Simple prompt: Direct classification request.
        Pros: Fast, simple
        Cons: May not return valid JSON, inconsistent formatting
        """
        prompt = f"""Read this review and predict the star rating (1-5).

Review: {review_text}

Respond in JSON format:
{{"predicted_stars": <number>, "explanation": "<reason>"}}"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text

            # Try to parse JSON
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = {"predicted_stars": -1, "explanation": "Parse error"}
        except Exception as e:
            result = {"predicted_stars": -1, "explanation": f"Error: {str(e)}"}

        return result

    # APPROACH 2: Structured with Clear Constraints
    def prompt_v2_structured(self, review_text: str) -> Dict:
        """
        Structured prompt: Clear format + constraints.
        Pros: Better JSON validity, more consistent
        Cons: Still may have edge cases
        Logic: More explicit about requirements
        """
        prompt = f"""You are a sentiment analysis expert. Analyze this review and predict a star rating.

REVIEW:
{review_text}

INSTRUCTIONS:
1. Determine overall sentiment (1=very negative, 5=very positive)
2. Return ONLY valid JSON (no extra text)
3. predicted_stars must be an integer between 1 and 5
4. explanation must be 1-2 sentences

REQUIRED JSON OUTPUT:
{{"predicted_stars": <1-5>, "explanation": "<brief reason>"}}"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Parse JSON
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                result = json.loads(json_str)
                # Validate stars
                if not (1 <= result.get("predicted_stars", -1) <= 5):
                    result["predicted_stars"] = 3
            else:
                result = {"predicted_stars": 3, "explanation": "Parse error"}
        except Exception as e:
            result = {"predicted_stars": 3, "explanation": f"Error: {str(e)}"}

        return result

    # APPROACH 3: Advanced with Few-Shot Examples
    def prompt_v3_fewshot(self, review_text: str) -> Dict:
        """
        Few-shot prompt: Provides examples + detailed rubric.
        Pros: Best accuracy, consistent JSON, best explanations
        Cons: Longer, uses more tokens
        Logic: Shows LLM exactly what good outputs look like
        """
        prompt = f"""You are an expert review analyst. Rate this review on a 1-5 scale.

EXAMPLES OF EXPECTED OUTPUT:
Example 1: "Service was terrible, waited 2 hours" 
→ {{"predicted_stars": 1, "explanation": "Customer experienced very poor service with long wait times"}}

Example 2: "Good food, reasonable prices, nice atmosphere"
→ {{"predicted_stars": 4, "explanation": "Positive experience with quality food and fair pricing"}}

Example 3: "Just okay, nothing special"
→ {{"predicted_stars": 3, "explanation": "Neutral experience, met basic expectations but lacked standout qualities"}}

RATING RUBRIC:
1 = Extremely negative (serious issues, strong dissatisfaction)
2 = Negative (multiple problems, disappointed)
3 = Neutral (mixed experience, average)
4 = Positive (mostly good, some minor issues)
5 = Extremely positive (excellent, highly satisfied)

REVIEW TO ANALYZE:
{review_text}

RESPONSE RULES:
- Return ONLY valid JSON
- predicted_stars must be 1-5 integer
- explanation must be 1-2 sentences, specific to this review
- No markdown, no extra text

JSON OUTPUT:
{{"predicted_stars": <1-5>, "explanation": "<specific reason>"}}"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Parse JSON
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                result = json.loads(json_str)
                # Validate
                if not (1 <= result.get("predicted_stars", -1) <= 5):
                    result["predicted_stars"] = 3
            else:
                result = {"predicted_stars": 3, "explanation": "Parse error"}
        except Exception as e:
            result = {"predicted_stars": 3, "explanation": f"Error: {str(e)}"}

        return result

    def predict_with_all_approaches(self, review_text: str) -> Dict:
        """Run all 3 approaches and return results"""
        results = {}

        # If you want less terminal spam, comment these prints
        print("  Running Prompt V1 (Simple)...", end=" ")
        results["v1_simple"] = self.prompt_v1_simple(review_text)
        print("✓")
        time.sleep(1)  # Rate limiting

        print("  Running Prompt V2 (Structured)...", end=" ")
        results["v2_structured"] = self.prompt_v2_structured(review_text)
        print("✓")
        time.sleep(1)

        print("  Running Prompt V3 (Few-shot)...", end=" ")
        results["v3_fewshot"] = self.prompt_v3_fewshot(review_text)
        print("✓")
        time.sleep(1)

        return results



# PART 3: EVALUATION METRICS

def is_valid_json(prediction: Dict) -> bool:
    """Check if prediction has required fields with valid types"""
    try:
        stars = prediction.get("predicted_stars")
        expl = prediction.get("explanation")
        return isinstance(stars, int) and 1 <= stars <= 5 and isinstance(expl, str)
    except Exception:
        return False


def calculate_accuracy(actual_stars: int, predicted_stars: int) -> bool:
    """Check if prediction matches actual"""
    return actual_stars == predicted_stars


def evaluate_approach(results_df: pd.DataFrame, approach: str) -> Dict:
    """Evaluate one approach across all reviews"""

    # Extract predictions for this approach
    predictions = results_df[f"{approach}_prediction"].tolist()
    # Use correct column name for actual rating:
    actual = results_df["actual_stars"].tolist()

    # Metrics
    valid_count = sum(1 for p in predictions if is_valid_json(p))
    json_validity_rate = valid_count / len(predictions) * 100

    correct_count = sum(
        1
        for i, p in enumerate(predictions)
        if is_valid_json(p) and calculate_accuracy(actual[i], p.get("predicted_stars"))
    )
    accuracy = correct_count / len(predictions) * 100

    return {
        "approach": approach,
        "json_validity_rate": json_validity_rate,
        "accuracy": accuracy,
        "total_tests": len(predictions),
    }



# PART 4: MAIN EXECUTION


def main():
    """Main execution pipeline"""

    print("=" * 70)
    print("TASK 1: RATING PREDICTION VIA PROMPTING")
    print("=" * 70)

    # Load data
    print("\n[1] LOADING DATA...")
    df = load_and_sample_data('Rating_Prediction\data\yelp.csv', sample_size=200)

    # Initialize predictor
    print("\n[2] INITIALIZING PREDICTOR...")
    predictor = RatingPredictor()

    # Run predictions
    print("\n[3] RUNNING PREDICTIONS ON 200 REVIEWS...")
    print("(This may take several minutes due to API rate limits)")

    all_results = []
    total = len(df)

    for idx, row in df.iterrows():
        print(f"Review {idx + 1}/{total}", end="\r", flush=True)

        predictions = predictor.predict_with_all_approaches(row["text"])

        all_results.append(
            {
                "review_idx": idx,
                "review_text": row["text"][:100] + "...",
                "actual_stars": row["stars"],
                "v1_simple_prediction": predictions["v1_simple"],
                "v2_structured_prediction": predictions["v2_structured"],
                "v3_fewshot_prediction": predictions["v3_fewshot"],
            }
        )

    print()  # newline after progress

    # Create results dataframe
    results_df = pd.DataFrame(all_results)

    # Evaluate each approach
    print("\n[4] EVALUATING APPROACHES...")
    evaluation_results = []

    for approach in ["v1_simple", "v2_structured", "v3_fewshot"]:
        metrics = evaluate_approach(results_df, approach)
        evaluation_results.append(metrics)
        print(f"\n{approach}:")
        print(f"  JSON Validity Rate: {metrics['json_validity_rate']:.1f}%")
        print(f"  Accuracy: {metrics['accuracy']:.1f}%")

    comparison_df = pd.DataFrame(evaluation_results)

    print("\n[5] RESULTS SUMMARY")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # Save results
    results_df.to_csv("predictions_results.csv", index=False)
    comparison_df.to_csv("approach_comparison.csv", index=False)

    print("\n✓ Results saved to CSV files")
    return results_df, comparison_df


if __name__ == "__main__":
    results_df, comparison_df = main()
