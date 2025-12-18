import json
from sklearn.metrics import precision_score, recall_score

def evaluate_memory_performance(memory, expected_traits, expected_emotional_patterns):
    """
    Evaluates the quality of semantic and episodic memory updates.
    
    Args:
        memory (dict): User's memory containing semantic and episodic memory.
        expected_traits (dict): Ground truth of user personality traits.
        expected_emotional_patterns (list): Expected emotional patterns from user behavior.

    Returns:
        dict: Evaluation metrics.
    """

    semantic_memory = memory.get("semantic_memory", {})
    episodic_memory = memory.get("episodic_memory", [])

    # Evaluating personality traits
    actual_traits = semantic_memory.get("personality_traits", {})
    trait_stability = {}
    trait_recall = []
    trait_precision = []

    for trait, expected_value in expected_traits.items():
        actual_value = actual_traits.get(trait, None)
        if actual_value is not None:
            # Measure stability: how consistent the trait remains over time
            trait_stability[trait] = round(abs(expected_value - actual_value), 2)

            # Binary classification for precision and recall: close match = 1, mismatch = 0
            trait_recall.append(1 if abs(expected_value - actual_value) < 0.2 else 0)
            trait_precision.append(1 if actual_value != 0 else 0)

    # Evaluating emotional patterns
    actual_patterns = semantic_memory.get("emotional_patterns", [])
    matched_patterns = len(set(actual_patterns) & set(expected_emotional_patterns))
    pattern_recall = matched_patterns / len(expected_emotional_patterns) if expected_emotional_patterns else 0
    pattern_precision = matched_patterns / len(actual_patterns) if actual_patterns else 0

    # Advanced metrics for personality traits
    trait_recall_score = recall_score(trait_recall, [1] * len(trait_recall)) if trait_recall else 0
    trait_precision_score = precision_score(trait_precision, [1] * len(trait_precision)) if trait_precision else 0
    trait_stability_score = round(sum(trait_stability.values()) / len(trait_stability), 2) if trait_stability else 0

    return {
        "trait_recall": trait_recall_score,
        "trait_precision": trait_precision_score,
        "trait_stability": trait_stability_score,
        "pattern_recall": pattern_recall,
        "pattern_precision": pattern_precision
    }

# Example usage
if __name__ == "__main__":
    with open("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json", "r", encoding="utf-8") as f:
        memory_data = json.load(f)

    expected_traits = {
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "agreeableness": 0.9,
        "neuroticism": 0.3
    }

    expected_emotional_patterns = ["calm", "inquisitive", "thoughtful"]

    results = evaluate_memory_performance(memory_data["Keyvan2"], expected_traits, expected_emotional_patterns)
    print("\n📊 Evaluation Results:", results)
