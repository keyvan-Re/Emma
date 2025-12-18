import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz  import fuzz

def calculate_text_similarity(text1, text2):
    """Calculates cosine similarity between two text blocks."""
    if not text1 or not text2:
        return 0.0
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def jaccard_similarity(str1, str2):
    """Compute Jaccard similarity between two strings."""
    set1, set2 = set(str1.lower().split()), set(str2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

def fuzzy_topic_match(actual_topics, expected_topics, threshold=75):
    """Matches topics using fuzzy similarity."""
    matched = np.zeros(len(expected_topics))  # Track matches for recall
    tp = 0  # True Positives
    
    for actual in actual_topics:
        for i, expected in enumerate(expected_topics):
            similarity = fuzz.partial_ratio(actual.lower(), expected.lower())
            if similarity >= threshold:
                matched[i] = 1  # Mark expected topic as matched
                tp += 1
                break  # Move to next actual topic after a match

    fn = len(expected_topics) - sum(matched)  # False Negatives
    fp = len(actual_topics) - tp  # False Positives

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

    return recall, precision, f1

def compute_emotion_match(actual_emotion, expected_emotion):
    """Compute whether the actual and expected emotions match."""
    return 1 if jaccard_similarity(actual_emotion, expected_emotion) >= 0.5 else 0

def evaluate_single_session_episodic_memory(user_name, memory_data, session_id=None, expected_session=None):
    """
    Evaluates the episodic memory of a **specific session** for a given user.
    """

    # Check if the user exists in memory
    if user_name not in memory_data or "episodic_memory" not in memory_data[user_name]:
        print(f"⚠️ No episodic memory found for user: {user_name}")
        return None

    # Retrieve the episodic memory entries for this user
    user_episodic_memory = memory_data[user_name]["episodic_memory"]

    if not user_episodic_memory:
        print(f"⚠️ No recorded sessions for user: {user_name}")
        return None

    # Locate the specific session by session_id
    actual_session = next((s for s in user_episodic_memory if s.get("session_id") == session_id), None)
    
    if not actual_session:
        print(f"⚠️ No session found with session_id: {session_id}")
        return None

    if expected_session is None:
        print("⚠️ No expected session provided for evaluation.")
        return None

    # Compute topic recall, precision, and F1 using fuzzy matching
    topic_recall, topic_precision, topic_f1_score = fuzzy_topic_match(
        actual_session.get("topics_discussed", []), expected_session.get("topics_discussed", []))

    # Compute emotion match
    emotion_match = compute_emotion_match(actual_session.get("emotional_state", ""), expected_session.get("emotional_state", ""))

    # Compute insight similarity
    insight_similarity = calculate_text_similarity(actual_session.get("insights", ""), expected_session.get("insights", ""))

    return {
        "user": user_name,
        "session_id": session_id,
        "topic_recall": round(topic_recall, 2),
        "topic_precision": round(topic_precision, 2),
        "topic_f1_score": round(topic_f1_score, 2),
        "emotion_match": emotion_match,
        "insight_similarity": round(insight_similarity, 2),
    }

# Example usage
if __name__ == "__main__":
    # Load stored episodic memory from file
    with open("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json", "r", encoding="utf-8") as f:
        memory_data = json.load(f)

    user_name = "Keyvan2"
    expected_session = {
        "topics_discussed": [
            "Social anxiety and lack of confidence",
                    "Nighttime anxiety and sleep difficulties",
                    "Overthinking and worrying about things beyond control",
                    "Managing physical symptoms of anxiety"
        ],
        "emotional_state": "Anxious, stressed",
                "insights": "The user is seeking help to overcome social anxiety, improve nighttime routines for better sleep, stop overthinking, and manage physical symptoms of anxiety. They are open to seeking support and implementing strategies to address their mental health concerns."
    }

    session_id = 25  # Change this to the session ID you want to evaluate

    results = evaluate_single_session_episodic_memory(user_name, memory_data, session_id=session_id, expected_session=expected_session)

    if results:
        print("\n📊 Episodic Memory Evaluation for User Session:", results)
