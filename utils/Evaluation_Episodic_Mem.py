import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_text_similarity(text1, text2):
    """Calculates cosine similarity between two text blocks."""
    if not text1 or not text2:
        return 0.0
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def evaluate_single_session_episodic_memory(user_name, memory_data, session_id=None, expected_session=None):
    """
    Evaluates the episodic memory of a **specific session** for a given user.

    Args:
        user_name (str): The user whose session is being evaluated.
        memory_data (dict): The full stored memory dataset.
        session_id (int): The unique session ID to evaluate.
        expected_session (dict): The expected episodic memory entry for comparison.

    Returns:
        dict: Evaluation metrics for this session.
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
    # Ensure session_id is a string for comparison
    #actual_session = next((s for s in user_episodic_memory if str(s["session_id"]) == str(session_id)), None)

    # Locate the specific session by session_id
    #actual_session = next((s for s in user_episodic_memory if s["session_id"] == session_id), None)

    actual_session = next((s for s in user_episodic_memory if s.get("session_id") == session_id), None)
    print("actual_session:",actual_session)
    print("expected_session:",expected_session)
    

    if not actual_session:
        print(f"⚠️ No session found with session_id: {session_id}")
        return None

    if expected_session is None:
        print("⚠️ No expected session provided for evaluation.")
        return None

    # Evaluate topics discussed
    expected_topics = set(expected_session.get("topics_discussed", []))
    actual_topics = set(actual_session.get("topics_discussed", []))

    true_positive = len(expected_topics & actual_topics)
    false_positive = len(actual_topics - expected_topics)
    false_negative = len(expected_topics - actual_topics)

    topic_recall = true_positive / len(expected_topics) if expected_topics else 0
    topic_precision = true_positive / len(actual_topics) if actual_topics else 0
    topic_f1 = (
        2 * (topic_precision * topic_recall) / (topic_precision + topic_recall)
        if topic_precision + topic_recall > 0 else 0
    )

    # Evaluate emotional state
    expected_emotion = expected_session.get("emotional_state", "").lower()
    actual_emotion = actual_session.get("emotional_state", "").lower()
    emotion_match = 1 if expected_emotion == actual_emotion else 0

    # Evaluate insight similarity using cosine similarity
    expected_insight = expected_session.get("insights", "").lower()
    actual_insight = actual_session.get("insights", "").lower()
    insight_similarity = calculate_text_similarity(expected_insight, actual_insight)

    return {
        "user": user_name,
        "session_id": session_id,
        "topic_recall": round(topic_recall, 2),
        "topic_precision": round(topic_precision, 2),
        "topic_f1_score": round(topic_f1, 2),
        "emotion_match": emotion_match,
        "insight_similarity": round(insight_similarity, 2),
    }


# Example usage
if __name__ == "__main__":
    # Load stored episodic memory from file
    with open("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json", "r", encoding="utf-8") as f:
        memory_data = json.load(f)

    # Specify the user whose session we want to evaluate
    user_name = "Keyvan2"

    # Hypothetical expected memory entry for this session
    
    expected_session = {
    "topics_discussed": [
        "social anxiety and confidence",
        "anxiety at night and sleep issues",
        "overthinking and worry",
        "physical symptoms of anxiety"
    ],
    "emotional_state": "anxious",
    "insights": "The user feels anxious in social situations and wants to improve confidence. Nighttime anxiety affects their sleep, leading them to seek effective routines. They struggle with overthinking and worry about things beyond their control. Additionally, they experience physical symptoms of anxiety, such as a racing heart and sweating, and seek coping strategies."
}


    # Specify the session_id we want to evaluate
    session_id = 25  # Change this to the session ID you want to evaluate

    # Evaluate the session
    results = evaluate_single_session_episodic_memory(user_name, memory_data, session_id=session_id, expected_session=expected_session)

    if results:
        print("\n📊 Episodic Memory Evaluation for User Session:", results)
