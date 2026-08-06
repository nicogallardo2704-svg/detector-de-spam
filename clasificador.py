"""
Enterprise Email Fraud & Phishing Detection Engine (NLP Core)
Designed for text classification in corporate cybersecurity environments.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class EmailThreatClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english')
        self.model = MultinomialNB()

    def train(self, corpus, labels):
        """Fits the vectorizer and trains the Multinomial Naive Bayes model."""
        X_vectors = self.vectorizer.fit_transform(corpus)
        self.model.fit(X_vectors, labels)

    def predict(self, texts):
        """Runs batch inference on a list of input string vectors."""
        X_vectors = self.vectorizer.transform(texts)
        return self.model.predict(X_vectors)

# Main Execution Vector for Local and Production Evaluation
if __name__ == "__main__":
    # Simulated Enterprise Dataset (0 = Authentic Corporate, 1 = Phishing/BEC Threat)
    training_corpus = [
        "Urgent: Update your employee banking credentials immediately via this link.",
        "Hey Team, please review the attached Q3 financial performance report.",
        "Wire transfer requested from the CEO. Process $50,000 to the external vendor now.",
        "Are we still on for the sync meeting scheduled for tomorrow at 9 AM?",
        "Security Alert: Your corporate cloud account access expires in 24 hours.",
        "Please send over the updated software deployment documentation by Friday."
    ]
    labels = [1, 0, 1, 0, 1, 0]  # Ground truth annotations

    # Initialize and train the internal NLP classification model
    threat_engine = EmailThreatClassifier()
    threat_engine.train(training_corpus, labels)

    # Validate pipeline consistency using training metrics
    train_predictions = threat_engine.predict(training_corpus)
    
    print("=== Enterprise Threat Intel: Evaluation Matrix ===")
    print(classification_report(labels, train_predictions, target_names=["Authentic Email", "Security Threat"]))

    # Test Live Production Sample Inference
    live_test_payload = ["URGENT ACTION REQUIRED: Verify password on unsecure network now"]
    prediction_result = threat_engine.predict(live_test_payload)
    
    status_label = "🚨 CRITICAL RISK DETECTED" if prediction_result[0] == 1 else "✅ VERIFIED SAFE"
    print(f"Payload Evaluated: '{live_test_payload[0]}'")
    print(f"Analysis Verdict: {status_label}")

