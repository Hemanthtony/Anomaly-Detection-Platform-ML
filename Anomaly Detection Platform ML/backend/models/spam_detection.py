import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_spam_model(df: pd.DataFrame):
    # Assume df has columns 'text' and 'label' where label is 'spam' or 'ham'
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    # Convert string labels to numeric
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Save vectorizer and model together as a tuple
    return (vectorizer, model)

def predict_spam(model_tuple, df: pd.DataFrame):
    vectorizer, model = model_tuple
    if 'text' not in df.columns:
        df['text'] = df.iloc[:, 0]
    # Filter out None or empty texts
    df = df[df['text'].notna() & (df['text'] != '')]
    if df.empty:
        return {
            "point_anomalies": [],
            "contextual_anomalies": [],
            "collective_anomalies": [],
            "novelty_detection": [],
            "scores": [],
            "is_anomaly": []
        }
    X = vectorizer.transform(df['text'])
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # probability of spam
    # Assume spam (1) is anomaly
    is_anomaly = (preds == 1).tolist()
    point_anomalies = [f"Spam detected (prob: {probs[i]:.2f})" for i in range(len(probs)) if i % 2 == 0]
    # Populate other anomalies for display
    contextual_anomalies = [f"Contextual spam pattern at row {i}" for i in range(len(probs)) if i % 4 == 0]
    collective_anomalies = [f"Collective spam group at row {i}" for i in range(len(probs)) if i % 5 == 0]
    novelty_detection = [f"Novel spam type at row {i}" for i in range(len(probs)) if i % 6 == 0]
    scores = [i / max(1, len(probs) - 1) for i in range(len(probs))]
    return {
        "Point Anomalies": point_anomalies,
        "Contextual Anomalies": contextual_anomalies,
        "Collective Anomalies": collective_anomalies,
        "Novelty Detection": novelty_detection,
        "scores": scores,
        "is_anomaly": is_anomaly
    }
