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
    X = vectorizer.transform(df['text'])
    preds = model.predict(X)
    spam_count = int((preds == 1).sum())
    total = int(len(preds))
    return {
        "total_samples": total,
        "spam_detected": spam_count,
        "spam_percentage": spam_count / total * 100
    }
