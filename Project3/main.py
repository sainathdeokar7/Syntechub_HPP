import pandas as pd
import nltk
import re
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# 1. Load Dataset
try:
    messages = pd.read_csv(
        'SMSSpamCollection.txt',
        sep='\t',
        names=["label", "message"]
    )
except FileNotFoundError:
    print("Dataset file not found. Make sure 'SMSSpamCollection.txt' is in the working directory.")
    sys.exit()

# 2. Text Preprocessing
ps = PorterStemmer()
corpus = []

for message in messages['message']:
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    corpus.append(' '.join(review))

# 3. Feature Extraction (Better than CountVectorizer)
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(corpus)

# 4. Encode Labels Safely
if messages['label'].nunique() != 2:
    raise ValueError("Unexpected labels found in dataset.")

y = messages['label'].map({'ham': 0, 'spam': 1}).values

# 5. Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))