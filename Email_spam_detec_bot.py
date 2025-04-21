import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv("spam_assassin.csv", encoding='latin-1')

# Debugging: Print column names
print("Columns in the dataset:", data.columns)

# Use the correct column names
data = data[['text', 'target']]  # Select the correct columns
data.columns = ['message', 'label']  # Rename columns for clarity

# Map labels to numerical values (if not already numeric)
# Assuming 'target' is already numeric (0 for ham, 1 for spam), this step is optional
data['label'] = data['label'].astype(int)

# Drop missing values
data.dropna(inplace=True)

# Function to preprocess the emails
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+@\s+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Clean the dataset
data['cleaned_message'] = data['message'].apply(clean_text)

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_message'])

# Output labels
y = data['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predicting on new emails
def predict_email(text):
    try:
        cleaned = clean_text(text)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        return "Spam" if prediction == 1 else "Not Spam"
    except Exception as e:
        return f"Error in prediction: {e}"

# Test the prediction function
sample = "Congratulations! You have won a $1,000 gift card. Click here to claim your prize."
print("Sample email prediction:", predict_email(sample))







