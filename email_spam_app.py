from flask import Flask, request, render_template
import joblib
import re
import string

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean the email text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Route for the homepage
@app.route("/")
def home():
    return render_template("spam_detector_app.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the message from the form
        message = request.form["message"]
        if not message.strip():
            return render_template("spam_detector_app.html", result="Please enter a valid message.")
        
        # Clean and vectorize the input
        cleaned = clean_text(message)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict using the model
        prediction = model.predict(vectorized)[0]
        result = "SPAM" if prediction == 1 else "NOT SPAM"
        
        # Render the result back to the template
        return render_template("spam_detector_app.html", message=message, result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)