from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import re
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')



app = FastAPI()


model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
ps = PorterStemmer()


class EmailRequest(BaseModel):
    email: str

def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)
    
    x = []
    for i in text:
        if i.isalnum():
            x.append(i)

    y = []
    for i in x:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    z = []
    for i in y:
        z.append(ps.stem(i))

    return " ".join(z)

@app.post("/predict")
def predict_spam(data: EmailRequest):
    transformed_text = transform_text(data.email)
    vector_input = vectorizer.transform([transformed_text])
    prediction = model.predict(vector_input)
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return {"prediction": result}