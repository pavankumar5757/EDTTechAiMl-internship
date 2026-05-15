import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Mock Intent Dataset

intents = {
    "intents": [

        {
            "tag": "greeting",

            "patterns": [
                "Hi",
                "Hello",
                "Hey",
                "Good morning"
            ],

            "responses": [
                "Hello!",
                "Hi there!",
                "Welcome!"
            ]
        },

        {
            "tag": "goodbye",

            "patterns": [
                "Bye",
                "See you later",
                "Goodbye"
            ],

            "responses": [
                "Goodbye!",
                "Take care!",
                "See you soon!"
            ]
        },

        {
            "tag": "fees",

            "patterns": [
                "What is the course fee?",
                "Tell me the fees",
                "Fee details",
                "What are the fees?"
            ],

            "responses": [
                "The course fee is ₹5000.",
                "Fees depend on the selected course."
            ]
        },

        {
            "tag": "admission",

            "patterns": [
                "How to apply?",
                "Admission process",
                "How can I join?"
            ],

            "responses": [
                "You can apply online through our website.",
                "Fill out the admission form and upload documents."
            ]
        }

    ]
}

# Prepare Training Data

texts = []
labels = []

for intent in intents['intents']:

    for pattern in intent['patterns']:

        texts.append(pattern)

        labels.append(intent['tag'])

# TF-IDF Vectorization

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# Train Naive Bayes Model

model = MultinomialNB()

model.fit(X, labels)

# Rule-Based Responses

rule_responses = {

    "help": "I can help you with admissions, fees, and courses.",

    "contact": "You can contact us at support@example.com"
}

# Chatbot Function

def chatbot(user_input):

    user_input_lower = user_input.lower()

    # Rule-Based Responses
    for keyword in rule_responses:

        if keyword in user_input_lower:

            return rule_responses[keyword]

    # ML Prediction
    user_vector = vectorizer.transform([user_input])

    prediction = model.predict(user_vector)[0]

    # Return response
    for intent in intents['intents']:

        if intent['tag'] == prediction:

            return random.choice(intent['responses'])

# Start Chatbot

print("================================")
print("       CHATBOT STARTED")
print("Type 'quit' to stop")
print("================================")

running = True

while running:

    user_message = input("\nYou: ")

    if user_message.lower() == "quit":

        print("Bot: Goodbye!")

        running = False

    else:

        response = chatbot(user_message)

        print("Bot:", response)