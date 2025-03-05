import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

faq_data = data['faq']
questions = [entry["question"] for entry in faq_data]
answers = [entry["answer"] for entry in faq_data]

def get_response(user_question):
    vectorizer = TfidfVectorizer(stop_words="english")
    
    tfidf_matrix = vectorizer.fit_transform(questions + [user_question])  
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]) 
    
    best_match_index = cosine_sim.argmax()
    return answers[best_match_index]

while True:
    user_input = input("Pregunta al asistente (o 'salir' para terminar): ").lower().strip()
    
    if user_input == "salir":
        break
    
    response = get_response(user_input)
    print(f"Respuesta: {response}")