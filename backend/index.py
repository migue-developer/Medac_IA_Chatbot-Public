import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from collections import defaultdict
import random

# Cargar modelos
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar modelo T5
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# Cargar FAQ
faq_file = 'faq.json'
with open(faq_file, 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# Organizar preguntas, respuestas y categorías
category_dict = defaultdict(list)
categories = []
questions = []
answers = []
labels = []

for i, entry in enumerate(faq_data):
    category_dict[entry['category']].append((entry['question'], entry['answer']))
    questions.append(entry['question'])
    answers.append(entry['answer'])
    if entry['category'] not in categories:
        categories.append(entry['category'])
    labels.append(categories.index(entry['category']))

# Obtener embeddings de las preguntas FAQ
question_embeddings = embed_model.encode(questions, convert_to_tensor=True)

# Vectorizador para TF-IDF y modelo SVM
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(questions)
classifier = SVC(kernel='linear')
classifier.fit(X, labels)

class SupportChatBot:
    def __init__(self, name, model, tokenizer, embed_model, vectorizer, classifier):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.embed_model = embed_model
        self.vectorizer = vectorizer
        self.classifier = classifier

    def generate_t5_response(self, question, category, context=""):
        # Create a more specific prompt in Spanish
        prompt = f"Genera una respuesta detallada en español para: {question}\nTema: {category}\nInformación relevante: {context}\nLa respuesta debe ser clara y específica."
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=5,
                temperature=0.6,  # Reduced for more focused responses
                top_k=30,
                top_p=0.85,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0  # Encourage more detailed responses
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure the response is meaningful
        if len(generated_text.split()) < 10 or not any(word in generated_text.lower() for word in question.lower().split()):
            return f"Para resolver tu consulta sobre {question}: {context}"
            
        return generated_text

    def process_question(self, user_question):
        user_vector = self.vectorizer.transform([user_question])
        predicted_index = self.classifier.predict(user_vector)[0]
        predicted_category = categories[predicted_index]

        filtered_questions = [q for q, _ in category_dict[predicted_category]]
        filtered_answers = [a for _, a in category_dict[predicted_category]]

        if not filtered_questions:
            return "Lo siento, no tengo información sobre ese tema específico."

        filtered_embeddings = self.embed_model.encode(filtered_questions, convert_to_tensor=True)
        user_embedding = self.embed_model.encode(user_question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_embedding, filtered_embeddings)[0]

        top_indices = torch.topk(similarities, min(3, len(similarities))).indices.tolist()
        best_responses = [(filtered_answers[i], similarities[i].item()) for i in top_indices]
        max_similarity = best_responses[0][1]

        print(best_responses)
        print(max_similarity)
        # If similarity is very low (< 0.2), use T5 generation
        if max_similarity < 0.4:
            context = " ".join([ans for ans, _ in best_responses])
            return self.generate_t5_response(user_question, predicted_category, context)
        
        # If similarity is moderate (0.2 - 0.4), combine top responses
        elif max_similarity < 0.7:
            combined_response = "Basado en preguntas similares: \n"
            for answer, score in best_responses[:2]:
                combined_response += f"- {answer}\n"
            return combined_response
        
        # If similarity is good (>= 0.4), return best match
        return best_responses[0][0]

# Crear instancia del chatbot
chatbot = SupportChatBot("DigiBite Assistant", t5_model, t5_tokenizer, embed_model, vectorizer, classifier)

# Configuración de FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    correct_answer: str = None  # Campo opcional para las correcciones

@app.post("/ask")
async def ask(request: QuestionRequest):
    user_input = request.question
    correct_answer = request.correct_answer
    
    if not user_input:
        raise HTTPException(status_code=400, detail="No se proporcionó ninguna pregunta")
    
    # Procesar la pregunta y obtener la respuesta
    response = chatbot.process_question(user_input)

    # Si se proporciona una corrección, actualizar el FAQ
    if correct_answer:
        chatbot.update_faq(user_input, correct_answer)
    
    return {"answer": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)