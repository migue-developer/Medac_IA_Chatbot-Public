import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model = RobertaForQuestionAnswering.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

faq_file = 'faq.json'
with open(faq_file, 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

category_dict = defaultdict(list)
categories = []
questions = []
answers = []

for i, entry in enumerate(faq_data):
    category_dict[entry['category']].append((entry['question'], entry['answer']))
    questions.append(entry['question'])
    answers.append(entry['answer'])
    if entry['category'] not in categories:
        categories.append(entry['category'])

question_embeddings = embed_model.encode(questions, convert_to_tensor=True)

class SupportChatBot:
    def __init__(self, name, model, tokenizer, embed_model):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.embed_model = embed_model

    def answer_question(self, user_question, context):
        inputs = tokenizer(user_question, context, return_tensors="pt", truncation=True, max_length=150)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_index = outputs.start_logits.argmax()
        end_index = outputs.end_logits.argmax()

        answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Usar categoría por defecto "General"
        predicted_category = chatbot.get_category()
        
        # Guardar la respuesta con la categoría predicha
        new_entry = {
            'question': user_question,
            'answer': answer,
            'category': predicted_category
        }

        faq_data.append(new_entry)
        
        with open(faq_file, 'w', encoding='utf-8') as f:
            json.dump(faq_data, f, ensure_ascii=False, indent=4)

        return answer

    def process_question(self, user_question):
        user_embedding = self.embed_model.encode([user_question], convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
        
        top_indices = torch.topk(similarities, min(3, len(similarities))).indices.tolist()
        
        best_responses = [(answers[i], similarities[i].item()) for i in top_indices]
        
        max_similarity = best_responses[0][1]
        if max_similarity < 0.7:
            context = " ".join(answers)
            return self.answer_question(user_question, context), None
        
        return best_responses[0][0], None

    def get_category(self):
        # Categoría por defecto
        return "General"

chatbot = SupportChatBot("DigiBite Assistant", model, tokenizer, embed_model)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    user_input = request.question
    
    if not user_input:
        raise HTTPException(status_code=400, detail="No se proporcionó ninguna pregunta")
    
    response, _ = chatbot.process_question(user_input)
    predicted_category = chatbot.get_category()
    
    return {"answer": response, "category": predicted_category}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
