from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize app and template renderer
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Emotion labels from GoEmotions (simplified)
label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

# Load tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("emotion-model")
model = AutoModelForSequenceClassification.from_pretrained("emotion-model")

# GET route to show the form
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# POST route to process input and return predictions
@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, text: str = Form(...)):
    # Tokenize user input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Predict emotion probabilities
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    # Return emotions with prob ≥ 0.5
    predictions = [(label, round(float(prob), 3)) for label, prob in zip(label_names, probs) if prob >= 0.5]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": predictions,
        "text": text
    })
