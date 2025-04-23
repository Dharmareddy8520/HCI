import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# List of GoEmotions simplified labels
label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emotion-model")
model = AutoModelForSequenceClassification.from_pretrained("emotion-model").to(device)
model.eval()

# Function to predict emotions
def predict_emotions(text, threshold=0.5):
    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Inference without gradient
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    # Filter by threshold
    results = [(label, round(float(prob), 3)) for label, prob in zip(label_names, probs) if prob >= threshold]
    return results

# ---- MAIN ----
if __name__ == "__main__":
    while True:
        text = input("\nEnter a sentence (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        emotions = predict_emotions(text)
        if emotions:
            print("\nPredicted Emotions:")
            for label, score in emotions:
                print(f" - {label}: {score}")
        else:
            print("No strong emotions detected.")
