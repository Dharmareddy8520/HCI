# 🎭 Emotion Detection System using BERT (GoEmotions)

This project is a **multi-label emotion classification system** built using a fine-tuned BERT model trained on the [GoEmotions dataset by Google](https://github.com/google-research/goemotions). It can detect **28 nuanced emotions** from natural language input, such as gratitude, fear, excitement, sadness, and more.

## 📌 Key Features

- Fine-tuned **BERT (`bert-base-uncased`)** on GoEmotions (simplified).
- **Multi-label classification** (a sentence can have multiple emotions).
- Supports **CPU and GPU** (auto-detection).
- Interactive **web UI** for real-time input.
- Command-line script for quick local predictions.

---

## 📂 Project Structure

```
emotion-project/
├── emotion_model_training.ipynb  # Jupyter Notebook for training
├── emotion-model/                # Fine-tuned model & tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── ...
├── predict.py                    # Command-line emotion predictor
├── app.py                        # FastAPI web application
├── requirements.txt              # Project dependencies
├── templates/
│   └── index.html                # Web form frontend
├── static/
│   └── style.css                 # Optional styling
└── README.md                     # Project documentation
```

---

## 📘 1. `emotion_model_training.ipynb`

This Jupyter Notebook:

- Loads and preprocesses the **GoEmotions (simplified)** dataset.
- Tokenizes and binarizes the multi-label targets.
- Fine-tunes `bert-base-uncased` for emotion classification.
- Saves the trained model into the `emotion-model/` folder using `save_pretrained()`.
- Includes model evaluation, error analysis, and metric reporting (accuracy, F1, precision, recall).

Run this in **Google Colab** (with GPU enabled) or locally (with CUDA if available).

---

## 💬 2. `predict.py` – Terminal-Based Emotion Prediction

This Python script:

- Loads the fine-tuned model from `emotion-model/`.
- Accepts user input via the command line.
- Predicts emotion(s) with ≥ 0.5 probability.
- Runs on **GPU if available**, otherwise CPU.

### ▶️ Run:

```bash
python predict.py
```

Example Output:

```
Enter a sentence (or type 'exit' to quit'): I feel nervous and excited!
Predicted Emotions:
 - nervousness: 0.724
 - excitement: 0.842
```

---

## 🌐 3. `app.py` – Web Application (FastAPI)

This FastAPI-based app provides a web interface for real-time predictions.

- Sends form data from a web form.
- Displays predicted emotions and their confidence scores.
- Easy to run locally.

### ▶️ Run:

```bash
uvicorn app:app --reload
```

Then go to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Form example:

```
Enter text: I'm feeling grateful and just a little bit scared.
→ [Analyze]
Output:
 - gratitude: 0.813
 - fear: 0.502
```

---

## ⚙️ Installation

1. Create a virtual environment (recommended):

   ```bash
   python -m venv env
   source env/bin/activate        # macOS/Linux
   env\Scripts\activate           # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   ✅ Includes: `transformers`, `torch`, `fastapi`, `uvicorn`, `jinja2`, `python-multipart`.

---

## 🧠 Supported Emotions (28 Labels)

Examples include:

- admiration, amusement, anger, approval, caring, confusion, curiosity, desire, disappointment, gratitude, joy, fear, sadness, neutral, and more.

---

## 🛠️ Troubleshooting

### ❌ PyTorch not detecting GPU?

Install with CUDA support (e.g., for GTX 1650):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then check:

```python
import torch
print(torch.cuda.is_available())  # Should return True
```

---

## 📈 Future Ideas

- Visualize emotion scores using bar charts.
- Add a chat-style interface or Telegram/Discord bot.
- Deploy on Hugging Face Spaces or Render.
- Batch prediction from CSV files.

---
