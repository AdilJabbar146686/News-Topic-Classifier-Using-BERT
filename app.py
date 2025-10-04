import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load Pretrained Model
model_name = "textattack/roberta-base-ag-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # evaluation mode
device = torch.device("cpu")
model.to(device)

# AG News labels
labels = ['World', 'Sports', 'Business', 'Sci/Tech']

# Prediction Function
def predict_news_category(text):
    if not text.strip():
        return "Please enter a valid news headline."
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    predicted_class_id = int(torch.argmax(logits, dim=1))
    
    prob_text = "\n".join([f"{labels[i]}: {probs[i]*100:.2f}%" for i in range(len(labels))])
    
    return f"📰 **Predicted Category:** {labels[predicted_class_id]}\n\n**Probabilities:**\n{prob_text}"

# Gradio Interface
title = "🧠 BERT News Topic Classifier"
description = """
Classify news headlines into one of four categories:
- 🌍 World  
- ⚽ Sports  
- 💼 Business  
- 🔬 Science/Technology  
"""

iface = gr.Interface(
    fn=predict_news_category,
    inputs=gr.Textbox(lines=2, placeholder="Enter a news headline..."),
    outputs="markdown",
    title=title,
    description=description,
    examples=[
        ["Stock markets rally after Federal Reserve announcement"],
        ["NASA discovers water on Mars"],
        ["Pakistan wins ICC World Cup 2025"],
        ["Global leaders meet to discuss climate change policy"]
    ],
)

if __name__ == "__main__":
    iface.launch()