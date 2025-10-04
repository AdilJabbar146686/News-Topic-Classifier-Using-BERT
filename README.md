News Classification using BERT

This repository contains a project for classifying news headlines into multiple categories using a fine-tuned BERT model. The main focus is on training, evaluating, and saving a transformer-based model for news classification.

Repository Structure:

.
├── News_Classification.ipynb       # Jupyter Notebook for training and evaluation
├── requirements.txt               # Python dependencies
├── app.py                          # Optional Gradio demo for prediction
├── README.txt                       # Project documentation
└── data/                           # Optional folder for dataset

Features:

- Fine-tuning a BERT model on a news dataset (e.g., AG News or custom dataset)
- Predicting news categories: World, Sports, Business, Sci/Tech
- Optional Gradio web interface for interactive predictions
- Saves trained model for reuse without retraining

Dataset:

The notebook uses the AG News dataset by default, but you can replace it with any CSV/TSV file containing news headlines and labels.

Columns:  
- text: News headline  
- label: Category (World, Sports, Business, Sci/Tech)

Notebook Overview:

The News_Classification.ipynb notebook contains:

1. Data Loading & Exploration – Inspect dataset and check distribution of classes
2. Preprocessing – Clean text, tokenize using BERT tokenizer
3. Model Setup – Load pretrained BERT, set up classifier
4. Training & Evaluation – Train model with train/test split, evaluate accuracy
5. Saving Model – Save fine-tuned model for later inference
6. Inference – Predict category for a sample headline

Usage:

1. Install dependencies:

pip install -r requirements.txt

2. Run notebook:

Open News_Classification.ipynb and execute all cells to train the model.

3. Optional: Gradio App

python app.py

Input a news headline and get the predicted category interactively.

Dependencies:

- torch
- transformers
- gradio (optional)
- pandas
- scikit-learn
- numpy

Results:

- Model achieves reasonable accuracy on AG News dataset (can be replaced with your own dataset)
- Predictions are available in real-time via the Gradio demo (app.py)

References:

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (https://arxiv.org/abs/1810.04805)
- Hugging Face Transformers (https://huggingface.co/transformers/)
- AG News Dataset (https://www.kaggle.com/amananandrai/ag-news-classification-dataset)

License:

This project is licensed under the MIT License.

Author:

Adil Jabbar
- GitHub: [https://github.com/yourusername](https://github.com/AdilJabbar146686/News-Topic-Classifier-Using-BERT.git)
- Portfolio: developershub
