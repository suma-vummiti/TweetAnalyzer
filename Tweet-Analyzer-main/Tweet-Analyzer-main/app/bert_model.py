from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def analyze_sentiment(text):
    # Tokenize the input text and convert it to tensor
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted sentiment
    sentiment = torch.argmax(logits, dim=1).item()

    # Return corresponding sentiment based on the model's output
    # The model returns values from 0 to 4, where:
    # 0: very negative, 1: negative, 2: neutral, 3: positive, 4: very positive
    sentiments = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    return sentiments[sentiment]
