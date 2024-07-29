import streamlit as st
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import base64
import json
import time

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

file_path = 'label_info.json'

# Load the JSON data from the file
with open(file_path, 'r') as json_file:
    label_dict = json.load(json_file)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        texts=df.sentence.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

# Load and preprocess the dataset
def load_and_preprocess_data(df):
    # Handle missing values
    df = df.dropna(subset=['sentence'])
    
    # Text cleaning
    df['sentence'] = df['sentence'].apply(clean_text)
    
    return df

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Remove stopwords and punctuation, and lemmatize
    filtered_text = [
        lemmatizer.lemmatize(word.lower()) for word in word_tokens if word.isalnum() and word.lower() not in stop_words
    ]
    
    return ' '.join(filtered_text)

# Function to make predictions
def predict(text, model, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()

# Load the pre-trained BERT tokenizer and model
model_save_path = './saved_model'
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path)

# Move model to GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Streamlit app
st.title('Text Classification using BERT')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            df_dummy = df.copy()
            break
        except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
            st.write(f"Error loading file with encoding {encoding}: {e}")
            df = None
    
    st.write("Uploaded Data")
    st.write(df.head())

    with st.spinner('Processing...'):
        time.sleep(1)  # Adding a delay to simulate long processing times
        df = load_and_preprocess_data(df)
        test_data_loader = create_data_loader(df, tokenizer, 128, 16)

        predictions = []

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch in test_data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        label_predictions = [label_dict[str(prediction)] for prediction in predictions]

        # Add predictions to the dataframe
        df['predictions'] = label_predictions

        df['sentence'] = df_dummy['sentence']

        # Save the results
        results_file = 'results.csv'
        df.to_csv(results_file, index=False)

        st.write("Results")
        st.write(df.head())

        # Provide download link for the results
        def download_link(file_path, file_label):
            with open(file_path, "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{file_path}">{file_label}</a>'
                return href

        st.markdown(download_link(results_file, 'Download Results'), unsafe_allow_html=True)

    st.success('Processing complete!')