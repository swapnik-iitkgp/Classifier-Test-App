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
import time

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


label_dict = {
    0: "Irrelevant",
    1: "Call opening greeting",
    2: "Call closing greeting",
    3: "Purpose of Call",
    4: "Did Rep Ask The Spelling Of Customer Name?",
    5: "Did Rep Ask The Exact Customer Address?",
    6: "Was The Policy Disclaimer Readout?",
    7: "Was The Customer Asked For Secondary & Next Of Kin Contact Numbers?",
    8: "Was Client Residency Information Taken?",
    9: "Did Rep ask for Date of Birth?",
    10: "Was Policy Cover Cost Given?",
    11: "Policy Benefits & Limits(Exclusion Conditions)",
    12: "Was Cover Description Given To Customer?",
    13: "Cancellation of Policy",
    14: "Call Hold",
    15: "Did Rep Informed The Customer About Call Transfer?",
    16: "Did Rep Explain Renewal Notification (Policy Contract)?",
    17: "Did The Rep Read Out All Accidental Serious Injuries Cover?",
    18: "Did The Rep Share The Policy Number With The Client?",
    19: "Provision Of Disclosure Documents (General Advice Only Clients)",
    20: "Whether Accident & Sickness Policy Cover Disclaimer Readout Under Accident & Sickness Insurance?",
    21: "Denial of claim-exclusion/condition",
    22: "Did The Rep Explain The Item Description Needs At The Time Of Claim?",
    23: "Call Recording Disclaimer",
    24: "General Advice Warning",
    25: "Was Policy Cover Confirmation Asked And Confirmed?",
    26: "Did Rep Say He Will Be Sharing Policy Copy?",
    27: "Confirmation to start policy",
    28: "Did The Rep Correctly Get The Travel Date?",
    29: "Health exclusion",
    30: "Did Rep Ask About Pre-Existing Condition Of The Client?",
    31: "Pre-Screening Medical Questionnaire",
    32: "Estimation of premium disclaimer",
    33: "Was Travel Policy Cover Disclaimer Readout?",
    34: "Were Current Bank Account Details Taken?",
    35: "Duty of disclosure",
    36: "Consent to proceed"
}

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
    df = pd.read_csv(uploaded_file)
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

        label_predictions = [label_dict[prediction] for prediction in predictions]

        # Add predictions to the dataframe
        df['predictions'] = label_predictions

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