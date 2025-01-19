import os
import pdfplumber
import pandas as pd
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API using the API key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))      #replace with your gemini api key

# Initialize Flask app
app = Flask(__name__)

# Google Drive API authentication
def authenticate_google_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("credentials.json")

    drive = GoogleDrive(gauth)
    return drive

# Function to get PDF files from a Google Drive folder
def get_pdf_files_from_drive(folder_id, drive):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    pdf_files = [file for file in file_list if file["title"].endswith(".pdf")]
    return pdf_files

# Function to download PDF files from Google Drive
def download_files_from_drive(pdf_files, download_folder, drive):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    downloaded_files = []
    for pdf in pdf_files:
        file_name = pdf["title"]
        file_id = pdf["id"]
        file_path = os.path.join(download_folder, file_name)
        file = drive.CreateFile({'id': file_id})
        file.GetContentFile(file_path)
        downloaded_files.append(file_path)

    return downloaded_files

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

# Function to query the Gemini API
import json

def query_gemini_api(text):
    prompt = (
        "Please extract the following details from the resume text below and return them as a JSON object:\n"
        "1. name\n"
        "2. contact_details (e.g., phone, email, address)\n"
        "3. university\n"
        "4. year_of_study\n"
        "5. course\n"
        "6. discipline\n"
        "7. cgpa_percentage\n"
        "8. key_skills (comma separated list)\n"
        "9. supporting_info (additional relevant information, if any)\n\n"
        "Input text:\n"
        f"{text}\n\n"
        "Output JSON should be structured like:\n"
        "{\n"
        "  'name': 'Full Name',\n"
        "  'contact_details': 'Contact info',\n"
        "  'university': 'University name',\n"
        "  'year_of_study': 'Year',\n"
        "  'course': 'Course name',\n"
        "  'discipline': 'Discipline',\n"
        "  'cgpa_percentage': 'CGPA or percentage',\n"
        "  'key_skills': ['skill1', 'skill2', 'skill3'],\n"
        "  'supporting_info': 'Additional info'\n"
        "}"
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Clean up the response and parse as JSON
    try:
        # Remove code block markers if present
        cleaned_response = response.text.strip().strip('```json').strip('```')
        # Parse JSON string into Python dictionary
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini API response: {response.text}") from e



# Function to calculate experience scores based on keyword matches
def calculate_experience_score(text, category="gen_ai"):
    keywords = {
        "gen_ai": [
            "GPT", "RAG", "Generative AI", "Agentic", "Evals", "Large Language Models", "LLMs", "ChatGPT",
            "BERT", "T5", "Transformer models", "OpenAI", "Stable Diffusion", "DALL-E", "MidJourney", "Claude",
            "Anthropic", "Hugging Face", "LangChain", "AutoGPT", "BabyAGI", "Text generation", "Image generation",
            "Prompt engineering", "Diffusion models", "Foundation models", "Semantic search", "Vector databases",
            "Embeddings", "Retrieval-augmented generation", "Instruction-tuned models", "Generative pretrained transformer",
            "Few-shot learning", "Zero-shot learning", "Tokenization", "Whisper", "Image-to-text models",
            "RLHF", "Reinforcement Learning with Human Feedback", "Fine-tuning", "Open-source models",
            "Bloom", "OPT", "Megatron", "DeepSpeed", "LoRA", "Parameter-efficient fine-tuning",
            "NeMo", "Mistral", "Falcon", "Vicuna", "Baize", "FLAN-T5", "LLama", "ALBERT", "XLNet",
        ],
        "ai_ml": [
            "machine learning", "deep learning", "neural networks", "AI", "Artificial Intelligence",
            "Reinforcement learning", "Supervised learning", "Unsupervised learning", "Self-supervised learning",
            "Feature engineering", "PyTorch", "TensorFlow", "Scikit-learn", "Keras", "Explainable AI", "Bias detection",
            "Hyperparameter tuning", "Gradient boosting", "Random forest", "Support vector machines", "XGBoost",
            "LightGBM", "CatBoost", "Convolutional neural networks", "Recurrent neural networks",
            "Long short-term memory", "Transformers", "GANs", "Generative adversarial networks", "Autoencoders",
            "Decision trees", "Clustering", "K-means", "PCA", "Principal component analysis", "Dimensionality reduction",
            "Bayesian networks", "Naive Bayes", "Logistic regression", "Linear regression", "Gradient descent",
            "Backpropagation", "Activation functions", "Dropout", "Batch normalization", "Attention mechanisms",
            "Graph neural networks", "GNNs", "Natural language processing", "NLP", "Speech recognition",
            "Computer vision", "Image classification", "Object detection", "YOLO", "ResNet", "EfficientNet",
            "Time series analysis", "Sequence modeling", "Markov chains", "Anomaly detection", "Recommendation systems",
            "Predictive analytics", "Knowledge graphs", "Data preprocessing", "Data augmentation", "Synthetic data",
        ]
    }

    score = 0
    for keyword in keywords[category]:
        if keyword.lower() in text.lower():
            score += 1
    return min(score, 3)  # Cap the score at 3

# Function to process a batch of resumes
def process_batch(resume_paths):
    raw_results = []
    processed_results = []

    for resume_path in resume_paths:
        text = extract_text_from_pdf(resume_path)
        extracted_data = query_gemini_api(text)

        # Append raw extracted data for the first file
        raw_results.append(extracted_data)

        # Calculate experience scores for the text
        gen_ai_score = calculate_experience_score(text, "gen_ai")
        ai_ml_score = calculate_experience_score(text, "ai_ml")

        processed_results.append({
            **extracted_data,  # Include all extracted data fields
            "gen_ai_experience": gen_ai_score,
            "ai_ml_experience": ai_ml_score
        })

    return raw_results, processed_results

# Save raw data and processed data to Excel
def save_raw_data_to_excel(data, output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

def save_processed_data_to_excel(data, output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

# Web interface for user to enter Google Drive Folder ID
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    folder_id = request.form['folder_id']                   # Get the folder ID from the google drive folder's link (see README for more clarity)              
    download_folder = "resumes"
    raw_output_file = "raw_extracted_resumes.xlsx"
    processed_output_file = "processed_resumes_with_scores.xlsx"

    # Authenticate and fetch files from Google Drive
    drive = authenticate_google_drive()
    pdf_files = get_pdf_files_from_drive(folder_id, drive)
    if not pdf_files:
        return "No PDF files found in the folder. Please check the folder ID."

    downloaded_files = download_files_from_drive(pdf_files, download_folder, drive)

    raw_data, processed_data = process_batch(downloaded_files)
    save_raw_data_to_excel(raw_data, raw_output_file)
    save_processed_data_to_excel(processed_data, processed_output_file)

    return f"Process completed. Raw data saved to '{raw_output_file}', and processed data saved to '{processed_output_file}'."

if __name__ == '__main__':
    app.run(debug=True)
