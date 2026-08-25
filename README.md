# \# Personalized Chatbot

# 

# A hybrid retrieval-based chatbot developed as an internship project. The system combines TF-IDF lexical retrieval with MiniLM semantic retrieval to find relevant responses from a question-response knowledge base.

# 

# \## Project Repository

# 

# GitHub:

# https://github.com/Sunil-Bhukya-25/internsSunil\_INBT021913\_iNeuBytes

# 

# \## Features

# 

# \- Hybrid TF-IDF + MiniLM retrieval

# \- Semantic similarity using MiniLM embeddings

# \- Keyword matching using TF-IDF

# \- Confidence-based fallback for weak matches

# \- Flask backend

# \- HTML/CSS/JavaScript frontend

# \- REST API

# \- CPU-based inference support

# 

# \## Technologies Used

# 

# \- Python

# \- Flask

# \- HTML

# \- CSS

# \- JavaScript

# \- scikit-learn

# \- TF-IDF

# \- PyTorch

# \- Sentence Transformers

# \- MiniLM

# \- NumPy

# \- SciPy

# \- Joblib

# 

# \## System Architecture

# 

# User

# &#x20; ↓

# Web Frontend

# &#x20; ↓

# Flask Backend

# &#x20; ↓

# TF-IDF Retrieval + MiniLM Semantic Retrieval

# &#x20; ↓

# Hybrid Score

# &#x20; ↓

# Confidence Check

# &#x20; ↓

# Response / Fallback

# 

# \## Retrieval Method

# 

# Hybrid Score = (0.3 × TF-IDF Score) + (0.7 × MiniLM Semantic Score)

# 

# TF-IDF provides lexical/keyword matching, while MiniLM provides semantic similarity between differently worded queries.

# 

# \## Model Configuration

# 

# \- TF-IDF weight: 0.3

# \- MiniLM weight: 0.7

# \- MiniLM embedding dimension: 384

# \- Knowledge-base examples: 276,753

# \- Retrieval candidate count: 100

# \- Inference device: CPU

# 

# \## Confidence and Fallback

# 

# The chatbot checks whether the retrieved result is sufficiently relevant.

# 

# If the retrieved result has insufficient confidence, the system does not return a potentially unrelated answer. Instead, it returns a fallback response explaining that a sufficiently relevant answer could not be found.

# 

# The API also validates user input.

# 

# Examples:

# 

# Missing message:

# The 'message' field must be a string.

# 

# Empty message:

# message cannot be empty.

# 

# Whitespace-only message:

# message cannot be empty.

# 

# \## API

# 

# The main chatbot endpoint is:

# 

# POST /chat

# 

# Example request:

# 

# {

# &#x20; "message": "What is solar energy?"

# }

# 

# \## Running the Project

# 

# \### 1. Clone the repository

# 

# git clone https://github.com/Sunil-Bhukya-25/internsSunil\_INBT021913\_iNeuBytes.git

# 

# cd internsSunil\_INBT021913\_iNeuBytes

# 

# \### 2. Create a virtual environment

# 

# Windows:

# 

# python -m venv venv

# 

# venv\\Scripts\\activate

# 

# \### 3. Install dependencies

# 

# pip install -r requirements.txt

# 

# \### 4. Obtain model artifacts

# 

# The trained retrieval artifacts are large and are intentionally excluded from GitHub.

# 

# The required model files are documented in:

# 

# MODEL\_MANIFEST.txt

# 

# The helper script is:

# 

# download\_models.py

# 

# Run:

# 

# python download\_models.py

# 

# If the configured model repository requires authentication, the appropriate Hugging Face access must be provided.

# 

# \### 5. Start the Flask application

# 

# python app.py

# 

# Open:

# 

# http://127.0.0.1:5000

# 

# \## Project Structure

# 

# internsSunil\_INBT021913\_iNeuBytes/

# │

# ├── app.py

# ├── requirements.txt

# ├── README.md

# ├── .gitignore

# ├── download\_models.py

# ├── MODEL\_MANIFEST.txt

# │

# ├── static/

# │   ├── script.js

# │   └── style.css

# │

# └── templates/

# &#x20;   └── index.html

# 

# \## Model Files

# 

# The following retrieval artifacts are required by the final application:

# 

# tfidf\_vectorizer.pkl

# tfidf\_train\_matrix.npz

# semantic\_train\_embeddings\_fixed.npy

# semantic\_train\_indices\_fixed.npy

# semantic\_train\_responses\_fixed.json

# semantic\_embedding\_config\_fixed.json

# 

# These files are excluded from GitHub because of their large size.

# 

# \## Validation

# 

# The final system was tested for:

# 

# \- Normal chatbot questions

# \- Relevant response retrieval

# \- Low-confidence / unsupported questions

# \- Empty input

# \- Whitespace-only input

# \- Missing message field

# \- Flask server operation

# \- API operation

# \- Postman API testing

# 

# Final retrieval validation:

# 

# Relevant questions: 6/6

# 

# Problematic questions rejected: 4/4

# 

# \## Internship Project

# 

# Repository naming follows the required internship format:

# 

# interns\[FirstName]\_RegdNo\_iNeuBytes

# 

# Final repository:

# 

# internsSunil\_INBT021913\_iNeuBytes

# 

# \## Author

# 

# Sunil Bhukya

