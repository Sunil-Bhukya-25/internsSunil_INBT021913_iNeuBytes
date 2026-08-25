# Personalized Chatbot

A hybrid retrieval-based chatbot that combines TF-IDF lexical retrieval with MiniLM semantic retrieval. The application uses a Flask backend and a browser-based HTML/CSS/JavaScript frontend.

## 1. Project Overview

The chatbot searches a question–answer knowledge base and returns a stored response when the retrieved result is sufficiently relevant.

Final configuration:

| Item | Value |
|---|---|
| Retrieval model | Hybrid TF-IDF + MiniLM |
| TF-IDF weight | 0.3 |
| MiniLM weight | 0.7 |
| MiniLM embedding size | 384 |
| Candidate count | 100 |
| Knowledge-base examples | 276,753 |
| Backend | Python Flask |
| Frontend | HTML, CSS, JavaScript |
| API testing | Postman |
| Inference | CPU supported |

The hybrid score is:

`Hybrid Score = (0.3 × TF-IDF Score) + (0.7 × MiniLM Score)`

The system also performs a confidence check. If both the semantic and lexical evidence are weak, it returns a safe fallback instead of an unrelated stored response.

## 2. Dataset

### Dataset used

The project was developed using the **Ubuntu Dialogue Corpus** as the source conversational dataset.

Official dataset information:
https://www.i2c2.aut.ac.nz/ubuntu-corpus/

The project preprocessing pipeline converted the source conversations into the question–response knowledge base used by the retrieval system.

Final processed knowledge base:

- 276,753 training examples
- Stored question/response pairs
- TF-IDF representation for lexical retrieval
- MiniLM embeddings for semantic retrieval

### Data processing

```text
Ubuntu Dialogue Corpus
        ↓
Question / Response extraction
        ↓
Text processing
        ↓
TF-IDF + MiniLM embeddings
        ↓
Aligned retrieval artifacts
        ↓
Hybrid chatbot
```

## 3. Model / Retrieval Artifacts

This project does not use a single generative LLM checkpoint. The deployed retrieval system uses indexed artifacts:

```text
models/
├── tfidf_vectorizer.pkl
├── tfidf_train_matrix.npz
├── semantic_train_embeddings_fixed.npy
├── semantic_train_indices_fixed.npy
├── semantic_train_responses_fixed.json
└── semantic_embedding_config_fixed.json
```

The local model artifacts are large and are intentionally not stored in the GitHub source repository.

### Model artifact repository

The project model artifacts were uploaded to the Hugging Face Dataset repository:

`Sunil-25-10/personalized-chatbot-models`

Before using the clean-clone instructions below, make sure this Hugging Face repository is accessible to the evaluator. For a no-login clean-clone workflow, make the model repository **public**.

If the repository remains private, a Hugging Face access token with read permission is required.

## 4. Clean Clone Setup

### Step 1 — Clone the GitHub repository

```bash
git clone https://github.com/Sunil-Bhukya-25/Personalized-Chatbot.git
cd Personalized-Chatbot
```

### Step 2 — Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the retrieval artifacts

Run:

```bash
python download_models.py
```

The script downloads the six required retrieval artifacts into:

```text
models/
```

If the Hugging Face repository is private, authenticate first using a Hugging Face token with read access.

### Step 5 — Start the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 5. Manual Model Loading Option

If automatic download is not available, obtain these six files from the project's model-artifact repository and place them directly inside `models/`:

```text
tfidf_vectorizer.pkl
tfidf_train_matrix.npz
semantic_train_embeddings_fixed.npy
semantic_train_indices_fixed.npy
semantic_train_responses_fixed.json
semantic_embedding_config_fixed.json
```

The Flask application checks for the required files during startup.

## 6. Requirements

Install:

```bash
pip install -r requirements.txt
```

Main libraries:

- Flask
- flask-cors
- NumPy
- SciPy
- scikit-learn
- joblib
- PyTorch
- sentence-transformers
- huggingface_hub
- gunicorn

## 7. How the System Works

```text
User
  ↓
Browser Chat Interface
  ↓
Flask API
  ↓
Retrieve candidates
  ↓
TF-IDF score + MiniLM semantic score
  ↓
Normalize and combine scores
  ↓
Hybrid ranking
  ↓
Confidence check
  ├── Sufficient confidence → stored response
  └── Low confidence → fallback message
```

TF-IDF provides lexical/keyword matching. MiniLM provides semantic similarity so that differently worded questions can still be related.

## 8. Confidence / Fallback

The system checks the original semantic and lexical similarity values rather than relying only on the normalized hybrid score.

If both are below their configured fallback thresholds, the system returns:

> Sorry, I couldn't find a sufficiently relevant answer to that question in my knowledge base. Please try asking about one of the topics supported by the chatbot.

This prevents a weak nearest result from being presented as a confident answer.

## 9. API

Example request:

```http
POST /respond
Content-Type: application/json
```

```json
{
  "message": "What is solar energy?"
}
```

The API returns the selected response together with retrieval information such as model, scores, confidence status, and retrieval time.

Health check:

```http
GET /
```

## 10. Postman Testing

Postman was used to test the Flask API independently of the browser interface.

The project includes Postman testing material in:

```text
postman/
```

The health check was successfully verified with HTTP 200 OK.

## 11. Validation

Final retrieval validation:

- Relevant questions matched: **6/6**
- Problematic questions rejected: **4/4**

The project was also verified through:

- Local Flask execution
- Browser chatbot testing
- Postman API testing
- Confidence/fallback demonstration
- GitHub repository verification

## 12. Deployment

The application was successfully demonstrated publicly using a Cloudflare Quick Tunnel connected to the local Flask server.

Important: the Quick Tunnel is a temporary demonstration method. It depends on the local Flask server and tunnel process remaining active and is not the permanent production deployment.

## 13. Project Structure

```text
Personalized-Chatbot/
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── download_models.py
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── postman/
└── reports/
```

## 14. Large Model Files

The retrieval artifacts are intentionally excluded from GitHub because of their large size.

This separation keeps the source repository manageable while allowing the model artifacts to be obtained separately.

The exact required files are documented in:

```text
MODEL_MANIFEST.txt
```

## 15. Project Repository

GitHub:

https://github.com/Sunil-Bhukya-25/Personalized-Chatbot

Model artifacts:

`Sunil-25-10/personalized-chatbot-models`

## 16. Author

Sunil Bhukya
