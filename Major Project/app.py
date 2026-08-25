
import os
import json
import time

import joblib
import numpy as np
import scipy.sparse as sp
import torch

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    APP_DIR
)

MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "models"
)

TEMPLATES_DIR = os.path.join(
    APP_DIR,
    "templates"
)

STATIC_DIR = os.path.join(
    APP_DIR,
    "static"
)


# ============================================================
# MODEL FILES
# ============================================================

TFIDF_VECTOR_FILE = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

TFIDF_MATRIX_FILE = os.path.join(
    MODEL_DIR,
    "tfidf_train_matrix.npz"
)

MINILM_EMBEDDING_FILE = os.path.join(
    MODEL_DIR,
    "semantic_train_embeddings_fixed.npy"
)

MINILM_RESPONSE_FILE = os.path.join(
    MODEL_DIR,
    "semantic_train_responses_fixed.json"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

TFIDF_WEIGHT = 0.30
MINILM_WEIGHT = 0.70
TOP_K = 100


# ============================================================
# RETRIEVAL FALLBACK CONFIGURATION
# ============================================================

# These values are intentionally conservative.
#
# The raw MiniLM score is used as the main semantic signal.
# TF-IDF is used as a supporting lexical signal.
# The top-1/top-2 margin is used to detect ambiguous matches.

MINILM_MIN_SCORE = 0.45
TFIDF_MIN_SCORE = 0.05

MINILM_STRONG_SCORE = 0.60
TFIDF_STRONG_SCORE = 0.20

MINILM_MARGIN = 0.035
TFIDF_MARGIN = 0.025


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("PERSONALIZED CHATBOT — FLASK SERVER")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [

    TFIDF_VECTOR_FILE,

    TFIDF_MATRIX_FILE,

    MINILM_EMBEDDING_FILE,

    MINILM_RESPONSE_FILE
]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required model file not found:\n"
            f"{file_path}"
        )


# ============================================================
# LOAD TF-IDF
# ============================================================

print("\nLoading TF-IDF...")

tfidf_vectorizer = joblib.load(
    TFIDF_VECTOR_FILE
)

tfidf_matrix = sp.load_npz(
    TFIDF_MATRIX_FILE
)

print(
    "TF-IDF matrix:",
    tfidf_matrix.shape
)


# ============================================================
# LOAD MINILM INDEX
# ============================================================

print("\nLoading corrected MiniLM index...")

minilm_embeddings = np.load(
    MINILM_EMBEDDING_FILE,
    mmap_mode="r"
)

with open(
    MINILM_RESPONSE_FILE,
    "r",
    encoding="utf-8"
) as f:

    minilm_responses = json.load(f)


print(
    "MiniLM embeddings:",
    minilm_embeddings.shape
)

print(
    "Responses:",
    len(minilm_responses)
)


# ============================================================
# ALIGNMENT CHECK
# ============================================================

if (
    minilm_embeddings.shape[0]
    != len(minilm_responses)
):

    raise RuntimeError(
        "MiniLM embeddings and responses "
        "are not aligned."
    )


if (
    minilm_embeddings.shape[0]
    != tfidf_matrix.shape[0]
):

    raise RuntimeError(
        "TF-IDF and MiniLM indexes "
        "are not aligned."
    )


print("Index alignment: OK")


# ============================================================
# LOAD MINILM
# ============================================================

print("\nLoading MiniLM...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

print(
    "MiniLM device:",
    model.device
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

CORS(app)


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    minimum = float(
        scores.min()
    )

    maximum = float(
        scores.max()
    )

    difference = (
        maximum - minimum
    )

    if difference < 1e-8:

        return np.zeros_like(
            scores,
            dtype=np.float32
        )

    return (
        scores - minimum
    ) / difference


# ============================================================
# ERROR RESPONSE
# ============================================================

def error_response(
    message,
    status_code=400
):

    return jsonify({
        "error": message
    }), status_code


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "online",

        "model":
            "Hybrid TF-IDF + MiniLM",

        "device":
            str(model.device),

        "tfidf_weight":
            TFIDF_WEIGHT,

        "minilm_weight":
            MINILM_WEIGHT,

        "candidate_count":
            TOP_K
    })


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    request_start = time.time()


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict
    ):

        return error_response(
            "Request body must be valid JSON.",
            400
        )


    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    question = data.get(
        "message"
    )

    if not isinstance(
        question,
        str
    ):

        return error_response(
            "The 'message' field must be a string.",
            400
        )


    question = question.strip()


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not question:

        return error_response(
            "message cannot be empty.",
            400
        )


    # --------------------------------------------------------
    # LENGTH CHECK
    # --------------------------------------------------------

    if len(question) > 1000:

        return error_response(
            "Message is too long. "
            "Please keep it under 1000 characters.",
            400
        )


    # ========================================================
    # TF-IDF RETRIEVAL
    # ========================================================

    query_tfidf = (
        tfidf_vectorizer.transform(
            [question]
        )
    )


    tfidf_scores = (
        query_tfidf
        @
        tfidf_matrix.T
    ).toarray().ravel()


    # ========================================================
    # TOP TF-IDF CANDIDATES
    # ========================================================

    candidate_count = min(
        TOP_K,
        len(tfidf_scores)
    )


    if candidate_count == 0:

        return error_response(
            "No indexed responses are available.",
            500
        )


    candidates = np.argpartition(
        tfidf_scores,
        -candidate_count
    )[-candidate_count:]


    # ========================================================
    # MINILM
    # ========================================================

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]


    candidate_embeddings = np.asarray(
        minilm_embeddings[
            candidates
        ],
        dtype=np.float32
    )


    minilm_scores = (
        candidate_embeddings
        @
        query_embedding
    )


    # ========================================================
    # CANDIDATE SCORES
    # ========================================================

    candidate_tfidf = (
        tfidf_scores[
            candidates
        ]
    )


    # ========================================================
    # SORT BY MINILM
    # ========================================================

    semantic_order = np.argsort(
        minilm_scores
    )[::-1]


    best_semantic_position = int(
        semantic_order[0]
    )


    second_semantic_position = int(
        semantic_order[1]
    )


    best_minilm_score = float(
        minilm_scores[
            best_semantic_position
        ]
    )


    second_minilm_score = float(
        minilm_scores[
            second_semantic_position
        ]
    )


    minilm_margin = (
        best_minilm_score
        -
        second_minilm_score
    )


    # ========================================================
    # SORT BY TF-IDF
    # ========================================================

    lexical_order = np.argsort(
        candidate_tfidf
    )[::-1]


    best_lexical_position = int(
        lexical_order[0]
    )


    second_lexical_position = int(
        lexical_order[1]
    )


    best_tfidf_score = float(
        candidate_tfidf[
            best_lexical_position
        ]
    )


    second_tfidf_score = float(
        candidate_tfidf[
            second_lexical_position
        ]
    )


    tfidf_margin = (
        best_tfidf_score
        -
        second_tfidf_score
    )


    # ========================================================
    # HYBRID RANKING
    # ========================================================

    tfidf_norm = minmax_normalize(
        candidate_tfidf
    )

    minilm_norm = minmax_normalize(
        minilm_scores
    )


    hybrid_scores = (
        TFIDF_WEIGHT
        *
        tfidf_norm

        +

        MINILM_WEIGHT
        *
        minilm_norm
    )


    best_position = int(
        np.argmax(
            hybrid_scores
        )
    )


    best_index = int(
        candidates[
            best_position
        ]
    )


    best_hybrid_score = float(
        hybrid_scores[
            best_position
        ]
    )


    # ========================================================
    # CONFIDENCE DECISION
    # ========================================================
    #
    # We use three signals:
    #
    # 1. Raw semantic similarity
    # 2. Raw lexical similarity
    # 3. Difference between the best and second-best
    #
    # Strong matches are accepted.
    #
    # Clearly weak matches are rejected.
    #
    # Ambiguous matches are rejected unless they have strong
    # semantic/lexical evidence.
    # ========================================================

    semantic_strong = (
        best_minilm_score
        >=
        MINILM_STRONG_SCORE
    )


    lexical_strong = (
        best_tfidf_score
        >=
        TFIDF_STRONG_SCORE
    )


    semantic_weak = (
        best_minilm_score
        <
        MINILM_MIN_SCORE
    )


    lexical_weak = (
        best_tfidf_score
        <
        TFIDF_MIN_SCORE
    )


    semantic_ambiguous = (
        minilm_margin
        <
        MINILM_MARGIN
    )


    lexical_ambiguous = (
        tfidf_margin
        <
        TFIDF_MARGIN
    )


    # --------------------------------------------------------
    # ACCEPTANCE LOGIC
    # --------------------------------------------------------

    # --------------------------------------------------------
    # FINAL CONFIDENCE DECISION
    # --------------------------------------------------------
    #
    # Strong semantic matches are accepted even when the
    # top-1/top-2 margin is small.
    #
    # Clearly weak matches are rejected.
    #
    # Ambiguous medium-strength matches are rejected when the
    # best semantic candidate is only slightly better than
    # the next candidate.
    #
    # This prevents cases such as:
    #
    # "latest cricket score"
    # "tell me a joke about a dog"
    #
    # from returning unrelated but vaguely similar answers.
    # --------------------------------------------------------

    if semantic_strong:

        # Strong semantic evidence is enough.
        accept_match = True

    elif semantic_weak:

        # Clearly weak semantic evidence.
        accept_match = False

    elif (
        semantic_ambiguous
        and
        lexical_ambiguous
    ):

        # Medium semantic score + very small separation
        # from competing candidates = uncertain retrieval.
        accept_match = False

    elif (
        best_minilm_score
        >=
        MINILM_MIN_SCORE
        and
        best_tfidf_score
        >=
        TFIDF_MIN_SCORE
        and
        not semantic_ambiguous
    ):

        # Reasonably strong combined evidence.
        accept_match = True

    else:

        accept_match = False


    # ========================================================
    # RESPONSE
    # ========================================================

    if accept_match:

        response = str(
            minilm_responses[
                best_index
            ]
        )

        confidence = "matched"

    else:

        response = (
            "Sorry, I couldn't find a sufficiently "
            "relevant answer to that question in my "
            "knowledge base. Please try asking about "
            "one of the topics supported by the chatbot."
        )

        confidence = "low_confidence"


    # ========================================================
    # TIME
    # ========================================================

    retrieval_time = (
        time.time()
        -
        request_start
    )


    # ========================================================
    # JSON RESPONSE
    # ========================================================

    return jsonify({

        "message":
            question,

        "response":
            response,

        "model":
            "Hybrid TF-IDF + MiniLM",

        "tfidf_weight":
            TFIDF_WEIGHT,

        "minilm_weight":
            MINILM_WEIGHT,

        "tfidf_score":
            best_tfidf_score,

        "minilm_score":
            best_minilm_score,

        "hybrid_score":
            best_hybrid_score,

        "minilm_margin":
            float(minilm_margin),

        "tfidf_margin":
            float(tfidf_margin),

        "confidence":
            confidence,

        "retrieval_time_seconds":
            float(retrieval_time)
    })


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    print(
        "Internal server error:",
        error
    )

    return jsonify({

        "error":
            "An internal server error occurred."
    }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("Starting Personalized Chatbot...")
    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
