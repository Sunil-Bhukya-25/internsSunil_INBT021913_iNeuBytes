# internsSunil_INBT021913_iNeuBytes

## AI/ML Internship Projects

This repository contains the work completed by **Sunil** during the **iNeuBytes AI/ML Internship**.

The repository contains:

1. **Task 1 — CIFAR-10 Image Classification using CNN**
2. **Task 2 — IMDb Sentiment Analysis using Machine Learning and Deep Learning**
3. **Major Project — Personalized Chatbot**

The projects include source code, Jupyter notebooks, project documentation, and application files.

---

# Author

- **Name:** Sunil
- **Internship ID:** INBT021913
- **Organization:** iNeuBytes

---

# Repository Structure

```text
internsSunil_INBT021913_iNeuBytes/
│
├── Task 1/
│   └── CIFAR10_CNN.ipynb
│
├── Task 2/
│   └── Sentimental_Analysis.ipynb
│
├── Major Project/
│   ├── static/
│   ├── templates/
│   ├── .gitignore
│   ├── MODEL_MANIFEST.txt
│   ├── README.md
│   ├── app.py
│   ├── download_models.py
│   └── requirements.txt
│
├── .gitignore
└── README.md
Task 1 — CIFAR-10 Image Classification
Overview

Task 1 focuses on image classification using a Convolutional Neural Network (CNN) with the CIFAR-10 dataset.

The experiments investigate:

CNN baseline architecture
Data preprocessing
Pixel normalization
Data augmentation
Batch Normalization
Dropout
L2 regularization
Optimizer comparison
Padding strategies
Increasing CNN depth
Final model selection
Confusion-matrix analysis

A fixed random seed of 42 was used.

Dataset

CIFAR-10 contains:

60,000 RGB images
Image size: 32 × 32 × 3
50,000 training images
10,000 test images
10 classes

The training data was divided into:

Training   : 45,000
Validation : 5,000
Test       : 10,000

Pixel values were normalized from 0–255 to 0–1.

Baseline CNN

The baseline CNN used:

64 → 128 → 256 filters
Baseline Results
MetricResult
Training Accuracy95.28%
Validation Accuracy70.92%
Test Accuracy70.47%
Precision70.69%
Recall70.47%
F1-Score70.42%
Training Time98.26 seconds
Parameters503,306
CNN Experiments
ExperimentTest Accuracy
Baseline CNN70.47%
L2 Regularization70.21%
Light Augmentation73.88%
Moderate Augmentation71.94%
Aggressive Augmentation11.13%
Batch Normalization72.99%
Dropout72.51%
SGD + Momentum65.82%
Adam69.37%
RMSprop71.45%
Same Padding74.69%
Extra Conv Block68.48%
Final CNN75.88%
Final Task 1 Result

The final selected CNN achieved:

75.88% test accuracy

The baseline accuracy was 70.47%.

Improvement:

75.88% - 70.47% = 5.41 percentage points

The final model was also evaluated using precision, recall, F1-score and a confusion matrix.

Notebook
Task 1/
└── CIFAR10_CNN.ipynb
Task 2 — IMDb Sentiment Analysis
Overview

Task 2 focuses on binary sentiment classification using IMDb movie reviews.

The task compares:

Classical Machine Learning
TF-IDF
Logistic Regression
Linear SVM
Unigram features
Bigram features
Vocabulary-size experiments
LSTM networks
Stacked LSTM
Dropout
Model complexity
Misclassification analysis
Confusion matrices
Dataset

The IMDb dataset contains:

50,000 movie reviews
25,000 positive reviews
25,000 negative reviews

The experiments used:

Training   : 20,000
Validation : 5,000
Test       : 25,000

The random seed was fixed to 42.

Text Preprocessing

The preprocessing included:

Lowercasing
Removing IMDb special tokens
Removing punctuation
Keeping alphabetic words
Stop-word removal
Preserving important negation words such as not, no, never, neither, nothing, and nowhere
Classical Machine Learning

TF-IDF representations were tested using unigram and unigram + bigram features.

Logistic Regression

Unigram Logistic Regression:

88.30% test accuracy

Unigram + Bigram Logistic Regression:

88.75% test accuracy

The 10,000-feature configuration achieved:

88.66% test accuracy

Best Classical ML Model

TF-IDF Unigram + Bigram + Logistic Regression

MetricResult
Test Accuracy88.75%
Precision88.33%
Recall89.30%
F1-Score88.81%
Parameters30,001
Training Time1.11 seconds
LSTM Experiments

The deep-learning section investigated:

Baseline LSTM
LSTM with Dropout
LSTM with Early Stopping
Increased LSTM units
Bidirectional LSTM
Reduced embedding dimension
Stacked LSTM
Best LSTM Model

The best-performing LSTM was a stacked LSTM.

MetricResult
Validation Accuracy85.56%
Test Accuracy84.77%
Precision82.96%
Recall87.51%
F1-Score85.17%
Parameters1,346,113

The best single LSTM achieved 83.68%, while the stacked LSTM achieved 84.77%, an improvement of 1.09 percentage points.

Classical ML vs LSTM
ModelAccuracyPrecisionRecallF1-Score
TF-IDF + Logistic Regression88.75%88.33%89.30%88.81%
Stacked LSTM84.77%82.96%87.51%85.17%

Classical ML accuracy advantage:

3.98 percentage points

Model complexity:

Classical ML : 30,001 parameters
LSTM         : 1,346,113 parameters

Therefore, the TF-IDF + Logistic Regression model provided better performance with substantially lower model complexity.

Misclassification Analysis

The best classical model was evaluated on 25,000 test samples.

Total test samples       : 25,000
Correct predictions      : 22,188
Misclassified samples    : 2,812
Error rate               : 11.25%

Difficult cases included:

Mixed positive and negative sentiment
Negation
Sarcasm
Irony
Context-dependent sentiment
Movie-specific language
Notebook
Task 2/
└── Sentimental_Analysis.ipynb
Major Project — Personalized Chatbot
Overview

The Major Project is a web-based Personalized Chatbot application.

The project combines a Python backend with a frontend interface to provide an interactive chatbot experience.

Project Structure
Major Project/
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── .gitignore
├── MODEL_MANIFEST.txt
├── README.md
├── app.py
├── download_models.py
└── requirements.txt
Backend

The backend is implemented using Python and Flask.

It is responsible for:

Running the chatbot application
Handling user requests
Processing user input
Generating chatbot responses
Connecting the frontend with chatbot functionality
Managing model-related functionality

Main backend file:

Major Project/app.py
Frontend

The frontend provides the chatbot interface through:

templates/index.html
static/style.css
static/script.js

It handles:

User input
Backend requests
Chatbot responses
Interactive user experience
Model Management

The project includes:

MODEL_MANIFEST.txt
download_models.py

These files document and manage the required model resources.

Requirements

Dependencies are listed in:

Major Project/requirements.txt

Install them using:

pip install -r requirements.txt
Running the Major Project
cd "Major Project"
pip install -r requirements.txt
python app.py

Then open the local address displayed by Flask.

Technologies Used
Task 1
Python
NumPy
TensorFlow
Keras
Scikit-learn
Matplotlib
CIFAR-10
Task 2
Python
NumPy
Pandas
Scikit-learn
TensorFlow
Keras
Matplotlib
IMDb Dataset
TF-IDF
Logistic Regression
Linear SVM
LSTM
Major Project
Python
Flask
HTML
CSS
JavaScript
Machine Learning
Natural Language Processing
Overall Results
ProjectBest Model / ConfigurationResult
Task 1 — CIFAR-10Final CNN75.88% Test Accuracy
Task 2 — IMDbTF-IDF + Bigram + Logistic Regression88.75% Test Accuracy
Task 2 — Deep LearningStacked LSTM84.77% Test Accuracy
Major ProjectPersonalized ChatbotWeb-based chatbot application
Key Learnings

Through these projects, the following concepts were explored:

Dataset preparation
Data preprocessing
Reproducible experimentation
CNN architecture design
Image classification
Data augmentation
Regularization
Batch Normalization
Dropout
Optimizer comparison
Text preprocessing
TF-IDF feature extraction
Logistic Regression
Linear SVM
Tokenization
Sequence padding
LSTM networks
Model comparison
Error analysis
Confusion matrices
Flask application development
Frontend/backend integration
Model management
Limitations
Task 1
CNN performance is limited by the relatively small CIFAR-10 image size.
Only a limited set of architectures and configurations were evaluated.
More advanced architectures such as ResNet were not explored.
Additional hyperparameter tuning could potentially improve performance.
Task 2
TF-IDF models do not fully capture long-range contextual relationships.
Sarcasm and context-dependent sentiment remain difficult.
The LSTM experiments used a limited training budget.
Transformer-based models were outside the scope of the current implementation.
Major Project
Chatbot capabilities depend on the available model resources and implemented response logic.
Additional model tuning and larger-scale evaluation could improve the system.
Further deployment and scalability improvements can be explored.
Future Work
Task 1
Explore deeper CNN architectures.
Test transfer learning.
Explore ResNet-style architectures.
Perform more extensive hyperparameter tuning.
Experiment with advanced augmentation techniques.
Task 2
Compare additional TF-IDF configurations.
Test Bidirectional LSTM and GRU.
Explore attention mechanisms.
Train LSTM models for more epochs.
Evaluate transformer-based models such as DistilBERT.
Major Project
Improve conversational understanding.
Expand the chatbot knowledge base.
Improve personalization.
Improve response quality.
Add additional model capabilities.
Deploy the application to a production environment.
Improve scalability and user experience.
Reproducibility

A fixed random seed of 42 was used in the machine-learning experiments wherever applicable.

Controlled data splits were maintained across the relevant experiments to make model comparisons more meaningful.

Internship

These projects were completed as part of the:

iNeuBytes AI/ML Internship

The repository demonstrates practical experience in:

Computer Vision
        +
Natural Language Processing
        +
Machine Learning
        +
Deep Learning
        +
Web Application Development
Final Summary
Task 1 — CIFAR-10 CNN

The final CNN achieved:

75.88% test accuracy

This improved the baseline from 70.47% by 5.41 percentage points.

Task 2 — IMDb Sentiment Analysis

The best classical model:

TF-IDF Unigram + Bigram + Logistic Regression

achieved:

88.75% test accuracy

The best LSTM model achieved:

84.77% test accuracy

The classical model achieved an accuracy advantage of:

3.98 percentage points

Major Project — Personalized Chatbot

A complete web-based personalized chatbot application containing:

Python backend
Flask
HTML
CSS
JavaScript
Machine-learning/NLP components
Model-management files
Author

Sunil

Internship ID: INBT021913

Organization: iNeuBytes

Repository: internsSunil_INBT021913_iNeuBytes
