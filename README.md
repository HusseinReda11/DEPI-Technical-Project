# Twitter Sentiment Analysis Project

A comprehensive sentiment analysis system for tweets, featuring data exploration, model training, and deployment as a web application.

## Project Overview

This project implements an end-to-end sentiment analysis pipeline for tweets. It consists of three main phases:

1. **Data Exploration & Analysis** - Exploring and visualizing tweet data to gain insights  
2. **Model Training** - Fine-tuning a DistilBERT model for sentiment classification  
3. **Deployment** - Deploying the model as a web application with an API

## Repository Structure

```tree
twitter-sentiment-analysis/
│
├── Phase-0_Data_Exploration/            # Data exploration phase
│   ├── main.py                          # Entry point for data exploration
│   ├── reports/                         # Generated reports and visualizations
│   └── src/                             # Source code for data exploration
│       ├── config/                      # Configuration files
│       ├── data/                        # Data loading and processing
│       ├── exploration/                 # Analysis modules
│       └── utils/                       # Utility functions
│
├── Phase-1_Model_Training/              # Model training phase
│   ├── main.py                          # Entry point for model training
│   ├── predict.py                       # Script for making predictions
│   ├── reports/                         # Training reports and visualizations
│   ├── models/                          # Saved model checkpoints
│   ├── logs/                            # Training logs
│   └── src/                             # Source code for model training
│       ├── config/                      # Configuration files
│       ├── data/                        # Data preparation
│       ├── model/                       # Model architecture
│       ├── training/                    # Training utilities
│       ├── visualization/               # Visualization utilities
│       └── utils/                       # Utility functions
│
├── website/                             # Deployment phase
│   ├── server.py                        # Flask application
│   ├── config.py                        # Application configuration
│   ├── wsgi.py                          # WSGI entry point
│   ├── Dockerfile                       # Docker configuration
│   ├── requirements.txt                 # Dependencies
│   ├── api/                             # API endpoints
│   ├── models/                          # Model loading and inference
│   ├── static/                          # Static assets (CSS, JS)
│   └── templates/                       # HTML templates
│
├── Data/                                # Shared data directory
│   ├── raw_data.csv                     # Original dataset
│   ├── processed_data.csv               # Processed dataset
│   └── token_frequencies.csv            # Word frequency data
│
└── README.md                            # Project documentation
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Docker (for deployment)

### Setting Up the Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/MrThetaIII/Twitter-Sentiment-Analysis
   cd Twitter-Sentiment-Analysis
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install common dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Phase 0: Data Exploration

The data exploration phase analyzes tweet data to understand patterns, distributions, and language features.

### Features

- Comprehensive text analysis (word frequencies, n-grams, hashtags)
- Sentiment distribution analysis
- Time-based analysis of tweet patterns
- User behavior analysis
- Visualization of insights

### Running the Data Exploration

```bash
cd Phase-0_Data_Exploration
python main.py
```

Command-line options:

- `--no-figures`: Don't save figures
- `--no-data`: Don't save processed data
- `--keep-stopwords`: Keep stop words in text analysis (default: remove them)

### Exploration Outputs

- Visualizations of word frequencies, hashtags, and sentiment patterns
- Processed datasets
- Text analysis reports

## Phase 1: Model Training

The model training phase fine-tunes a DistilBERT model for sentiment classification using the Twitter dataset.

### Model Training Features

- Fine-tuning of DistilBERT transformer model
- Partial fine-tuning by default (only last layer)
- MLflow experiment tracking
- Visualization of training metrics
- Confusion matrix and performance analysis
- Resource usage monitoring

### Running the Model Training

```bash
cd Phase-1_Model_Training
python main.py
```

Command-line options:

- `--full-finetune`: Fine-tune the entire model (default: partial fine-tuning)
- `--batch-size`: Set batch size (default: 16)
- `--epochs`: Number of training epochs (default: 3)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--max-len`: Maximum sequence length (default: 48)
- `--seed`: Random seed (default: 42)
- `--data-portion`: Data portion denominator (1/portion) (default: 1)

### Making Predictions with the Trained Model

```bash
python predict.py --text "I love this app, it's amazing!"
```

Command-line options:

- `--model-path`: Path to model checkpoint
- `--text`: Text to predict sentiment for
- `--input-file`: CSV file with texts to predict
- `--output-file`: CSV file to save predictions to
- `--with-probs`: Include prediction probabilities

### Model Training Outputs

- Trained model checkpoints
- Training metrics
- Confusion matrix
- Learning curves
- MLflow experiment tracking data

## Phase 2: Deployment

The deployment phase packages the trained model into a web application with an API for sentiment prediction.

### Website Features

- User-friendly web interface
- RESTful API for batch processing
- Docker containerization
- Error handling and logging
- Responsive design

### Running the Web Application Locally

First insure you put the trained model in `Phase-2_Deployment\models\checkpoints\destillbert.pt`
Download a pre-trained model from [here]<https://drive.google.com/drive/folders/1_U1NWvhLK8ID8SXCx3LeGTfxFTMVd56s?usp=sharing>

```bash
cd Phase-2_Deployment
python server.py
```

### Deploying with Docker

```bash
cd website
docker build -t sentiment-analysis-app .
docker run -p 8080:8080 sentiment-analysis-app
```

### API Documentation

#### Predict Sentiment

**Endpoint:** `/api/predict`

**Method:** POST

**Request Body:**

```json
{
  "tweets": [
    {"text": "I love this product!"},
    {"text": "This is terrible service."}
  ]
}
```

**Response:**

```json
[
  {
    "text": "I love this product!",
    "sentiment": "positive",
    "confidence": 0.98
  },
  {
    "text": "This is terrible service.",
    "sentiment": "negative",
    "confidence": 0.95
  }
]
```

#### Health Check

**Endpoint:** `/api/healthz`

**Method:** GET

**Response:**

```json
{
  "status": "ok"
}
```

## Model Performance

Our fine-tuned DistilBERT model achieves the following performance metrics on the test set:

- **Accuracy**: 86.90%

## Data

This project uses the Twitter Sentiment Analysis Dataset available on Kaggle. The dataset contains tweets labeled with sentiment (positive or negative).

## Acknowledgments

- HuggingFace Transformers library for providing DistilBERT implementation
- Kaggle for the Twitter Sentiment Analysis dataset
- NLTK team for text processing utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
