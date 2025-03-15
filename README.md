# Aspect-Based Sentiment Analysis for Financial News

## Introduction
This project focuses on aspect-based sentiment analysis (ABSA) for financial news using the SentFin dataset. The objective is to classify sentiment at the aspect level, which provides a more granular understanding of financial news sentiments. Various machine learning and deep learning techniques, including LSTMs, FinBERT, and traditional models like Logistic Regression and SVM, have been used to evaluate the effectiveness of different approaches.

## Methodology
![image](https://github.com/user-attachments/assets/ebdc86a8-8a67-41e5-9f82-a5e59b74015a)

The methodology followed in this project consists of the following steps:

1. **Dataset Collection and Preprocessing:**  
   - The SentFin dataset was used for training and evaluation.
   - Data cleaning steps such as removing special characters, stopwords, and tokenization were applied.
   - Aspect-specific sentiment labels were extracted.

2. **Feature Engineering:**  
   - Word embeddings such as Word2Vec and TF-IDF were used.
   - Text vectorization techniques were applied for machine learning models.

3. **Model Training and Evaluation:**  
   - **Traditional Machine Learning Models:** Logistic Regression, SVM, Random Forest, and XGBoost were trained and evaluated using precision, recall, and F1-score.
   - **Deep Learning Models:** LSTM networks were trained with word embeddings.
   - **FinBERT:** A transformer-based approach (FinBERT) was fine-tuned for financial sentiment classification.
   
4. **Performance Evaluation:**  
   - Accuracy, precision, recall, and F1-score were computed for each model.
   - Comparison of model performance was conducted to determine the best-performing approach.

## Results
- **LSTM Results:** The loss and accuracy curves indicate stable training with minor overfitting.
- **FinBERT Results:** FinBERT performed reasonably well, achieving an accuracy of 62% on the SentFin dataset.
- **Machine Learning Models:**
  - Logistic Regression achieved an accuracy of 76.01%.
  - SVM obtained an accuracy of 75.17%.
  - Random Forest had an accuracy of 73.36%.
  - XGBoost performed slightly better at 73.50%.

## Conclusion
The results suggest that deep learning models such as LSTMs and transformer-based models (FinBERT) provide competitive performance for aspect-based sentiment analysis in financial news. Traditional machine learning models performed well, with Logistic Regression achieving the highest accuracy among them. Future work can focus on improving deep learning models through hyperparameter tuning, data augmentation, and the inclusion of additional financial news sources to enhance model generalization.

## Installation and Usage
To use this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements.txt

# Run the model training script
python train.py
```

## Acknowledgments
This project utilizes the SentFin dataset, as introduced in the following research paper:

Citation:

SentFin: Entity-Specific Sentiment Analysis of Financial News
Shantanu Agarwal, Nishant Nikhil, Aritra Ghosh, Pawan Goyal
arXiv preprint arXiv:2305.12257 (2023)
[Paper Link](https://arxiv.org/abs/2305.12257)

