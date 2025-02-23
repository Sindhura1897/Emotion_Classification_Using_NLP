# Emotion Classification Using NLP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NLTK](https://img.shields.io/badge/NLTK-✔-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-✔-orange)

## Overview
This project builds a text classification model to categorize textual data into predefined classes using NLP and deep learning techniques.

## Tools Used
- **Python**
- **NLTK**
- **Scikit-Learn**
- **Pandas**
- **Gensim**
- **WordCloud**

## Machine Learning Models Used
- **LDA**
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

## Approach
1. **Data Preprocessing & Cleaning:**  
   - Removed special characters, punctuation, and extra spaces.  
   - Converted text to lowercase and removed stopwords.  
   - Performed tokenization, and stemming for text standardization.

2. **Feature Engineering & Representation:**  
   - Implemented **TF-IDF** and **CountVectorizer** to convert text into numerical representations.  
   - Used **Word2Vec** embeddings for deep learning models.
   - Generated **Word Cloud** for visualizing text data.
   - Applied **PCA** for dimensionality reduction and visualization.  

3. **Model Selection & Training:**
   - Applied Latent Dirichlet Allocation (LDA) to extract meaningful topics from text data.
   - Trained traditional ML models (Logistic Regression, Decision Tree, Random Forest).   
   - Used word embeddings to improve accuracy.

5. **Model Evaluation & Optimization:**
   - Evaluated performance using accuracy, precision-recall.  
   - Fine-tuned hyperparameters for optimal results.  

## Results
- Achieved **85% accuracy** in classifying text into predefined categories.  
