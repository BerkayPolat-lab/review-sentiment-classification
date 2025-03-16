# Book Review Sentiment Classification

## Overview

This project applies machine learning techniques to classify the sentiment of book reviews. It leverages a dataset of 10,000 book reviews and builds a sentiment analysis model capable of predicting the sentiment (positive or negative) based on review text.

## Dataset

The dataset used is `Books_small_10000.json`, which contains 10,000 JSON-formatted book reviews. Each entry includes:

- **reviewerID**: Unique ID of the reviewer
- **asin**: Amazon Standard Identification Number of the book
- **reviewerName**: Name of the reviewer
- **helpful**: Helpfulness votes as a list `[helpful_votes, total_votes]`
- **reviewText**: The actual text of the review
- **overall**: Rating given by the reviewer (1 to 5 stars)
- **summary**: Short summary of the review
- **unixReviewTime**: Unix timestamp of the review
- **reviewTime**: Human-readable review date

**Sentiment Labeling:**  
Ratings ≥ 4 are labeled as **Positive**, and ratings ≤ 3 as **Negative**.

## Project Structure

```
├── classification.ipynb      # Main Jupyter notebook with the model implementation
├── Books_small_10000.json    # Dataset file
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Python dependencies
```

## Approach

1. **Data Preprocessing:**
   - Load and parse JSON data.
   - Extract relevant fields (`reviewText`, `overall` rating).
   - Label reviews as Positive/Negative.
   - Clean and normalize review text (lowercasing, removing punctuation, etc.).
   - Tokenization and stopwords removal.

2. **Feature Extraction:**
   - Use **TF-IDF Vectorization** to convert text to numerical feature vectors.

3. **Model Selection:**
   - Train multiple classifiers such as:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest Classifier
     - Naive Bayes
   - Performed hyperparameter tuning where applicable.

4. **Model Evaluation:**
   - Split dataset into training and testing sets (80-20 split).
   - Use metrics like **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.
   - Compare performance across models.

5. **Results & Insights:**
   - Best-performing model details.
   - Visualizations.
   - Key observations.

## Dependencies

All dependencies are listed in `requirements.txt`. Key libraries include:

- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- json

Install via:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone or download the repository.
2. Install dependencies.
3. Open `classification.ipynb` using Jupyter Notebook.
4. Run all cells step by step.
5. The notebook will output:
   - Preprocessing steps
   - Model training and evaluation results
   - Visualizations

## Future Improvements

- Implement deep learning models (e.g., LSTM, BERT) for improved performance.
- Incorporate more advanced text cleaning (lemmatization, named entity recognition).
- Experiment with ensemble methods.
- Deploy the model as an API for real-time sentiment analysis.

## License

This project is released under the MIT License.