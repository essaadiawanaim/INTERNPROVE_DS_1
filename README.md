# INTERNPROVE_DS_1
##### Analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands.

This project aims to perform Entity Sentiment Analysis on Twitter data. The objective is to analyze sentiments (positive, negative, neutral,Irrelevant) associated with specific entities mentioned in tweets. By leveraging various machine learning and deep learning techniques, this notebook preprocesses the text data, applies feature extraction, and trains models to classify the sentiments associated with the tweets.

# Project Structure
### 1. Data Preprocessing:

* Load and inspect the Twitter dataset.
* Vectorize the text using TF-IDF (Term Frequency-Inverse Document Frequency) to transform text data into numerical format suitable for model input.
### 2. Model Building:

Different machine learning models are used to classify sentiments.
### 3. Model Evaluation:
Models are evaluated using metrics like accuracy, and classification report.
### 4. Visualization:

* WordCloud is generated to visualize the most frequent words.
* Clustering of entities is done using K-Means, and results are visualized with PCA (Principal Component Analysis).
## Algorithms Used
### Machine Learning Models:
* **Naive Bayes (MultinomialNB)**: A probabilistic classifier suitable for text classification.
* **Logistic Regression**: A linear model for binary and multi-class classification.
* **Random Forest**: An ensemble model that builds multiple decision trees and merges them together to get a more accurate prediction.
* **Decision Tree**: A tree-like structure that splits the data based on features to make predictions.
### Deep Learning:
* **Keras Sequential Model**: A neural network model with multiple layers, including dense layers and dropout for regularization. This is used to improve the model's performance in classifying sentiments.
### Clustering:
* **K-Means Clustering:** Unsupervised algorithm used to group the data into clusters based on text features. It is followed by PCA to reduce dimensionality for visualization.
### Text Preprocessing:
* **TF-IDF Vectorizer**: Converts text into a matrix of token counts weighted by term frequency-inverse document frequency.
* **PCA (Principal Component Analysis)**: A dimensionality reduction technique used to visualize high-dimensional data.
## Libraries Used
* _Pandas_: For data manipulation and analysis.
* _NumPy_: For numerical computing.
* _Matplotlib & Seaborn_: For data visualization.
* _WordCloud_: For creating word clouds of text data.
* _sklearn (scikit-learn)_:
   *TF-IDF Vectorizer for feature extraction.
   *Machine learning models (Naive Bayes, Logistic Regression, Random 
     Forest, Decision Tree).
   *KMeans for clustering and PCA for dimensionality reduction.
* _Keras_: For building neural networks.
* _NLTK_: For text processing and natural language processing (NLP) tasks.
## Data
The [dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) consists of a large number of tweets, with associated entities and their corresponding sentiment labels. The data is split into training and validation sets for model building and evaluation.

* **Training Data**: Contains tweet text and their respective sentiment labels (positive, negative, neutral, irrelevant).
* **Validation Data**: Used to validate and test the model’s performance.
## Usage
 1.Clone the repository:
 
`git clone https://github.com/your-username/entity-sentiment-analysis.git`

2. Install the required dependencies:
   
`pip install -r requirements.txt`

4. Run the Jupyter notebook to process the data and build the models:
   
`jupyter notebook entity_sentiment_analysis.ipynb`


## Conclusion: 
The model that delivers the best performance overall is Artificial Neural Network (ANN) with 97% accuracy
