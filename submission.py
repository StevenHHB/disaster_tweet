import pandas as pd
import numpy as np
from load import load_data
from preprocess import TextPreprocess
import joblib

model = joblib.load('disaster_tweet_recognition.pkl')
train_df, test_df, _ = load_data()
text_preprocess = TextPreprocess()
# In submission.py, before transforming test data
text_preprocess.vectorizer = joblib.load('vectorizer.pkl')
X_test = text_preprocess.transform(test_df['text'])
predictions = model.predict(X_test)
submission_df = pd.DataFrame({'id': test_df['id'], 'target': predictions})

file_path = 'submission.csv'
submission_df.to_csv(file_path, index=False)
