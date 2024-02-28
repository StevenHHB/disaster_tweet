from load import load_data
from preprocess import TextPreprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


train_df, test_df, _ = load_data()
text_preprocess = TextPreprocess()
X_train = text_preprocess.fit_transform(train_df['text'])
y_train = train_df['target']
X_test = text_preprocess.transform(test_df['text'])


model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
# y_pred = model.predict(X_train)
# print(classification_report(y_train, y_pred))

model_filename = 'disaster_tweet_recognition.pkl'

joblib.dump(model, model_filename)

# After fitting the vectorizer in predict.py
joblib.dump(text_preprocess.vectorizer, 'vectorizer.pkl')
