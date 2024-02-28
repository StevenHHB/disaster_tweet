from nltk.corpus import stopwords
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from load import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
nltk.download('vader_lexicon')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

train_df, test_df, sample_submission_df = load_data()

# check for head

print(train_df.head())

# check for isnull values
print(train_df.isnull().sum())
# 61 for keyword, 2533 for location

# check for length distributions vs target
train_df['length'] = train_df.length = train_df.text.apply(len)
# print(train_df.head())
plt.figure(figsize=(12, 6))
sns.boxplot(x='target', y='length', data=train_df)
plt.title('Box Plot for tweet_length distribution vs target')
plt.xlabel('Target')
plt.ylabel('tweet_length')
# plt.show()
# shows that longer tweets are generally 1 target

# check for sentiment distribution vs target
# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# Create a new column to hold the sentiment scores
train_df['sentiment'] = train_df.text.apply(
    lambda x: sia.polarity_scores(x)['compound'])
# print(train_df.head())
plt.figure(figsize=(12, 6))
sns.boxplot(x='target', y='sentiment', data=train_df)
plt.title('Box Plot for sentiment distribution vs target')
plt.xlabel('Target')
plt.ylabel('Sentiment')
# plt.show()
# negative values are more likely to be 1 target


# word frequency analyze
def most_common_words(text_series, title, num_words=20):
    all_words = " ".join(text_series).lower()
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = all_words.translate(translator)
    all_words = text_no_punctuation.split()
    all_words = [word for word in all_words if word.isalpha()
                 and word not in stop_words]
    word_freq = Counter(all_words)
    most_common = word_freq.most_common(num_words)

    words, frequencies = zip(*most_common)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(frequencies), y=list(words))
    plt.title(title)
    plt.show()

# positive words appear more in non disaster tweets
# most_common_words(train_df[train_df['target'] == 1]['text'],
    # 'Most common words in disaster tweets')

# most_common_words(train_df[train_df['target'] == 0]['text'],
    # 'Most common words in non-disaster tweets')
