from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
nltk.download('stopwords')


INPUT_FILE = 'training_dataset.csv'
INFORMATION_COLUMN = 'Information'
CATEGORY_COLUMN = 'Tag'
LABEL_COLUMN = 'Label'

dataset = pd.read_csv(INPUT_FILE)

print(dataset.info())
print(dataset.head())

classes = dataset['Tag'].unique()

# Label encoding and create a new column with the encoded data
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dataset[CATEGORY_COLUMN])

# create new column with the encoded values
dataset[LABEL_COLUMN] = label_encoder.transform(dataset[CATEGORY_COLUMN])

# create new column with the encoded values
dataset[LABEL_COLUMN] = label_encoder.transform(dataset[CATEGORY_COLUMN])


dataset.head()


X = dataset[INFORMATION_COLUMN]
y = dataset[LABEL_COLUMN]

# Checking the X and y values ...
print(X)
print('\n')
print(y)


all_stopwords = stopwords.words('english')
all_stopwords.append('like')
vectorizer = TfidfVectorizer(stop_words=all_stopwords, max_features=1000)
vectorized_X = vectorizer.fit_transform(X)
vectorized_X


X_train, X_test, y_train, y_test = train_test_split(
    vectorized_X, y, test_size=0.20, random_state=0)


print(X_train.shape)
print(X_train)
print()
print(y_train)


classifier = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
# classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
# Adjust font size as needed
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# plt.show()
plt.savefig("confusion.png")

print("\n")
print(accuracy_score(y_test, y_pred))


# Wrap the single value in a list
value = []
search = 'I am feeling for something cold'
value.append(search)

# Transform the single value using the same vectorizer used for training
vectorized_value = vectorizer.transform(value)

y_pred = classifier.predict(vectorized_value)
category = label_encoder.inverse_transform(y_pred)
print()
print(f"For the following search: {search}")
print(f"The category is: {category[0]}")
