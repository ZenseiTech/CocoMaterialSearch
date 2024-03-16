import pickle


classifier = pickle.load(open('classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Checking against single search value ...
value = []
search = 'I wish I was on a beach now'
value.append(search)

# Transform the single value using the same vectorizer used for training
vectorized_value = vectorizer.transform(value)

y_pred = classifier.predict(vectorized_value)
category = label_encoder.inverse_transform(y_pred)

print()
print(f"For the following search: {search}")
print(f"The category is: {category[0]}")
