from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import numpy as np
from sklearn.externals import joblib

# load embedding vectors if you want
embedded = np.loadtxt('embeddings/embed_updates.csv', delimiter=",")
targets = np.loadtxt('embeddings/names_updates.csv', delimiter=",", dtype=str)

print(embedded.shape)
print(targets.shape)

encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

# Index of training images
X_train = embedded

# Index of testing images
y_train = y

svc = LinearSVC()
svc.fit(X_train, y_train)
joblib.dump(svc, 'models/svm_model.pkl')
joblib.dump(encoder, 'models/name_encoder.pkl')