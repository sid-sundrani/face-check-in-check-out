# face-check-in-check-out
check in check out facial recognition system

SVM trained over a pre trained CNN for face recognition. Confidence metric of classification of face embedding is distance from 
SVM hyperplane ( d > -0.1). Continual learning is enabled. This implies more the system is used the better is gets. If person 
is classified with a very high confidence metric (d > 1.5) his embedding is added to the database and SVM is re trained. 

