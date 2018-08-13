# face-check-in-check-out
check in check out facial recognition system

SVM trained on top of a pre trained CNN for face recognition. Confidence metric of classification of face embedding (output of CNN) is distance from 
SVM hyperplane ( d > -0.1). Continual learning is enabled. This implies more the system is used the better is gets. If person 
is classified with a very high confidence metric (d > 1.5) his embedding is added to the database and SVM is re trained. 

main.py classfies and records time a person was seen inside, from camera #1. main_2.py classfies and records time a person was seen outside from camera #2. The video feeds fed in this project were just two pre recorded but can easiliy be extended to live capture. Additionally, this process is fast enough to run live. 

