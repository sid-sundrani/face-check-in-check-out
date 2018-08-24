# face-check-in-check-out
check in check out facial recognition system

SVM trained on top of a pre trained CNN for face recognition. Confidence metric of classification of face embedding (output of CNN) is distance from 
SVM hyperplane ( d > -0.1). Continual learning is enabled. This implies that more the system is used the better is gets. If a person 
is classified with a very high confidence metric (d > 0.7) his embedding is added to the database and SVM is re trained. 
Plot below shows the prediction accuracy vs distance threshold d (from hyperplane)and ratio of the number of photos classified vs threshold. As d increases, the ratio of the number of photos classified decreases and accuracy increases.

![alt text](https://github.com/sid-sundrani/face-check-in-check-out/blob/master/models/Snip20180805_1.png= =250x250)


files: 
main.py classfies and records time a person was seen inside, from camera #1. main_2.py classfies and records time a person was seen outside from camera #2. The video feeds fed in this project were just two pre recorded but can easiliy be extended to live capture. Additionally, this process is fast enough to run live at 480p. 


