from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import CascadeClassifier
from cv2 import putText, FONT_HERSHEY_SIMPLEX
from cv2 import  rectangle as cv_rect
from dlib import shape_predictor, face_recognition_model_v1
import dlib
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# loading models required for face detection
sp = shape_predictor('models/5_landmarks.dat')
# pre trained face embedding model by dlib
face_rec_model_path = 'models/dlib_face_recognition_resnet_model_v1.dat'
facerec = face_recognition_model_v1(face_rec_model_path)
face_cascade = CascadeClassifier('models/haarcascade_frontalface_alt.xml')

# specifications for writing text on the frames
font = FONT_HERSHEY_SIMPLEX
fontScale = 0.9
fontColor = (255, 255, 255)
lineType = 2


# returns a list of bounding boxes of the detected faces
def faces_in_frame(frame):

    gray_frame = cvtColor(frame, COLOR_BGR2GRAY)
    faces_det = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

    # getting bounding box
    if len(faces_det) == 0:
        ww = 0
        faces_det_bb = [()]
        return faces_det_bb
    else:
        for (xx, yy, ww, hh) in faces_det:
            faces_det_bb = [(dlib.rectangle(left=xx + 1, top=yy + 1,
                                            right=ww + xx - 1, bottom=yy + hh))]

    return faces_det_bb


# returns identities of persons in frame, the embedding of their faces, times at which they were recognized,
# and a frame with bounded box annotations
def classify_face_video(frame, dets, face_width_thresh, svc, encoder):
    embed_database = np.loadtxt('embeddings/embed_updates.csv', delimiter=',')
    labels_update = np.loadtxt('embeddings/names_updates.csv', dtype=str, delimiter=',')

    opt_dist = - 0.2
    dist_learn = 1.0

    check = np.array(dets)
    frame_out = frame
    # if face detected

    face_descriptors = []
    identities = []
    time_seen = []

    frame_out = frame
    if check.size > 0:
        for k, d in enumerate(dets):
            # detect faces occupying a fourth of the screen
            if d.right() - d.left() > face_width_thresh/4:

                shape = sp(frame, d)
                # computing embedding for the face
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                face_descriptor = np.array(face_descriptor).reshape(-1, 1).T

                pred_dists = svc.decision_function(face_descriptor)
                example_prediction = np.argmax(pred_dists)
                max_dist = pred_dists[0][example_prediction]
                # print(max_dist)
                identity = encoder.inverse_transform(example_prediction)

                if max_dist < opt_dist:
                    break

                elif max_dist > dist_learn:
                    embeddings_update = np.vstack((embed_database, face_descriptor))
                    names_update = np.stack((labels_update, identity))
                    np.savetxt("embeddings/embed_updates.csv", np.asarray(embeddings_update),
                               delimiter=",")
                    np.savetxt("embeddings/names_updates.csv", np.asarray(names_update), fmt="%s", delimiter=",")

                frame_out = cv_rect(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)

                # adding persons name to the frame
                frame_out = putText(frame_out, str(identity),
                                    (d.left(), d.bottom() + 15),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                face_descriptors.append(face_descriptor)
                identities.append(identity)
                time_seen.append(datetime.now())

    return frame_out, face_descriptors, identities, time_seen


# trains a svm on the embeddings
def train_svm(embed_filename, targets_filename):

    # load embedding vectors if you want
    embedded = np.loadtxt(embed_filename, delimiter=",")
    targets = np.loadtxt(targets_filename, delimiter=",", dtype=str)
    # encode target variables
    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)
    # training set
    X_train = embedded
    # targets
    y_train = y
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    # saving the svm model
    joblib.dump(svc, 'models/svm_model.pkl')


        



