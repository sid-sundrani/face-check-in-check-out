import cv2
import numpy as np
from timeit import default_timer as timer
from sklearn.externals import joblib
import face_recognition
import time
from datetime import datetime, timedelta
import json
from dateutil.parser import *

# importing scikit-learn svm
svc = joblib.load('models/svm_model.pkl')
encoder = joblib.load('models/name_encoder.pkl')

# Setting up open CV video object
video = cv2.VideoCapture('videos/test7.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
# the size of the frame will be downscaled to a fourth of its size
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) / 4)
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 4)
# video encoding
video_encoding = cv2.VideoWriter_fourcc(*'DIVX')
# object to write processed frames to video file
video_out = cv2.VideoWriter('video_output/output7.mp4', video_encoding, 15.0, (int(w), int(h)), isColor=True)
# total number of frames in video
total_frames = video.get(7)
# will sample the video at half the current frame rate
frame_intervals = np.arange(1, total_frames, 4)
print('new frame rate', fps / 4)

np.warnings.filterwarnings('ignore')
# saving execution times
face_detection_times = []
processing_times = []
all_check_in_times = []
all_recognized_faces = []

check_in_details = {}

start1 = time.time()

# sequentially selecting every n frames from video
for frame_no in frame_intervals:
    # reading frame #frame_no
    video.set(1, frame_no)
    status, frame = video.read()
    # downscaling frame
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    # obtaining bounding box of faces in the frame
    start_det = timer()
    faces_detected = face_recognition.faces_in_frame(frame)
    end_det = timer()
    face_detection_times.append(end_det-start_det)
    # minimum width of the faces we want to recognize/classify. eg: only classify close-by faces
    face_width = w/5
    frame_out = frame
    # time persons are seen/classified
    times_seen = []
    identities = []
    # if the frame contains faces classify the faces in the frame
    if np.array(faces_detected).size > 0:
        frame_out, face_descriptors, identities, times_seen = face_recognition.classify_face_video(frame, faces_detected,
                                                                                                   face_width, svc,
                                                                                                   encoder)
    for name, last_seen in zip(identities, times_seen):
        if name in check_in_details:
            if datetime.now() - check_in_details[name][-1] > timedelta(seconds=2):
                check_in_details[name].append(last_seen)
        else:
            check_in_details[name] = []
            check_in_details[name].append(last_seen)
    # writing frame
    video_out.write(frame_out)

end1 = time.time()

print('total time taken ', end1 - start1)
print('face detection time', np.mean(np.array(face_detection_times)))
# destroying running processes
video.release()
video_out.release()
cv2.destroyAllWindows()

