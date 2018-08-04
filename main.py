import cv2
import numpy as np
from timeit import default_timer as timer
from sklearn.externals import joblib
import face_recognition
import time
from datetime import datetime, timedelta
import json
from dateutil.parser import *

# importing classification models
svc = joblib.load('models/svm_model.pkl')
encoder = joblib.load('models/name_encoder.pkl')

# Setting up open CV video object
video = cv2.VideoCapture('videos/test6.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
# the size of the frame will be downscaled to a fourth of its size 
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) / 4)
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 4)
# video encoding
video_encoding = cv2.VideoWriter_fourcc(*'DIVX')
# object to write processed frames to video file
video_out = cv2.VideoWriter('video_output/output6.mp4', video_encoding, 15.0, (int(w), int(h)), isColor=True)
# total number of frames in video
total_frames = video.get(7)
# will sample the video at half the current frame rate
frame_intervals = np.arange(1, total_frames, 4)
print('new frame rate', fps / 4)

np.warnings.filterwarnings('ignore')
# saving execution times
face_detection_times = []
processing_times = []
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
    face_width = w/4
    frame_out = frame
    # time persons are seen/classified
    time_seen = []
    # if the frame contains faces classify the faces in the frame
    if np.array(faces_detected).size > 0:
        frame_out, face_descriptors, identities, time_seen = face_recognition.classify_face_video(frame, faces_detected,
                                                                                                  face_width, svc,encoder)
    check_in_out = {'check-in': [], 'check-out': []}

    if len(time_seen) > 0:
        for i in range(len(time_seen)):
            if identities[i] in check_in_details:

                # parse converts string to datetime
                if datetime.now() - check_in_details[identities[i]]['check-in'][-1] > timedelta(seconds=10):
                    print(1)
                    print(check_in_details)
                    time_seen[i]
                    check_in_details[identities[i]] = check_in_out['check-in'].append(time_seen[i])
            else:
                print(2)
                check_in_details[identities[i]] = check_in_out
                check_in_details[identities[i]]['check-in'].append(time_seen[i])

        json_file = json.dumps(check_in_details, indent=4, sort_keys=True, default=str)
        f = open("check_in_times.json", "w")
        f.write(json_file)
        f.close()

    video_out.write(frame_out)

end1 = time.time()

print('total time taken ', end1 - start1)
print('face detection time', np.mean(np.array(face_detection_times)))

video.release()
video_out.release()
cv2.destroyAllWindows()

