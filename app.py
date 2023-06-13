import face_recognition
import cv2 as cv
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv.VideoCapture(1)

naveed_img = face_recognition.load_image_file('pictures/naveed.jpg')
naveed_encoding = face_recognition.face_encodings(naveed_img)[0]

moiz_img = face_recognition.load_image_file('pictures/moiz.jpg')
moiz_encoding = face_recognition.face_encodings(moiz_img)[0]

qammer_img = face_recognition.load_image_file('pictures/qammer.jpg')
qammer_encoding = face_recognition.face_encodings(qammer_img)[0]

known_faces_encoding = [
    naveed_encoding,
    moiz_encoding,
    qammer_encoding
]

known_faces_names = [
    'Naveed Ahmed',
    'Abdul Moiz',
    'Qammer din'
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()

    if not ret:  # Check if the frame is empty
        break

    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_faces_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv.imshow("Attendance", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()
f.close()
