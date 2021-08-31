import face_recognition
import cv2
import numpy as np
vid=cv2.VideoCapture(0)
my=face_recognition.load_image_file("photo.jpeg")
my_encoding=face_recognition.face_encodings(my)[0]
mama=face_recognition.load_image_file("elon.jpg")
mama_encoding=face_recognition.face_encodings(mama)[0]
known_coding=[
    mama_encoding,
    my_encoding
]
known_name=[ 
    "Elon",
    "Sankit"
]
while(True):
    ret, frame = vid.read()
    unknown_location=face_recognition.face_locations(frame)
    unknown_encoding=face_recognition.face_encodings(frame,unknown_location)
    for (top,right, bottom, left), face_encoding in zip(unknown_location, unknown_encoding):
        name="Unknown"
        match=face_recognition.compare_faces(known_coding,face_encoding)
        distance=face_recognition.face_distance(known_coding,face_encoding)
        best=np.argmin(distance)
        if match[best]:
            name=known_name[best]
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
        cv2.putText(frame,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
vid.release()
cv2.destroyAllWindows()
