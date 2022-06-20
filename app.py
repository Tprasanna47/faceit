from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy
import numpy as np
import datetime
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore
# import firebase_admin
# from firebase_admin import credentials


# cred = credentials.Certificate("faceit-2f784-firebase-adminsdk-5z8za-fe87f32d59.json")
# firebase_admin.initialize_app(cred)
# db=firestore.client()
path='training'
images=[]
classnames=[]


mylist=os.listdir(path)
todays = datetime.datetime.today().strftime('%d-%m-%y')
if not os.path.exists(f'{todays}'):
    open(f'{todays}.csv', 'w').close()
# print(mylist) will give output ['elon1.jpg', 'elon2.jpg', 'virat.jpg']

for cls in mylist:
    curimg=cv2.imread(f'{path}/{cls}')
    # it is a format of writin path is training adn cls in image name
    images.append(curimg)
    classnames.append(os.path.splitext(cls)[0])
    # upper i dont want to print name.jpg so did splitiing
# print(classnames)


# print(images)
# what we did here was we took all the images from
def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


encodelistknown=findencodings(images)
print("Encoding Complete")


def markattendance(name):
    todays = datetime.datetime.today().strftime('%d-%m-%y')
    # if not os.path.exists(f'{todays}'):
    #  open(f'{todays}.csv', 'w').close()

    with open(f'{todays}.csv','r+') as f:
    # with open('attendance.csv','r+') as f:
    # here we open attendance file and do read and write as r+
      mydatalist=f.readlines()
      namelist=[]
      for line in mydatalist:
        entry=line.split(',')
        namelist.append(entry[0])
      if name not in namelist:
         now = datetime.datetime.now()
         dtstring = now.strftime('%H:%M:%S')
         f.writelines(f'\n{name},{dtstring}')
#          db.collection(f'{todays}').add({f'Name': f'{name}', 'Time': f'{dtstring}'})


app = Flask(__name__)
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:

        ## read the camera frame
        success, img = camera.read()
        if not success:
            break
        else:

            imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faceCurframe = face_recognition.face_locations(imgs)
            encodeCurframe = face_recognition.face_encodings(imgs, faceCurframe)
            for encodeface, faceloc in zip(encodeCurframe, faceCurframe):
                matches = face_recognition.compare_faces(encodelistknown, encodeface)
                facedistance = face_recognition.face_distance(encodelistknown, encodeface)
                # print(facedistance)
                matchindex = np.argmin(facedistance)

                if matches[matchindex]:
                    name = classnames[matchindex].upper()
                    print(name)
                    y1, x2, y2, x1 = faceloc
                    # y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 255), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
                    # all the codes written are for format of how the text are displayed in the webcam
                    markattendance(name)

                ret, buffer = cv2.imencode('.jpg', img)
                img = buffer.tobytes()
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')










@app.route('/')
def index():
    return render_template('hello.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
