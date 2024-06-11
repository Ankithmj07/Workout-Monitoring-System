import cv2
import numpy as np
import mediapipe as mp

import os

import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
import matplotlib.pyplot as plt

def text_to_speech(text1):
    print(text1)
    myobj = gTTS(text=text1, lang='en-us', tld='com', slow=False)
    myobj.save("voice.mp3")
    print('\n------------Playing--------------\n')
    song = MP3("voice.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    #pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

def gen_frames(d):
  try:
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose

    counter = 0

    stage = None
    accuracy_scores = []
    correct_movements = 0
    total_movements = 0

    def findPosition(image, draw=True):

      lmList = []

      if results.pose_landmarks:

          mp_drawing.draw_landmarks(

            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

          for id, lm in enumerate(results.pose_landmarks.landmark):

              h, w, c = image.shape

              cx, cy = int(lm.x * w), int(lm.y * h)

              lmList.append([id, cx, cy])

              #cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

      return lmList

    cap = cv2.VideoCapture(d)

    with mp_pose.Pose(

        min_detection_confidence=0.7,

        min_tracking_confidence=0.7) as pose:

      while cap.isOpened():

        success, image = cap.read()

        image = cv2.resize(image, (640,480))

        if not success:

          print("Ignoring empty camera frame.")

          # If loading a video, use 'break' instead of 'continue'.

          continue

        # Flip the image horizontally for a later selfie-view display, and convert

        # the BGR image to RGB.

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to

        # pass by reference.

        results = pose.process(image)

        # Draw the pose annotation on the image.

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lmList = findPosition(image, draw=True)

        if len(lmList) != 0:

          cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)

          cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)

          cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)

          cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)

          if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):

            cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)

            cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)

            stage = "down"
            total_movements += 1

          if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "down":

            stage = "up"

            counter += 1

            counter2 = str(int(counter))

            print(counter)
            correct_movements += 1
            accuracy_scores.append(correct_movements / total_movements)
            text_to_speech('counter is {}'.format(counter))
            # os.system("echo '" + counter2 + "' | festival --tts")

        text = "{}:{}".format("Push Ups and Downs", counter)
        cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,

                    1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()

  except:
      cap.release()
      cv2.destroyAllWindows()
      plt.plot(np.arange(len(accuracy_scores)), accuracy_scores)
      plt.xlabel('Time')
      plt.ylabel('Accuracy')
      plt.title('Pushup Accuracy Graph')
      plt.savefig('static/pushup.png')
      print('image saved')