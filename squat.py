import cv2
import mediapipe as mp
import numpy as np 
import time 
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
        def calculate(p1,p2,p3):
            p1 = np.array(p1) 
            p2 = np.array(p2) 
            p3 = np.array(p3) 
            
            radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle >180.0:
                angle = 360-angle
                
            return angle

        draw = mp.solutions.drawing_utils
        ps = mp.solutions.pose
        counter = 0
        stage = None
        accuracy_scores = []
        correct_movements = 0
        total_movements = 0

        cap = cv2.VideoCapture(d)
        with ps.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            prev_ft = 0 #previous frame time
            new_ft = 0  #new frame time
            
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.resize(frame,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                pic = cv2.imread(r'squat.jpeg.jpg')

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                pic.flags.writeable = False

                results = pose.process(image)
                

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                pic.flags.writeable = True

                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    hip = [landmarks[ps.PoseLandmark.LEFT_HIP.value].x,landmarks[ps.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[ps.PoseLandmark.LEFT_KNEE.value].x,landmarks[ps.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[ps.PoseLandmark.LEFT_ANKLE.value].x,landmarks[ps.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate(hip, knee, ankle)
            
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                    if angle > 160:
                        stage = 'down'
                        total_movements += 1
                    if angle  < 100 and stage == 'down':
                        stage = 'up'
                        counter += 1
                        correct_movements += 1
                        accuracy_scores.append(correct_movements / total_movements)
                        text_to_speech('count is {}'.format(counter))
                        print('count is {}'.format(counter))
                except:
                    pass

                new_ft = time.time()
                fps2 = 1/(new_ft-prev_ft)
                prev_ft = new_ft

                draw.draw_landmarks(image, results.pose_landmarks, ps.POSE_CONNECTIONS,
                                        draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                draw.draw_landmarks(pic, results.pose_landmarks, ps.POSE_CONNECTIONS,
                                        draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )

                cv2.rectangle(image, (25, 25), (300,75), (255,174,201), -1)
                cv2.rectangle(image, (25, 25), (300,75), (0,0,255))
                cv2.putText(image,'squat counter',(35,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

                cv2.rectangle(image, (350, 25), (610,75), (255,174,201), -1)
                cv2.rectangle(image, (350, 25), (610,75), (0,0,255))
                cv2.putText(image,str(counter),(380,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(image,'count',(440,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
                
                if counter > 4:
                    break
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
      plt.title('Squat Accuracy Graph')
      plt.savefig('static/squat.png')
      print('image saved')