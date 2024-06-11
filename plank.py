import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
import matplotlib.pyplot as plt


def speak(message):
    print(message)

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
        cap = cv2.VideoCapture(d)

        def calculate_angle(a,b,c):
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle >180.0:
                angle = 360-angle
                
            return angle 

        # Plank stage variables
        stage = None
        flag = 0
        accuracy_scores = []
        correct_movements = 0
        total_movements = 0

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    shoulder = [landmarks[11].x,landmarks[11].y]
                    elbow = [landmarks[13].x,landmarks[13].y]
                    wrist = [landmarks[15].x,landmarks[15].y]
                    hip = [landmarks[23].x,landmarks[23].y]
                    knee = [landmarks[25].x,landmarks[25].y]
                    
                    # Calculate angle
                    angle1 = calculate_angle(shoulder, elbow, wrist)
                    angle2 = calculate_angle(shoulder, hip, knee)
                    
                    # Visualize angle
                    cv2.putText(image, str(angle1), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    cv2.putText(image, str(angle2), 
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    # Plank pose logic
                    if angle1>80 and angle1<100 and angle2>170 and angle2<190:
                        stage = "perfect"
                        text_to_speech('Correct')
                        correct_movements += 1
                        total_movements += 1
                    else:
                        stage = "wrong"
                        text_to_speech('wrong')
                        total_movements += 1
                    accuracy_scores.append(correct_movements / total_movements)
                        
                    if flag == 0 and stage == "perfect" :
                        speak("perfect")
                        flag = 1
                        text_to_speech('Correct')
                        
                    elif flag == 1 and stage == "wrong" :
                        speak("wrong")
                        flag = 0
                        text_to_speech('wrong')
                except:
                    pass
                
                # Render plank pose detector
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                

                
                # Stage data
                cv2.putText(image, 'STAGE', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
                
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )               
                
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
        plt.title('Plank Accuracy Graph')
        plt.savefig('static/plank.png')
        print('image saved')