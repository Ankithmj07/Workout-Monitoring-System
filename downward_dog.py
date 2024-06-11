import cv2
import mediapipe as mp
import numpy as np
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
        cap = cv2.VideoCapture(d)#cap = cv2.VideoCapture(0) for live feed

        # Curl counter variables
        stage = None
        accuracy_scores = []
        correct_movements = 0
        total_movements = 0
        i = 0

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
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    
                    def calculate_angle(a,b,c):
                        a = np.array(a) # First
                        b = np.array(b) # Mid
                        c = np.array(c) # End
                        
                        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                        angle = np.abs(radians*180.0/np.pi)
            
                        if angle >180.0:
                            angle = 360-angle
                
                        return angle 
                    
                    # Calculate angle
                    angle1 = calculate_angle(shoulder, elbow, wrist)
                    angle2 = calculate_angle(knee, hip, shoulder)
                    angle3 = calculate_angle(ankle, knee, hip)
                    # Visualize angle
                    cv2.putText(image, str(angle1), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    cv2.putText(image, str(angle2), 
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    cv2.putText(image, str(angle3), 
                                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    # pose judging logic
                    if angle1 > 170 and angle2 > 52 and angle2 < 67 and angle3 > 170:
                        stage = "Correct"
                        text_to_speech('Correct')
                        correct_movements += 1
                        total_movements += 1
                    else:
                        stage="Incorrect"
                        # text_to_speech('Incorrect')
                        total_movements += 1
                    
                    accuracy_scores.append(correct_movements / total_movements)

                    if i == 0: flag = 0
                    if stage == "Correct" and flag == 0:
                        #speak(stage)
                        flag = 1
                        #text_to_speech('Correct')
                        
                    else:
                        if stage == "Incorrect" and flag == 1:                      
                            #speak(stage)
                            flag = 0
                            # text_to_speech('Incorrect')
                    i += 1

                except:
                    pass
                
                # Render pose status
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                
                # Rep data
                cv2.putText(image, 'Pose', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, 'Stage', (65,12), 
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
        plt.title('Downward Dog Accuracy Graph')
        plt.savefig('static/downward_dog.png')
        print('image saved')