import time
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import os

# 0:'nomarl', 1:'near', 2:'danger'

# 특징 저장

actions = np.array(['safe', 'safe_phone', 'close', 'danger_hit', 'danger_neck'])
video_loc = np.array(['data_mp4\Safe_sit.mp4', 'data_mp4\Safe_phone.mp4', 'data_mp4\warn_close.mp4', 'data_mp4\d1.mp4', 'data_mp4\d2.mp4'])
secs_for_action = 240 # 시간
seq_length = 30 # 프레임 수
created_time = int(time.time())

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def mediapipe_detection(image, model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = model.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  return image, results

def draw_landmarks(image, results):
  mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  return 


# 33개의 랜드마크중 23부터 하체이기 때문에 제외
def extract_keypoint_only_joint(results):
    joint = np.zeros((23,4))
    for j, lm in enumerate(results.pose_landmarks.landmark):
        if j >22 : break
        joint[j] = [lm.x, lm.y, lm. z, lm.visibility]
    
    return joint.flatten()
    
def extract_keypoint_with_angle(results,idx):
    joint = np.zeros((23,4))
    for j, lm in enumerate(results.pose_landmarks.landmark):
        if j >22 : break
        joint[j] = [lm.x, lm.y, lm. z, lm.visibility]

    # 두 점 사이의 선 구하기
    v1 = joint[[0,11,13,15,0,12,14,16],:3] 
    v2 = joint[[11,13,15,19,12,14,16,20],:3]
    v = v2 - v1 
    # 정규화 
    v = v / np.linalg.norm(v, axis=1)[:,np.newaxis] #TODO: 코드 이해 필요
    # 닷 프로덕트의 아크코사인으로 각도를 구함
    angle = np.arccos(np.einsum('nt,nt->n', v[[0,1,2,4,5,6],:], v[[1,2,3,5,6,7],:])) 
    angle = np.degrees(angle)
    
    angle_label = np.array([angle], dtype=np.float32)
    angle_label = np.append(angle_label, idx) # 정답 라벨 포함

    return np.concatenate([joint.flatten(), angle_label.flatten()]) 
    
def extract_keypoint_with_angle_no_face(results,idx): # 13*4+6+1 = 59
    joint = np.zeros((13,4))
    for j, lm in enumerate(results.pose_landmarks.landmark):
        if j > 22 : break
        if j==0 : joint[j] = [lm.x, lm.y, lm. z, lm.visibility]
        if j>10 : joint[j-10] = [lm.x, lm.y, lm. z, lm.visibility] # 얼굴 랜드마크의 빈자리를 땡겨서 압축

    # 두 점 사이의 선 구하기
    v1 = joint[[0,1,3,5,0,2,4,6],:3] 
    v2 = joint[[1,3,5,9,2,4,6,10],:3]
    v = v2 - v1 
    # 정규화 
    v = v / np.linalg.norm(v, axis=1)[:,np.newaxis] #TODO: 코드 이해 필요
    # 닷 프로덕트의 아크코사인으로 각도를 구함
    angle = np.arccos(np.einsum('nt,nt->n', v[[0,1,2,4,5,6],:], v[[1,2,3,5,6,7],:])) 
    angle = np.degrees(angle)
    
    angle_label = np.array([angle], dtype=np.float32)
    angle_label = np.append(angle_label, idx) # 정답 라벨 포함

    return np.concatenate([joint.flatten(), angle_label.flatten()]) 



# 실시간 캠
# cap = cv2.VideoCapture(0) 
# 동영상
# video_file = "data_mp4\d1.mp4"
# cap = cv2.VideoCapture(video_file) 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for idx, action in enumerate(actions):
        video_file = video_loc[idx]
        cap = cv2.VideoCapture(video_file) 

        while cap.isOpened():
        
            data =  []
            
            success, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.resize(image, (720, 400), interpolation=cv2.INTER_AREA)

            cv2.putText(image, "starting collection", (120, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255, 0), 2, cv2.LINE_AA)
            cv2.imshow('img',image)
            cv2.waitKey(1000)

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                success, image = cap.read()
                image = cv2.resize(image, (720, 400), interpolation=cv2.INTER_AREA)
                image = cv2.flip(image,1)

                image, results = mediapipe_detection(image, pose)
                
                if results.pose_landmarks is not None: 
                    data.append(extract_keypoint_with_angle(results, idx))

                draw_landmarks(image, results)

                cv2.imshow('img', image)   
                if cv2.waitKey(1) == ord('q'):
                    break
            data = np.array(data)
            print(action, data.shape)
            np.save(os.path.join('dataset', f'raw_{action}_{created_time}_{secs_for_action}sec'), data)

            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('dataset', f'seq_{action}_{created_time}_{secs_for_action}sec'), full_seq_data)

            break

    cap.release()
    cv2.destroyAllWindows()



