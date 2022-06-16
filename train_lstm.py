import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt

# tensorboard --logdir=.파일명



actions = np.array(['safe', 'safe_phone', 'close', 'danger_hit', 'danger_neck'])

data = np.concatenate([
    np.load('dataset/no face/seq_safe_1649913288_240sec.npy'), 
    np.load('dataset/no face/seq_safe_phone_1649913288_240sec.npy'), 
    np.load('dataset/no face/seq_close_1649913288_240sec.npy'),
    np.load('dataset/no face/seq_danger_hit_1649913288_240sec.npy'),
    np.load('dataset/no face/seq_danger_neck_1649913288_240sec.npy'),
    ], axis=0)



x_data = data[:, :, :-1] # 정답 label 값을 분리
labels = data[:, 0, -1] # 시퀀스기 때문에 첫30개 다 같으므로 번째 label만 가져옴
y_data = to_categorical(labels, num_classes=len(actions))


x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2022) # TODO: 랜덤스테이트 먼지 모름

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


## LSTM 모델
#LSTM 모델 / 다중 클래스 분류

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
# early_stopping = EarlyStopping(monitor='val_loss')
model_name = 'tanh64'
# relu sigmoid tanh
model = Sequential()

model.add(LSTM(64, return_sequences=False, activation='tanh', dropout=0.1, input_shape=(30,58)))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
# model.load_weights(f'weights_{model_name}')

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()
# model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=100, callbacks=[tb_callback])
# model.save_weights(f'weights_{model_name}')
# model.save(model_name)
# tf.saved_model.save(model, './mod2')





# 실시간 적용
sequence = []
sentence = ["loading"]
threshold = 0.75

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

def extract_keypoint_with_angle(results):
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
    # angle_label = np.append(angle_label, idx) # 정답 라벨 포함

    return np.concatenate([joint.flatten(), angle_label.flatten()]) 

def extract_keypoint_with_angle_no_face(results): # 13*4+6+1 = 59
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
    # angle_label = np.append(angle_label, idx) # 정답 라벨 포함

    return np.concatenate([joint.flatten(), angle_label.flatten()])    

def prob_viz(res, actions, input_frame):
    colors = [(16,245,16),(16,245,16),(16,235,235),(16,16,245),(16,16,245)]
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*150), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame
# 실시간 캠
# cap = cv2.VideoCapture(0) 
# 동영상
video_file = "test_mp4/test1.mp4"
cap = cv2.VideoCapture(video_file) 

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    # 캠에서 이미지 읽기
    success, image = cap.read()
    image = cv2.resize(image, (720, 400), interpolation=cv2.INTER_AREA)
    image = cv2.flip(image, 1)
    
    if not success:
      print("Ignoring empty camera frame.")
      continue
    # mediapipe 모션인식
    image, results = mediapipe_detection(image, pose)

    # 화면에 landmarks 그리기
    draw_landmarks(image, results)

     # 특징 추출
    if results.pose_landmarks is not None: 
        keypoints = extract_keypoint_with_angle_no_face(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:30]

    # 행동 분류
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence,axis=0))[0]
        print(actions[np.argmax(res)])
        

    # 결과 가시화
        if res[np.argmax(res)]>threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else : 
                sentence.append(actions[np.argmax(res)])
        
        if len(sentence) > 5:
            sentence = sentence[-5:]
            
        image = prob_viz(res, actions, image)

        # cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
        # cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)
        cv2.rectangle(image, (0,0), (150,40), (245,117,16), -1)
        cv2.putText(image, sentence[-1], (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)        

    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(2) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()




# 레이어를 최적화한 배경 기록, 활성함수 어케 바꿧는지 레이어 어떻게 바꿧는지 등등 
# 외형 그냥 깔끔하게 해라
# 특징 나눈것들 차이 적어두기 러닝 코스트 등