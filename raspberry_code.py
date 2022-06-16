from multiprocessing import Process, Queue
from re import S
import RPi.GPIO as GPIO
import serial
from awscrt import io, mqtt
from awsiot import mqtt_connection_builder
import time
import json
import sys
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import boto3
from botocore.client import Config

# < 딥러닝 모션 분류 모델 >

def mediapipe_detection(image, model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = model.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  return image, results

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
    # angle_label = np.append(angle_label, idx) # 데이터 셋 구성 시 마지막에 정답 라벨 포함하는 코드

    return np.concatenate([joint.flatten(), angle_label.flatten()])    

def prob_viz(res, threshold, input_frame): # 이미지에 게이지 만드는 코드
    title = ["SAFETY", "WARNING", "DANGER"]
    colors = [(112,255,112),(112,255,255),(52,52,255)]
    output_frame = input_frame.copy()
    res = [sum(res[0:2]), res[2], sum(res[3:])] # 같은 클래스 통합
    cv2.putText(output_frame, "Security level", (460, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (520-int(prob*100),260+num*40), (520, 290+num*40), colors[num], -1)
        cv2.putText(output_frame, title[num], (540, 285+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[num] if prob > threshold else (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def motion_detecting(model_path, thres, set_time, taxi_id, mqtt_msg_q):
    interpreter = tf.lite.Interpreter(model_path=model_path) # './LSTM_model.tflite'
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    mp_pose = mp.solutions.pose

    actions = np.array(['sf_sit', 'sf_phone', 'w_close', 'dg_hit', 'dg_neck'])
    mqtt_msg_q.put({"type":1, "security" : "SAFETY"})

    sequence = []
    sentence = ["loading"]
    threshold = thres # 0.75

    motion_active = 0
    vid_active = 0
    secure_time = set_time

    # 실시간 캠
    cap = cv2.VideoCapture(0) 
    # 동영상
    # video_file = "test_mp4/final.mp4"
    # cap = cv2.VideoCapture(video_file) 

    # 비디오 추출
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    
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

            image, results = mediapipe_detection(image, pose)
            now_time = time.time()
            # 화면에 landmarks 그리기
            # mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 특징 추출
            if results.pose_landmarks is not None: 
                keypoints = extract_keypoint_with_angle_no_face(results)
                sequence.insert(0,keypoints)
                sequence = sequence[:30]

            # 행동 분류
            if len(sequence) == 30:
                input_data = np.array(np.expand_dims(sequence,axis=0), dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                res = interpreter.get_tensor(output_details[0]['index'])[0]
                # print(actions[np.argmax(res)])

                # 결과 가시화
                if res[np.argmax(res)]>threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] == sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                        else :
                            sentence = []
                    else : 
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 10:
                    if sentence[0] in actions[2]:
                        mqtt_msg_q.put({"type":1, "security" : "WARNING"})
                        motion_active = now_time
                    elif sentence[0] in actions[3:]:
                        mqtt_msg_q.put({"type":1, "security" : "DANGER"})
                        motion_active = now_time
                        if not vid_active : 
                            vid_active = now_time
                            vid_name = time.strftime("%y년%m월%d일%H시%M분%S초", time.localtime(vid_active))
                            out = cv2.VideoWriter(f'./video/{vid_name}.avi', fourcc, 25.0, (720,400))
                    else :
                        if motion_active and now_time - motion_active > secure_time : 
                            mqtt_msg_q.put({"type":1, "security" : "SAFETY"})
                            print({"type":1, "security" : "SAFETY"})
                            motion_active = 0
                    sentence = []

                image = prob_viz(res, threshold, image)
            
            # 영상 채증    
            if vid_active :
                if now_time - vid_active < 15 :  # 채증 시간(초)
                    out.write(image)
                else : 
                    out.release()
                    aws_s3_send(f'./video/{vid_name}.avi', vid_name, "avi", taxi_id)
                    vid_active = 0

            if int(now_time) % 3 == 0:  
                cv2.imwrite(f"./images/preview.jpg", image)
                aws_s3_send(f"./images/preview.jpg", "preview", "jpg", taxi_id)
            

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', image) 

            if cv2.waitKey(3) & 0xFF == 27:
                break
    
    cap.relaese()

def aws_s3_send(file_path, active_time, type, taxi_id):
    ACCESS_KEY_ID = '' #s3 관련 권한을 가진 IAM계정 정보
    ACCESS_SECRET_KEY = ''
    BUCKET_NAME = ''

    s3 = boto3.client(
        's3', 
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        config=Config(signature_version='s3v4'))
    s3.upload_file(file_path, BUCKET_NAME, f"{taxi_id}/{active_time}.{type}") # 폴더 생성 가능


# < GPS 센서 >

def convert_to_degrees(raw_value):
    decimal_value = raw_value/100.00
    degrees = int(decimal_value)
    mm_mmmm = (decimal_value - int(decimal_value))/0.6
    position = degrees + mm_mmmm
    position = "%.4f" %(position)
    return position

def gps(mqtt_msg_q):
    gpgga_info = "$GPGGA,"
    ser = serial.Serial ("/dev/ttyS0")
    while True:
        received_data = (str)(ser.readline())                   
        GPGGA_data_available = received_data.find(gpgga_info)   
          
        if (GPGGA_data_available>0):
            GPGGA_buffer = received_data.split("$GPGGA,",1)[1]  
            NMEA_buff = (GPGGA_buffer.split(','))

            lat = float(NMEA_buff[1])   
            longi = float(NMEA_buff[3])
            lat_data = convert_to_degrees(lat)    
            long_data = convert_to_degrees(longi) 

            mqtt_msg_q.put({"type":2, "lat" : lat_data, "long" : long_data})
            time.sleep(10)

# < 소음, 알코올 센서 >
def sensor(set_time, mqtt_msg_q):
    soundpin = 7
    alcoholpin = 22
    sound_active = 0
    alcohol_active = 0
    sound_go, alcohol_go = 0, 0
    secure_time = set_time # 신호 유지 시간 

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(soundpin, GPIO.IN)
    GPIO.setup(alcoholpin, GPIO.IN)

    while True:
        time.sleep(0.5)
        now_time = time.time()
        sound = GPIO.input(soundpin)
        alcohol = GPIO.input(alcoholpin) 

        alcohol ^= 1

        if sound: 
            if not sound_active :
                sound_go = 1
            sound_active = now_time
        else : 
            if sound_active and now_time - sound_active > secure_time :
                sound_go = 1
                sound_active = 0

        if alcohol: 
            if not alcohol_active :
                alcohol_go = 1
            alcohol_active = now_time
        else : 
            if alcohol_active and now_time - alcohol_active > secure_time :
                alcohol_go = 1
                alcohol_active = 0

        if sound_go or alcohol_go :
            mqtt_msg_q.put({"type" : 3,"sound" : str(sound), "alcohol": str(alcohol)})
            sound_go = 0
            alcohol_go = 0

if __name__ ==  '__main__':
    mqtt_msg_q = Queue()
    try:
        # TODO : AWS 기기 엔드포인트 정보 입력 필요
        ENDPOINT = ""
        CLIENT_ID = ""
        PATH_TO_CERT = "./home/pi/certs/certificate.pem.crt"
        PATH_TO_KEY = "./home/pi/certs/private.pem.key"
        PATH_TO_ROOT = "./home/pi/certs/Amazon-root-CA-1.pem"
        TOPIC = "taxi/signal" # 토픽 
        TAXI_ID = "12하 1234"


        event_loop_group = io.EventLoopGroup(1)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        mqtt_connection = mqtt_connection_builder.mtls_from_path(
                    endpoint=ENDPOINT,
                    cert_filepath=PATH_TO_CERT,
                    pri_key_filepath=PATH_TO_KEY,
                    client_bootstrap=client_bootstrap,
                    ca_filepath=PATH_TO_ROOT,
                    client_id=CLIENT_ID,
                    clean_session=False,
                    keep_alive_secs=6
                    )

        print("Connecting to {} with client ID '{}'...".format(
                ENDPOINT, CLIENT_ID))
        connect_future = mqtt_connection.connect()
        connect_future.result()
        print('Begin Publish')

        p1 = Process(target=motion_detecting, args=("LSTM_model.tflite", 0.75, 5, TAXI_ID, mqtt_msg_q, ))
        p2 = Process(target=gps, args=(mqtt_msg_q, ))
        p3 = Process(target=sensor, args=(5, mqtt_msg_q, ))
        p1.start()
        p2.start()
        p3.start()
        

        while 1:
            time.sleep(1)
            while not mqtt_msg_q.empty():
                message = mqtt_msg_q.get()
                message['taxi_id'] = TAXI_ID
                mqtt_connection.publish(topic=TOPIC, payload=json.dumps(message), qos=mqtt.QoS.AT_LEAST_ONCE)
                print(message)

        p1.join()
        p2.join() 
        p3.join()             

    except KeyboardInterrupt:
        disconnect_future = mqtt_connection.disconnect()
        disconnect_future.result() 
        mqtt_msg_q.close()
        cv2.destroyAllWindows()
        sys.exit(0)