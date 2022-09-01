# TAXI_DRIVER_KEEPER
2022 한성대 사물인터넷 캡스톤 디자인

---
## 프로젝트 개요

### - 택시<br>
> 택시 내에서 발생하는 각종 위험 요인을 자동으로 식별하여 주의 신호를 발생
### - DB<br>
> 여러 가지 위험 징후들을 데이터화 하여 관리
### - 관제센터<br>
> 택시의 위험도와 내부 영상 모니터링을 통해 위험 상황을 판단해 상황을 인지하고 실시간 위치를 이용하여 빠른 신고 절차로 이어져 상황에 대처할 수 있음
---
## 요구 기술
### 1) 폭력 모션 인식
> Tensorflow Lite, OpenCV, MediaPipe
### 2) AWS
> IoT Core, LAMBDA, DynamoDB, S3, API
### 3) python
> 라프베리파이에서의 구동 언어 Multiprocessing, aws s3, mqtt 모듈 사용
---
## 실행방법
### - 라즈베리 파이
#### 1. 모션 인식 모듈 설치
> Tensorflow Lite, OpenCV, MediaPipe 설치
#### 2. 파이썬 모듈 설치
> Multiprocessing, RPI.GPIO, AWSIOT 파이썬 모듈 설치
#### 3. raspberry_code.py
> 내부의 IAM 정보 추가, ENDPOINT,CLIENT_ID 정보 입력 
``` C
python3 final_test.py #프로그램 실행
```
### - 웹 서버
``` C
#사용 방법 추가 필요
```
---
