# UnknownProcessing (백종원 모자이크)

## Team
* 김민지(Minji Kim), 김서원(Seowon Kim)

## Project
![캡처](https://user-images.githubusercontent.com/67955977/131096230-5f90b499-bade-4e01-90d7-8a7b0c1e51dc.PNG)
* 목적 : 특정 인물을 제외한 인물 모자이크 자동화
* 개발 배경 : SNS에서 화제가 된 모자이크 방법인 <백종원 모자이크>. 영상 속에서 백종원 즉, 주요 인물을 제외한 다른 사람들의 얼굴을 해당인물의 스티커로 가려주는 방법이다. 그러나 영상 편집을 할 때 이런 모자이크를 하려면 프레임 단위로 얼굴 하나하나 좌표를 찍으며 편집해야해서 많은 시간이 걸린다. 따라서 편집자들의 노동 시간을 단축시키는 것은 물론, 누구나 손쉽고 간편하게 센스있는 모자이크를 사용할 수 있다는 취지로 서비스를 구현하게 되었다.
* 목표
    * 뉴스 또는 유튜브에서 일반인 모자이크 처리를 자동화 하여 영상 편집 시간을 단축시킨다
    * 영상에 의도치 않게 출연한 일반인들을 자동 모자이크 처리 하여 초상권 침해를 방지한다


## Demo
![모자이크1](https://user-images.githubusercontent.com/67955977/131105868-b3a15009-d455-40e5-a56b-79726225b521.PNG)
* 특정 인물(minji)의 얼굴을 학습시킨 후 face recognition을 통해 해당 인물을 제외한 unknown을 모자이크 처리한다.
* 왼쪽은 일반 모자이크, 오른쪽은 스티커 모자이크이다.
   
   
![모자이크2](https://user-images.githubusercontent.com/67955977/131106432-b76cab82-91f2-4519-9301-44deb8b64fb7.PNG)
* 위와 같은 과정으로 특정 인물(seowon)의 얼굴을 학습시킨 후 프로세싱해 추출한 이미지이다. 


## 개발 도구
* Anaconda 
* Python
* Google colab 
* 라이브러리 : OpenCV, Tensorflow, FFmpeg, FaceRecognition, pickle


## 구조
![structure](https://user-images.githubusercontent.com/67955977/131110753-780676cb-7684-419c-9e83-271b36ac632d.PNG)


## 가상환경 세팅

1. 아나콘다 프롬프트 or CMD 실행
2. (pip 패키지 업그레이드)
    ```bash
    conda upgrade pip
    pip install upgrade
    ```
3. 가상환경 새로 설치
    ```bash
    conda create -n (env) python=3.7 activate (env)
    ```
4. tensorflow 설치
    ```bash
    pip install tensorflow==2.0
    ```
5. 버전 확인
    ```bash
    python import tensorflow as tf tf.__version__
    ```
6. 라이브러리 설치
    ```bash
    pip install numpy matplotlib pillow opencv-python
    ```
    
    
https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

## 실행
1. PretrainedModel 폴더 다운로드   
2. Build_Dataset_CaffeCNN.py로 dataset 생성 (webcam)  
   ```bash
   python build_dataset_CaffeCNN.py --output dataset/swkim/
   ```
   output경로에 특정 인물의 이름으로 폴더를 만들어 두고 경로를 지정한다.   
   설정해둔 횟수에 도달하면 자동 종료된다.   
3. encode_faces.py로 pickle 파일 생성   
   ```bash
   python encode_faces.py --dataset dataset --encodings encodings.pickle
   ```
   dataset의 경로와 encodings의 경로를 지정해준다.  
   이미지를 BGR에서 RGB로 변환하고 얼굴에 해당하는 영역의 좌표를 감지한다.  
   face_encodings 함수를 호출하면 얼굴 영역을 128 크기의 vector로 변환한다.  
   모든 얼굴이 변환되어 encodings 변수에 담기게 되고, pickle 파일이 완성된다.  

4. stickers 폴더에 스티커 이미지 파일 생성   
5.   
   **1) Image Processing**
   ```bash
   python recognize_faces_image.py --encodings encodings.pickle --image testset/test.jpg --method overlay --sticker stickers/mj.png
   ```
   pickle 파일 경로, test image 경로를 지정하고, method로 mosaic(일반 모자이크), overlay(스티커)를 지정한다.    
   method를 overlay로 설정하면 sticker 경로도 지정해준다.  
   
   **2) Video Processing**
   ```bash
   python unknown_processing_video.py --encodings encodings.pickle --input videos/video.mp4
   ```
   ```bash
   python unknown_processing_video.py --encodings encodings.pickle --input videos/video.mp4 --method overlay --sticker stickers/osw.png
   ```
   
   **3) WebCam Processing**
   ```bash
   python recognize_faces_video.py --encodings encodings.pickle
   ```
   ```bash
   python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0 --method overlay --sticker overlay_stickers/sticker.png
   ```
