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
   김민지의 얼굴을 학습시킨 후 face recognition을 통해 해당 인물을 제외한 unknown을 모자이크 처리한다.
   왼쪽은 일반 모자이크, 오른쪽은 스티커 모자이크이다.
![모자이크2](https://user-images.githubusercontent.com/67955977/131106432-b76cab82-91f2-4519-9301-44deb8b64fb7.PNG)
   위와 같은 과정으로 김서원의 얼굴을 학습시킨 후 프로세싱해 추출한 이미지이다. 


## 개발 도구
* Anaconda 
* Python
* Google colab 
* 라이브러리 : OpenCV, Tensorflow, FFmpeg, FaceRecognition, pickle

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

## 테스트 실행하기
이 시스템을 위한 자동화된 테스트를 실행하는 방법을 적어주세요.

## 배포
추가로 실제 시스템에 배포하는 방법을 노트해 두세요.

