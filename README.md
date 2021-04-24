# UnknownProcessing (백종원 모자이크)

## Team
* 김민지(Minji Kim), 김서원(Seowon Kim)

## Project
* 목적 : 특정 인물을 제외한 인물 모자이크 자동화
* 목표
    * 뉴스 또는 유튜브에서 일반인 모자이크 처리를 자동화 하여 영상 편집 시간을 단축시킨다
    * 영상에 의도치 않게 출연한 일반인들을 자동 모자이크 처리 하여 초상권 침해를 방지한다
## Demo


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

## 사용된 도구
* [Tensorflow](https://www.tensorflow.org/api_docs)
* google colab
* FFmpeg
* opencv

