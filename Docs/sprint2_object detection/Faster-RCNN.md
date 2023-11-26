---
layout: post
title: "[Paper] Faster R-CNN: Towards Real-Time Object\r Detection with Region Proposal Networks"
author: Kade
categories:
  - Devlog
  - ML/DL
image: 
description: ""
tags:
  - CV
  - Paper
  - ObjectDetection
sitemap: true
---


# Overview of R-CNN structure

[![image](https://github.com/mouseku/Image/assets/64977390/c4de6db7-a1ac-489f-8422-c14a52b5b8b6)](https://www.arxiv-vanity.com/papers/1908.03673/)

[![image](https://github.com/mouseku/Image/assets/64977390/7d09374e-4c2a-4ec1-8eec-a77f58f059b9)](https://www.arxiv-vanity.com/papers/1908.03673/)

---

# Faster R-CNN

Fast RCNN에서 CNN 연산을 공유하고 RoI Pooling을 적용함으로써 비용도 감소시키고 깊은 네트워크에서의 학습도 가능하도록 만들었다. Faster RCNN은 Fast RCNN의 구조에서 실시간 객체 탐지를 위한 속도 개선에 초점을 맞춰 구조를 개선했다.

[![image](https://github.com/devkade/Image/assets/64977390/8c554a91-ff9e-48c3-b044-a6ede85e0973)
](https://paperswithcode.com/method/faster-r-cnn)
Faster R-CNN의 구조를 간단하게 압축해보면 RPN(Regional Proposals Networks) + Fast R-CNN과 같고, 전체적인 동작 순서는 아래와 같다.

1. 전체 이미지를 CNN에 입력하여 특징맵을 추출한다.
2. 특징맵을 RPN에 입력해 Regional Proposals를 추출한다.
3. 특징맵을 RPN을 통해 뽑아낸 Regional Proposals에 따라 특징맵을 RoI Pooling 하고 고정된 크기의 특징맵을 추출한다.
4. 고정된 크기의 특징맵을 통해 분류, 객체 탐지를 한다.

- CNN을 통한 특징맵이 RPN, Classifier에 함께 사용되면서 특징맵을 공유하는 형태를 갖는다.


# Problems of Fast R-CNN

[![image](https://github.com/mouseku/Image/assets/64977390/82c51758-cceb-4b7c-9907-455a5a425964)](https://cseweb.ucsd.edu/classes/sp17/cse252C-a/CSE252C_20170426.pdf)

- Fast R-CNN은 전체 이미지를 입력 받고, RoI Pooling, Softmax를 통한 End-to-End 학습을 할 수 있는 구조를 만들어내며 더 개선된 구조를 보였다. 
- 하지만, Fast R-CNN에서 Regional Proposals를 추출할 때 사용하는 Selective Search 알고리즘은 Regional Proposals를 추출하는 시간이 많이 걸린다는 단점을 갖는다.

[![image](https://github.com/devkade/Image/assets/64977390/c4f90153-1e9b-4cb1-b5fd-424583d61073)](https://www.mdpi.com/2078-2489/10/2/37)
- 위 그림은 Regional Proposals를 추출하는 알고리즘마다 시간, 결과에 대해 측정한 도표이다. 
- 주로 사용됐던 SS의 경우 다른 방식에 비해 확실히 많은 시간이 걸리는 것을 확인할 수 있다.
- SS보다 더 빠른 EdgeBoxes 방식이 고안되었으나, 그럼에도 Regional Proposal 부분의 시간 소모가 많다.
- Fast R-CNN에서 사용한 기존의 Regional Proposals 방식은 CPU를 기반으로 구현되었고 딥러닝 네트워크는 GPU를 사용한다. 때문에 Faster R-CNN에서는 Regional Proposals 방식에서 병목 현상이 발생한다는 점에 주목했다.

병목 현상을 해결하기 위해 CNN을 사용해 Region Proposals를 추출하는 RPN과 Anchor Box 고안했다. RPN과 Anchor Box는 다음의 장점을 갖는다.

1. 학습한 가중치를 Proposals을 추출할 때 공유해 사용할 수 있기 때문에 추론 단계에서 Proposal을 계산하는 비용이 매우 작다. (이미지 당 10ms)
2. Regional Proposals를 추출하는 방식까지 End-to-End로 학습해 더 좋은 객체를 감지할 수 있다.
3. CNN도 RPN을 통해서 attention 매커니즘과 유사하게 특정 부분을 더 집중해서 볼 수 있도록 학습한다. 


# Network Structure

![image](https://github.com/devkade/Image/assets/64977390/7b796950-100a-4ac5-bf96-5884b82a0253)

Faster R-CNN의 전체적인 네트워크 구조는 다음과 같다. [herbwood님](https://herbwood.tistory.com/10)의 네트워크 그림을 참고했다. Faster R-CNN은 다음과 같이 RPN, Fast R-CNN의 두 구조로 나누어지고, VGG를 통과한 같은 특징맵이 RPN, Fast R-CNN으로 나누어져 공유된다. 


## RPN

RPN(Regional Proposal Networks)는 FCN(Fully Convolutional Networks)로 이루어져 있어, 위의 네트워크 그림과 같이 3x3 conv, 1x1 conv를 적용한다. 

[![image](https://github.com/devkade/Image/assets/64977390/eac6f3df-39a0-41a5-ae30-0cd1db586ca6)](https://arxiv.org/pdf/1506.01497.pdf)

위는 RPN의 구조이다. RPN의 작동 방식은 다음과 같다.
1. 위 네트워크의 구조와 같이 VGG를 통과한 특징맵에 3x3 conv를 적용시킨다. (패딩을 적용해 특징맵의 크기는 변하지 않고 동일하다.)
2. Sliding Window 방식을 사용하여 특징맵의 cell 각각에 대해 k개의 anchor box들을 사용하여 객체를 탐지한다. 
3. 1x1 conv를 사용해 각 anchor box에 대한 cls layer, reg layer의 값을 산출한다.
4. Anchor box 마다 cls, reg layer를 통해 객체 포함 유무 점수를 가지고 조정된 bbox 값을 갖는다.
5. Non-Maximum Suppression을 사용하여 가장 객체 포함 점수가 높은 bbox를 산출한다. 

전체 네트워크 구조 그림에서 RPN 구조를 보면 1x1 conv가 2가지로 나누어지는 것을 볼 수 있는데, 위 그림에서 볼 수 있다시피 Classifier, Bbox regressor로 나누어진다. 
- Classifier : 각 Anchor box에 대해 객체 포함 유무에 대한 점수를 매긴다. (Object, Not Object 점수를 지녀야 하기 때문에 h x w x 2\*9 특징맵을 추출한다.)
- Bounding box regressor : Anchor box의 위치를 조정하는 bbox 조정 좌표를 설정한다. (4개의 좌표값을 가져야 하기 때문에 h x w x 4\*9 특징맵을 추출한다.)


### Anchor Box

객체를 탐지할 때 우리는 Selective Search와 같은 방식을 사용했다. 이 방식은 segmentation을 적용해 분류한 객체에 대해 bbox를 적용해 영역을 추출하는 방식이었다. 하지만 따로 객체를 분류하는 것 없이 영역을 추출해야 하기 때문에 Anchor를 통해 추출하는 방법을 사용했다. 

![image](https://github.com/devkade/Image/assets/64977390/31337431-353f-4876-b1f7-c88cc44d6cd2)

Conv를 통해 이미지는 다음과 같이 일정 크기의 grid만큼 연결되고, grid cell은 각각 일정 크기의 이미지를 함축하고 있다. 즉, 원본 이미지가 800x800이고 특징맵이 8x8일 때 cell 하나는 100x100 만큼의 이미지 정보를 함축하고 있다는 의미이다. Anchor Box 방식은 이런 성질을 이용해 특징맵에 anchor box를 적용시킨다. 



![image](https://github.com/devkade/Image/assets/64977390/9b666c68-7727-4675-ba49-c1b96db781b8)

Anchor는 위 그림과 같이 일정 크기와 일정 비율로 사전 설정한 k개의 bbox를 의미한다. 이 anchor를 특징맵의 각 cell에 모두 적용하여 다양한 크기의 객체를 인식할 수 있도록 한다. 
- ex) 위 고양이 이미지의 경우(원본 이미지 : 800x800, 특징맵 : 8x8)
- 8x8의 cell에 9개의 anchor를 적용해 8x8x9 개의 anchor가 형성된다. 

Anchor Box는 다음과 같은 식을 통해 정의한다.
$$w\times h = s^2$$
$$w = {1 \over 2} \times h$$
$${1 \over 2} \times {h^2} = s^2$$
$$h = \sqrt{2s^2}$$
$$w = {{\sqrt{2s^2}} \over 2}$$

- s : scale, 정사각형일 때의 길이
- w : width
- h : height
- 위 수식은 width:height = 1:2 인 경우를 의미한다.



[![image](https://github.com/devkade/Image/assets/64977390/c17f421b-ca64-466f-89b4-58d413663bd8)](https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns)
위 그림의 경우 Anchor Box를 사용했을 때 원본 이미지 크기에서 포함할 수 있는 크기를 보여준다. 원본 이미지의 대부분을 포함하는 것을 확인할 수 있다. 
- 원본 이미지 크기 : 800x600
- 특징맵 크기 : 50x38 (16배 sub-sampling)
- 총 anchor 개수 : 50x38x9 = 1900


### RPN Loss

$$L(\left\{  p_i  \right\}, \left\{ t_i \right\}) = \frac {1} {N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac {1} {N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)$$

- $i$ : mini-batch 안 anchor의 index
- $p_i$ : i번째 anchor의 객체가 포함되어 있을 확률
- $p^*_i$ : anchor가 positive인 경우 1, negative인 경우 0 
	- negative인 경우 $L_{reg}$ 적용하지 않는다.
- $t_i$ : 예측 bbox의 조정된 좌표
- $t^*_i$ : 정답 bbox의 좌표
- $L_{cls}$ : 객체인지 아닌지에 대한 log loss
- $L_{reg}$ : Smooth L1 loss
- $N_{cls}$ : mini-batch의 크기 (i.e., $N_{cls}$=256)
- $N_{reg}$ : anchor의 위치의 개수 (i.e., $N_{reg}$~2,400)
- $\lambda$ : balancing parameter (Default = 10) 
	- 10으로 $N_{cls}$와 $N_{reg}$가 유사하게 맞춰서 실험했다.

- Faster R-CNN의 전체 손실 함수와 RPN 손실 함수 2개가 함께 사용된다.

RPN은 End-to-End로 오차역전파, SGD를 통해서 학습된다. 


## Sharing Features (need to complement)

논문에서는 Faster R-CNN을 학습시키기 위해 RPN과 Fast R-CNN을 번갈아 학습하는 Alternating Training 방식을 사용한다. 

1. RPN에서 anchor box와 ground truth box를 통해 positive/negative 데이터셋을 구성해 RPN을 학습시킨다. 이 때 RPN에 사용되는 backbone 네트워크도 미세 조정된다.
2. RPN을 학습한 후 RPN에서 Regional Proposals를 생성해 Fast R-CNN을 학습시킨다. 
3. Fast R-CNN이 학습되면 고정시키고 RPN을 다시 미세 조정한다. 
4. 이후 RPN을 고정하고 Fast R-CNN을 학습한다.

- Backbone 네트워크를 통해 생성한 특징맵을 공유함으로써 서로의 특징을 공유하는 방식을 의미한다.


# Results

1. Sharing Features

![image](https://github.com/devkade/Image/assets/64977390/6da42483-c35b-4577-a255-0131c077bc7c)

특징 공유 측면에서 특징을 공유한 모델이 공유하지 않은 모델보다 더 높은 성능을 보인다는 면에서 Regional Proposals 추출과 탐지기의 특징을 공유하는 것이 더 높은 성능을 보인다는 것을 알 수 있다.  
또한 RPN의 cls와 reg가 없을 때를 각각 확인해봤을 때 reg가 없는 경우 성능이 비교적 많이 하락하는 것을 보아 regressor를 통해 bbox를 조정했을 때 고품질의 Regional Proposals이 추출됨을 알 수 있다. 

2. Inference time

![image](https://github.com/devkade/Image/assets/64977390/6579ed25-1287-4e4b-a645-b88c9e6c9ae6)

추론 성능을 확인해봤을 때 Fast R-CNN은 0.5 fps인 반면 Faster R-CNN은 5 fps, 더 작은 네트워크인 ZF를 사용했을 때는 17 fps를 보이며 추론 속도도 많이 줄였음을 확인할 수 있다. 

3. Comparing Fast R-CNN with Faster R-CNN

![image](https://github.com/devkade/Image/assets/64977390/1a6b0734-a987-4df9-90a6-87496ecf1832)

Faster R-CNN이 이전 Fast R-CNN 보다 더 높은 성능을 보인다는 것을 확인할 수 있다. 


# 참고문헌
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
- [https://herbwood.tistory.com/10](https://herbwood.tistory.com/10)
- [https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/](https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/)
