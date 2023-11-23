
# 1.overview

## 1.1objective
1. translation invariance 문제를 해결해보자.
2. 모델을 fully convolutional 하게 만들어보자.

## 1.2 Background

### translational invariance vs translational variance
#### translational invariance의 정의

positional invariance(translation invariance): 위치가 변하여도 결과가 똑같아야함
= 위치가 영향을 주지 않음
![image](https://github.com/ownvoy/DeepSync/assets/96481582/ea1a1ece-882c-4156-8e4c-ffb50d2a800b)

image classification에서의 주요 과제

> cnn은 translational invaraince하다.
##### weight sharing

- convolutiona filter를 활용한 계산은  원래 translational equivariance(translational variance)함. 
- 층이 깊어질 수록 tralational invariance가 됨. 
- 그 이유는 계속 같은 필터를 써서(weight sharing)
![image](https://github.com/ownvoy/DeepSync/assets/96481582/218e5479-1ad5-4dcf-a9f1-40eacf09da62)


##### max pooling
- max pooling 역시 translational invariance한 연산
![image](https://github.com/ownvoy/DeepSync/assets/96481582/474ff53e-ff66-48d0-9007-722c0beeaa70)


`cnn`은 어떤 위치에 사물이 있어도 잘 classify한다. (translational invariance)
#### Translational equivariance(variance)

위치가 변하면 결과가 변함
= 위치에 영향을 받음
object detection에서의 bounding box

> object detection은 translational invariance와 translational variance의 dielemma를 가지고 있음

### 1.2.2 이전 모델의 한계점

#### R-CNN
- R-CNN에서는 AlexNet이나 VGGNet을 갔다 씀
- 5번째 conv이후로는 fully-connected layer를 썼기에, weight sharing이 끊김.
![image](https://github.com/ownvoy/DeepSync/assets/96481582/f35d6f12-ab95-468b-a5f0-ea5a17e31da5)

[R-CNN | Region Based CNNs - GeeksforGeeks](https://www.geeksforgeeks.org/r-cnn-region-based-cnns/)

층이 깊다는 특징을 가지고 있음=> image classify는 잘하는데 detection은 썩?


![image](https://github.com/ownvoy/DeepSync/assets/96481582/6727f93a-00fb-4663-b8dd-da21567b263b)
- R-CNN: RoI 하나 당 CNN을 돌리니까 101층 모두 RoI-wise
- Faster R-CNN: 마지막에 RoI pooling layer 이후 RoI-wise 계산
![image](https://github.com/ownvoy/DeepSync/assets/96481582/79c3b274-263d-40bf-a686-ba8d517868d9)

- R-FCN: 101층 모두 shared, fully convolutional architectures
> Faster R-CNN의 마지막 Fully Connected Layers를 없애 보자.

# 2 Main

## 2.1 Overview
![image](https://github.com/ownvoy/DeepSync/assets/96481582/86308dbc-8c17-4022-9e59-48f50223648d)

- 마지막 conv layer를 통과한 결과물: $k^{2}\times(C+1)$
- $k$는 class의 위치 정보를 나타냄. 아래 그림서 $k=3$
  
![image](https://github.com/ownvoy/DeepSync/assets/96481582/8fb94ff0-cb32-418f-9830-62076db9ac38)

- $C$: object category

![image](https://github.com/ownvoy/DeepSync/assets/96481582/748598ca-764c-417d-993b-433396814f1f)

- 자전거에 대해 보자고 할 때, 그림 중앙에는 몸통이 있고 왼쪽 아래에는 앞바퀴가 있는 것을 알 수 있다.(사진에 대한 대략적인 정보)

![image](https://github.com/ownvoy/DeepSync/assets/96481582/939a874f-d1e8-4929-a4b1-c536b5e54fc9)

- 카테고리가 $C+1$개, $k\times k$개의 bin

![image](https://github.com/ownvoy/DeepSync/assets/96481582/6aea6550-1a71-43c3-bb9b-26f7ccd693f2)

> __마지막 conv layer를 지나고 나온 결과물__

### position-sensitive RoI pooling

$r_{bike}$이라는 $3\times3$ table이 있다고 할 때, 그 중 맨 위 왼쪽 칸($i=0,j=0$)을 어떤 식으로 pooling 할까?

RPN에서 나온 RoI들이 다음과 같다고 할 때,

![image](https://github.com/ownvoy/DeepSync/assets/96481582/56ea4969-e08a-409e-ba67-a07172c75d05)

$r_{bike}(0,0)$ 은 맨 왼쪽 위 칸에 대한 정보(핸들)를 이용하여 pooling 하고 싶을 것.

![image](https://github.com/ownvoy/DeepSync/assets/96481582/9d5dab84-207b-447a-8f74-4b18e145cb45)

핸들에 대한 정보가 $z_{0,0,bike}$임
$z_{0,0,bike}$와 RoI의 각 bin들과 곱해줘서 average pooling 해줄 거임.
$$r_{bike}(0,0) = \frac{z_{0,0,bike} \times (x_0,y_0)+z_{0,0,bike}\times(x_0,y_1)+ \cdots +z_{0,0,bike}\times(x_2,y_2)}{9}$$

결국 $r_{bike(0,0)}$은 $z_{0,0,bike} \times (z_0,y_0)$ 이 가장 많이 반영 되고, 다른 bin들은 섞여 들어갈 것. (self-attention이랑 비슷하다고 느낌)

마찬가지로, $r_{bike}(0,1)\cdots r_{bike}(2,2)$ 모든 셀에 대해 구할 수 있음

$r_{bike}$는 $r_{bike}(0,1)\cdots r_{bike}(2,2)$의 sum으로 구함.
$$r_{c}= \sum\limits_{i,j} r_c(i,j)$$
마찬가지로 $r_{dog},  r_{backgroud}$ 등을 구할 수 있을 것임.
이 값 을 활용하여 roi가 뭔지 맞출 수 있음(softmax function)

$$s_{c}=\frac{e^{r_c}}{\overset{C}{\underset{c=0}{\sum}} e^{r_c}}$$

이 $s_c$는 뒤에서 cross-entropy loss로 쓰임

- 그냥 계산만 한거여서 learnable layer가 아님. => speed up

![image](https://github.com/ownvoy/DeepSync/assets/96481582/f01d3c9c-8791-4c8e-b713-b1f889d07eba)


1. positional sensitive RoI pooling을 해서 __translation invariance__ 문제 해결
2. __fully connected layer를 없앰__ 으로써 속도 향상 + end-to-end 학습


### bounding box regression 
비슷하게 마지막 layer로 $4k^2$-d convolutional layer

## 2.2 training

### loss

summation of the cross-entropy loss and the box regression loss

![image](https://github.com/ownvoy/DeepSync/assets/96481582/fc5ea835-8845-417b-8637-5bfa571a54ff)


### online hard example mining(OHEM)

- N개의 RoI마다 loss를 구함 
- loss가 큰 순서로 정렬
- loss가 크다는 것은 어려운 sample이라는 거임
- 큰 순서대로 $B$개 뽑아, 학습

![image](https://github.com/ownvoy/DeepSync/assets/96481582/dd72e7ad-307f-437d-888f-840cec1b7761)

- RoI 하나당 계산하는 시간이 거의 _cost-free_ 임. 그래서 training time이 많이 늘어나지 않음(Faster R-CNN에서는 2배 증가)

## 2.3 Results

![image](https://github.com/ownvoy/DeepSync/assets/96481582/9c8e4df0-92a7-43ef-997d-ea3371d9c6f1)

test time이 faster r-cnn에 비해 압도적으로 빠른 것을 알 수 있다.

