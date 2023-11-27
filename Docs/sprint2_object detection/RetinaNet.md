# Overview
## Objective
**Positive / Negative example의 class imbalance problem을 해결하는 방법을 이해한다**
+ ![image](https://github.com/devkade/DeepSync/assets/11837072/3e648c34-ac83-4ea8-bf96-2aec32a6eb99)<br/>

## Background
### Terminology
+ Region proposals
+ bounding box
+ anchor box
+ regression box
+ classification box
+ object class
+ sampling heuristic
+ dense sampling
+ class imbalance
    + easy negative / hard positive인 경우에서 negative sample이 easy sample에 에 비해서 상당히 많아 class sample이 불균형인 것
+ positive / negative sample
    + positive는 target object, negative는 background
+ easy / hard sample
    + easy는 검출이 쉬운 sample, hard는 검출이 어려운 sample
    + hard positive / easy negative 또는 easy positive / hard negative case가 있을 수 있음
+ cross entropy(CE)
+ balanced cross entropy(BCE)
+ precision recall   
![image](https://github.com/devkade/DeepSync/assets/11837072/e0f743fb-0d6e-4c87-bd3b-c853673f132c)<br/>
    + **precision**
        + ![image](https://github.com/devkade/DeepSync/assets/11837072/5dfc6090-6c35-4b1e-9439-7d65b55a8acd)<br/>

    + **recall**   
        + ![image](https://github.com/devkade/DeepSync/assets/11837072/cb1497b5-8ea6-4e6c-a323-42422a740489)<br/>

+ precision recall graph   
![image](https://github.com/devkade/DeepSync/assets/11837072/b6f7816b-8c38-46f9-9078-09cc253efde8)<br/>

+ average precision loss(AP-loss)   
![image](https://github.com/devkade/DeepSync/assets/11837072/5a3c73cf-e632-4773-9873-01a7cf00f4ad)<br/>

<br></br>
### Type of object detection
**two-stage object detection**   
![image](https://github.com/devkade/DeepSync/assets/11837072/bfecfbb1-2858-4ec1-9dbe-caebe4cab44d)<br/>

**one-stage object detection**   
![image](https://github.com/devkade/DeepSync/assets/11837072/e255dcc4-786a-47f4-81f4-63215e25c370)<br/>
+ image당 10,000 ~ 100,000개의 candidate 존재함
+ 하지만 이중 target object을 포함한 candidate는 매우 적음
+ background는 easy negative sample이 되고 foreground는 hard positive sample이 됨

***
<br></br>
# Main
## Overview
![image](https://github.com/devkade/DeepSync/assets/11837072/eba719f9-cf04-4f7c-8ab1-0b70fb32c9ab)<br/>

### Loss design
+ **cross entropy loss**   
![image](https://github.com/devkade/DeepSync/assets/11837072/3b0cbc46-f65f-4741-92e6-39b8bc6e2578)<br/>

+ **balanced cross entropy loss**   
![image](https://github.com/devkade/DeepSync/assets/11837072/fae75310-4406-4ed7-907a-30a197a6ee2d)<br/>
    + a weighting factor α∈[0,1] for class 1 and 1−α for class−1

+ **focal loss**   
![image](https://github.com/devkade/DeepSync/assets/11837072/a990ec87-03c7-40dc-b9e4-cc6faf358d06)<br/>
    + For instance, with γ=2, an example classified with pt=0.9 would have 100× lower loss compared with CE and    
    with pt≈0.968 it would have 1000× lower loss
    + **balanced focal loss**   
    ![image](https://github.com/devkade/DeepSync/assets/11837072/650a47f8-0d66-4fbb-9388-9062c738a84d)<br/>

### Training
1. Feature Pyramid by ResNet + FPN
![image](https://github.com/devkade/DeepSync/assets/11837072/48d79be4-7c3a-4fe0-adfe-fa03b077d588)<br/>
+ **Input : image**
+ **Process : feature extraction by ResNet + FPN**
+ **Output : feature pyramid(P5~P7)**
    + 먼저 이미지를 backbone network에 입력하여 서로 다른 5개의 scale을 가진 feature pyramid를 출력합니다.    
    여기서 backbone network는 ResNet 기반의 FPN(Feature Pyramid Network)를 사용합니다. pyramid level은 P3~P7로 설정합니다.
<br></br>
2. Classification by Classification subnetwork
![image](https://github.com/devkade/DeepSync/assets/11837072/0872af74-d958-4760-8e60-6e8a0be15421)<br/>
+ **Input : feature pyramid(P5~P7)**
+ **Process : classification by classification subnetwork**
+ **Output : 5 feature maps with KxA channel** 
    + 1)번 과정에서 얻은 각 pyramid level별 feature map을 Classification subnetwork에 입력합니다.   
    해당 subnet는 3x3(xC) conv layer - ReLU - 3x3(xKxA) conv layer로 구성되어 있습니다.   
    여기서 K는 분류하고자 하는 class의 수를, A는 anchor box의 수를 의미합니다. 논문에서는 A=9로 설정합니다.    
    그리고 마지막으로 얻은 feature map의 각 spatial location(feature map의 cell)마다 sigmoid activation function을 적용합니다.    
    이를 통해 channel 수가 KxA인 5개(feature pyramid의 수)의 feature map을 얻을 수 있습니다. 
<br></br>
3. Bounding box regression by Bounding box regression subnetwork
![image](https://github.com/devkade/DeepSync/assets/11837072/7d668f9b-f9b6-4f8e-bee7-da56a250f7b6)<br/>
+ **Input : feature pyramid(P5~P7)**
+ **Process : bounding box regression by bounding box regression subnet**
+ **Output : 5 feature maps with 4xA channel**
    + 1)번 과정에서 얻은 각 pyramid level별 feature map을 Bounding box regression subnetwork에 입력합니다.    
    해당 subnet 역시 classification subnet과 마찬가지로 FCN(Fully Convolutional Network)입니다.    
    feature map이 anchor box별로 4개의 좌표값(x, y, w, h)을 encode하도록 channel 수를 조정합니다.    
    최종적으로 channel 수가 4xA인 5개의 feature map을 얻을 수 있습니다. 
<br></br>

### Inference

# Result
## Experiments
**focal loss performance**   
![image](https://github.com/devkade/DeepSync/assets/11837072/29a7d775-262b-40f1-80cb-0f8d10f44533)<br/>
   
**RetinaNet performance**   
![image](https://github.com/devkade/DeepSync/assets/11837072/7c8223c5-c801-439d-acf7-08a03462aa04)<br/>

# Conclusion
+ focal loss를 통해서 다른 여타의 복잡한 sampling 없이 효과적으로 class imbalance 문제를 해결 함.
+ 하지만 focal loss algorithm의 특성 상 학습 진행됨에 따라   
hard positive sample이 점점 easy positive sample이 되어갈 것이고   
이에 따라 학습이 saturation될 수 있어보임

# Reference
+ https://herbwood.tistory.com/19
