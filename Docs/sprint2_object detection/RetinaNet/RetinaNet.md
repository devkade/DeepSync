# Overview
## Objective
**Positive / Negative example의 class imbalance problem을 해결하는 방법을 이해한다**
+ <img src="figs\class imbalance.webp" title="class imbalance" alt="class imbalance"></img><br/>

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
<img src="figs\precision-recall.jpg"></img><br/>
    + **precision**
        + <img src="figs\precision.png"></img><br/>

    + **recall**   
        + <img src="figs\recall.png"></img><br/>

+ precision recall graph   
<img src="figs\precision-recall_graph.png"></img><br/>

+ average precision loss(AP-loss)   
<img src="figs\ap_graph.png"></img><br/>

<br></br>
### Type of object detection
**two-stage object detection**   
<img src="figs\twostage-obejct-detection.png" title="twostage-object-detection" alt="twostage-object-detection"></img><br/>

**one-stage object detection**   
<img src="figs\onestage-obejct-detection.png" title="onestage-object-detection" alt="onestage-object-detection"></img><br/>
+ image당 10,000 ~ 100,000개의 candidate 존재함
+ 하지만 이중 target object을 포함한 candidate는 매우 적음
+ background는 easy negative sample이 되고 foreground는 hard positive sample이 됨

***
<br></br>
# Main
## Overview
<img src="figs\RetinaNet.png" title="onestage-object-detection" alt="onestage-object-detection"></img><br/>

### Loss design
+ **cross entropy loss**   
<img src="figs\CE.png" title="cross entropy" alt="cross entropy"></img><br/>

+ **balanced cross entropy loss**   
<img src="figs\BCE.png" title="balanced cross entropy" alt="balanced cross entropy"></img><br/>
    + a weighting factor α∈[0,1] for class 1 and 1−α for class−1

+ **focal loss**   
<img src="figs\focal loss.png" title="focal loss" alt="focal loss"></img><br/>
    + For instance, with γ=2, an example classified with pt=0.9 would have 100× lower loss compared with CE and    
    with pt≈0.968 it would have 1000× lower loss
    + **balanced focal loss**   
    <img src="figs\Balanced focal loss.png" title="focal loss" alt="focal loss"></img><br/>

### Training
1. Feature Pyramid by ResNet + FPN
<img src="figs\RetinaNet_1.png" title="onestage-object-detection" alt="onestage-object-detection"></img><br/>
+ Input : image
+ Process : feature extraction by ResNet + FPN
+ Output : feature pyramid(P5~P7)
    + 먼저 이미지를 backbone network에 입력하여 서로 다른 5개의 scale을 가진 feature pyramid를 출력합니다.    
    여기서 backbone network는 ResNet 기반의 FPN(Feature Pyramid Network)를 사용합니다. pyramid level은 P3~P7로 설정합니다.
<br></br>
2. Classification by Classification subnetwork
<img src="figs\RetinaNet_2.png" title="onestage-object-detection" alt="onestage-object-detection"></img><br/>
+ Input : feature pyramid(P5~P7)
+ Process : classification by classification subnetwork
+ Output : 5 feature maps with KxA channel 
    + 1)번 과정에서 얻은 각 pyramid level별 feature map을 Classification subnetwork에 입력합니다.   
    해당 subnet는 3x3(xC) conv layer - ReLU - 3x3(xKxA) conv layer로 구성되어 있습니다.   
    여기서 K는 분류하고자 하는 class의 수를, A는 anchor box의 수를 의미합니다. 논문에서는 A=9로 설정합니다.    
    그리고 마지막으로 얻은 feature map의 각 spatial location(feature map의 cell)마다 sigmoid activation function을 적용합니다.    
    이를 통해 channel 수가 KxA인 5개(feature pyramid의 수)의 feature map을 얻을 수 있습니다. 
<br></br>
3. Bounding box regression by Bounding box regression subnetwork
<img src="figs\RetinaNet_3.png" title="onestage-object-detection" alt="onestage-object-detection"></img><br/>
+ Input : feature pyramid(P5~P7)
+ Process : bounding box regression by bounding box regression subnet
+ Output : 5 feature maps with 4xA channel
    + 1)번 과정에서 얻은 각 pyramid level별 feature map을 Bounding box regression subnetwork에 입력합니다.    
    해당 subnet 역시 classification subnet과 마찬가지로 FCN(Fully Convolutional Network)입니다.    
    feature map이 anchor box별로 4개의 좌표값(x, y, w, h)을 encode하도록 channel 수를 조정합니다.    
    최종적으로 channel 수가 4xA인 5개의 feature map을 얻을 수 있습니다. 
<br></br>

### Inference

# Result
## Experiments
**focal loss performance**   
<img src="figs\Focal loss performance per focusing parameter.png" title="onestage-object-detection" alt="onestage-object-detection"></img><br/>
   
**RetinaNet performance**   
<img src="figs\RetinaNet_performance.png" title="onestage-object-detection" alt="onestage-object-detection"></img><br/>

# Conclusion
+ focal loss를 통해서 다른 여타의 복잡한 sampling 없이 효과적으로 class imbalance 문제를 해결 함.
+ 하지만 focal loss algorithm의 특성 상 학습 진행됨에 따라   
hard positive sample이 점점 easy positive sample이 되어갈 것이고   
이에 따라 학습이 saturation될 수 있어보임

# Reference
+ https://herbwood.tistory.com/19