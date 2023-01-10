---
layout: single
title:  "[PyTorch] 로지스틱 회귀"
categories: PyTorch
tag: [Python, PyTorch, Machine Learning]
toc: True
---

<br>

# 로지스틱 회귀 (Logistic Regression)
* 로지스틱 회귀는 이진분류(binary classification) 문제를 풀기 위한 알고리즘입니다.
* 시그모이드(σ(x)) 함수는 0과 1중 하나의 값(binary)을 반환합니다.
* 이를 이용하여, 주어진 x값과 Weight(W)를 이용해 H(x)값을 구할 수 있습니다.

![시그모이드 함수](../../images/2023-01-10-Logistic_Regression/Singmoid.png)

* H(x) = σ(x * W + b)
* (m,d) 크기의 행렬 x와 (d, 1) 크기의 행렬 W가 있을 때:
  * |x * W| = (m, d) * (d, 1) = (m, 1)
  * 즉, 주어진 x에 W(weight)를 곱해주면 binary 정보가 담긴 (m,1) 행렬 y를 찾을 수 있습니다.

![로지스틱 회귀](../../images/2023-01-10-Logistic_Regression/Logistic_regression.png)

* 정리하자면, H(x) = P(x=1;W) = 1 - P(x=0;W)입니다.
* 즉, 로지스틱 회귀에서 H(x)는 W라는 weight가 주어졌을 때, x가 1일 확률을 의미한다고 할 수 있습니다.

## Imports


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

## Training Data
* x_data는 (6, 2) 행렬로, m = 6, d = 2임을 알 수 있습니다.
* y_data는 (6, ) 행렬로, m = 6, d = 1임을 알 수 있습니다.


```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```

## H(x) 계산
* (6, 2) * W = (6, 1)가 되야하기 때문에 W는 (2, 1) 행렬인 것을 알 수 있습니다.
* W와 b가 0으로 초기화되어 있기 때문에 H(x) 값이 σ(0), 즉 1 / (1 + e^0)이 되어 전부 0.5로 초기화되어 있는 것을 볼 수 있습니다.


```python
W = torch.zeros((2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b))) # H(x) = σ(x * W + b)

print(hypothesis)
print(hypothesis.shape)

# torch.sigmoid 함수로 대체 가능
# hypothesis = torch.sigmoid(x_train.matmul(W) + b)
```

    tensor([[0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000]], grad_fn=<MulBackward0>)
    torch.Size([6, 1])
    

## Cost Function 계산
* 로지스틱 회귀의 cost function은 아래와 같습니다.

![Cost Function](../../images/2023-01-10-Logistic_Regression/Cost_function.png)

### 하나의 원소에 대한 Cost Function
* y_train[0]이 1이면 (1 - y_train[0])은 0, y_train[0]이 0이면 (1 - y_train[0])은 1이 되기 때문에 윗줄과 아랫줄 둘 중 하나만 살아남게 됩니다.
* torch.log(hypothesis[0])은 log P(x=1;W), 1 - hypothesis[0])은 log P(x=0;W) = 1 - log P(x=1;W)를 나타냅니다.


```python
loss = -(y_train[0] * torch.log(hypothesis[0]) + 
         (1 - y_train[0]) * torch.log(1 - hypothesis[0]))
print(loss)
```

    tensor([0.6931], grad_fn=<NegBackward0>)
    

### 전체 샘플에 대한 Cost Function


```python
losses = -(y_train * torch.log(hypothesis) + 
         (1 - y_train) * torch.log(1 - hypothesis))
print(losses)

cost = losses.mean()
print(cost)
```

    tensor([[0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931]], grad_fn=<NegBackward0>)
    
    tensor(0.6931, grad_fn=<MeanBackward0>)
    

### Cross Entropy를 이용한 코드


```python
F.binary_cross_entropy(hypothesis, y_train)
```




    tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward0>)



## 전체 코드


```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimiser = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimiser.zero_grad()
    cost.backward()
    optimiser.step()

    if epoch % 100 == 0: # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    Epoch    0/1000 Cost: 0.693147
    Epoch  100/1000 Cost: 0.134722
    Epoch  200/1000 Cost: 0.080643
    Epoch  300/1000 Cost: 0.057900
    Epoch  400/1000 Cost: 0.045300
    Epoch  500/1000 Cost: 0.037261
    Epoch  600/1000 Cost: 0.031673
    Epoch  700/1000 Cost: 0.027556
    Epoch  800/1000 Cost: 0.024394
    Epoch  900/1000 Cost: 0.021888
    Epoch 1000/1000 Cost: 0.019852
    

## 모델 평가
* 학습할 때 사용했던 x값을 이용해 모델을 평가해보았습니다.
* 각 항이 P(x=1)을 의미합니다(x가 1일 확률).
  * 예를 들어, hypthesis[0]은 x가 1일 확률이 0.00027648라는 것을 의미합니다.
* 최종적으로, 구해진 확률을 바탕으로 0.5 이상이면 1, 미만이면 0으로 정합니다 (prediction).
* 실제 정답(y_train)과 일치하는 것을 보아, 해당 모델은 잘 학습되었다는 것을 확인할 수 있습니다.


```python
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
```

    tensor([[2.7648e-04],
            [3.1608e-02],
            [3.8977e-02],
            [9.5622e-01],
            [9.9823e-01],
            [9.9969e-01]], grad_fn=<SigmoidBackward0>)
    


```python
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction.type(torch.uint8))

print("실제 정답:")
print(y_train)
```

    tensor([[0],
            [0],
            [0],
            [1],
            [1],
            [1]], dtype=torch.uint8)
    
    실제 정답:
    tensor([[0.],
            [0.],
            [0.],
            [1.],
            [1.],
            [1.]])
    

<br>

# nn.Module로 구현하는 로지스틱 회귀
* nn.Linear을 통해 W는 (2, 1), b는 (1, 1) 행렬로 설정하였습니다.
* m은 알 수 없지만 d는 2로 설정된 것을 알 수 있습니다.


```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1) # self.linear = {W, b}
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
# model = BinaryClassifier()
```

## 전체 코드


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = BinaryClassifier()

optimiser = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000

for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimiser.zero_grad()
    cost.backward()
    optimiser.step()

    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train # 실제값과 prediction이 일치하면 True
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
```

    Epoch    0/1000 Cost: 0.539713 Accuracy 83.33%
    Epoch   10/1000 Cost: 0.614851 Accuracy 66.67%
    Epoch   20/1000 Cost: 0.441875 Accuracy 66.67%
    Epoch   30/1000 Cost: 0.373145 Accuracy 83.33%
    Epoch   40/1000 Cost: 0.316358 Accuracy 83.33%
    Epoch   50/1000 Cost: 0.266094 Accuracy 83.33%
    Epoch   60/1000 Cost: 0.220498 Accuracy 100.00%
    Epoch   70/1000 Cost: 0.182095 Accuracy 100.00%
    ... 중략 ...
    Epoch  980/1000 Cost: 0.020219 Accuracy 100.00%
    Epoch  990/1000 Cost: 0.020029 Accuracy 100.00%
    Epoch 1000/1000 Cost: 0.019843 Accuracy 100.00%

<br>

출처 | "모두를 위한 딥러닝 시즌2", Deep Learning Zero To All, https://www.youtube.com/playlist?list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv
