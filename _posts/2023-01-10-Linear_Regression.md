# 선형회귀 (Linear Regression)
* y = W * x + b (W는 weight, b는 bias입니다.)

[Image 1]


```python
import torch
from torch import optim
```

## 한 번만 학습
* 데이터 정의, Hypothesis 초기화, Optimiser 정의를 합니다.


```python
x_train = torch.FloatTensor([[1], [2], [3]]) # x값
y_train = torch.FloatTensor([[2], [4], [6]]) # y값

W = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

hypothesis = W * x_train + b # 현재의 weight와 bias를 가지고 낸 예측값

cost = torch.mean((hypothesis - y_train) ** 2) # Mean Squared Error(MSE)를 torch.mean을 이용해 구함

optimiser = optim.SGD([W, b], lr = 0.01) # [W, b]는 학습할 tensor들, lr은 learning rate

# 이 세개의 코드는 보통 함께 옴
optimiser.zero_grad() # gradient 초기화
cost.backward() # gradient 계산
optimiser.step() # gradient descent
```

## 전체 코드
* Hypothesis 예측을 하고, cost를 계산합니다.
* 그 후, optimiser을 통해 학습을 합니다.
* 그러면, W와 b가 각자 하나의 최적의 숫자로 수렴하게 됩니다 (Gradient Descent).


```python
x_train = torch.FloatTensor([[1], [2], [3]]) # x값
y_train = torch.FloatTensor([[2], [4], [6]]) # y값

W = torch.zeros(1, requires_grad = True) # require_grad: 해당 텐서에 대한 계산을 모두 tracking해서 기울기 구해주기
b = torch.zeros(1, requires_grad = True)

optimiser = optim.SGD([W, b], lr = 0.01)
nb_epochs = 1000 # 데이터로 학습한 횟수

for epoch in range(1, nb_epochs + 1):
    hypothesis = W * x_train + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    optimiser.zero_grad()
    cost.backward()
    optimiser.step()
```

# Gradient Descent (without Optim)
* y = W * x
* Weight 밖에 학습하지 못하기에 실제로 사용하기에는 무리가 있으나, Gradient Descent의 개념을 이해하기 위해 bias를 빼도록 하겠습니다.

[Image 2]

* 해당 예제에서 W가 1일 때 cost는 0이 되고, 1에서 멀어질 수록 높아집니다.
* 따라서, cost를 줄이기 위해서는 gradient가 양수일 때는 W를 줄이고, 음수일 때는 W를 늘리면 됩니다.
* W <- W - α∇W (∇W: Gradient)를 통해 cost를 줄일 수 있습니다. 이 과정을 Gradient Descent라고 부릅니다.


```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

cost = torch.mean((hypothesis - y_train) ** 2)

gradient = 2 * torch.mean((W * x_train - y_train) * x_train) # Gradient (∇W)
lr = 0.1 # Learning rate (α)
W = W - lr * gradient # W = W - α∇W
```

## 전체 코드
* Epoch가 늘어남에 따라(데이터로 학습한 횟수가 늘어남에 따라), Weight(W)가 점점 1에 수럼하고 Cost가 점점 줄어드는 것을 확인할 수 있습니다.


```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1)

lr = 0.1
nb_epochs = 10

for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W
    
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)
    
    print("Epoch {:4d}/{}    \tW: {:.3f}\tCost: {:.6f}".format(
        epoch, nb_epochs, W.item(), cost.item()
    ))
    
    W -= lr * gradient
```

    Epoch    0/10    	W: 0.000	Cost: 4.666667
    Epoch    1/10    	W: 1.400	Cost: 0.746666
    Epoch    2/10    	W: 0.840	Cost: 0.119467
    Epoch    3/10    	W: 1.064	Cost: 0.019115
    Epoch    4/10    	W: 0.974	Cost: 0.003058
    Epoch    5/10    	W: 1.010	Cost: 0.000489
    Epoch    6/10    	W: 0.996	Cost: 0.000078
    Epoch    7/10    	W: 1.002	Cost: 0.000013
    Epoch    8/10    	W: 0.999	Cost: 0.000002
    Epoch    9/10    	W: 1.000	Cost: 0.000000
    Epoch   10/10    	W: 1.000	Cost: 0.000000
    

## Optim 사용


```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1, requires_grad = True)

optimiser = optim.SGD([W], lr = 0.15)

nb_epochs = 10

for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W
    
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    print("Epoch {:4d}/{}    \tW: {:.3f}\tCost: {:.6f}".format(
        epoch, nb_epochs, W.item(), cost.item()
    ))
    
    # Cost로 hypothesis(H(x)) 개선
    optimiser.zero_grad()
    cost.backward()
    optimiser.step()
```

    Epoch    0/10    	W: 0.000	Cost: 4.666667
    Epoch    1/10    	W: 1.400	Cost: 0.746667
    Epoch    2/10    	W: 0.840	Cost: 0.119467
    Epoch    3/10    	W: 1.064	Cost: 0.019115
    Epoch    4/10    	W: 0.974	Cost: 0.003058
    Epoch    5/10    	W: 1.010	Cost: 0.000489
    Epoch    6/10    	W: 0.996	Cost: 0.000078
    Epoch    7/10    	W: 1.002	Cost: 0.000013
    Epoch    8/10    	W: 0.999	Cost: 0.000002
    Epoch    9/10    	W: 1.000	Cost: 0.000000
    Epoch   10/10    	W: 1.000	Cost: 0.000000
    

# 다중 선형 회귀 (Multivariate Linear Regression)
* 다중 선형 회귀는 다수의 x 값으로부터 y 값을 예측하는 것입니다.

[Image 3]

* H(x)를 계산하는 방법은 2가지가 있습니다:
  * 방법 1:
    * hypothesis = W1 * x1 + W2 * x2 + W3 * x3 + b
    * 이 방법은 간단하지만 x의 길이가 커지면 식이 너무 길어진다는 문제점이 있습니다.
  * 방법 2:
    * PyTorch에서 제공해주는 matmul() 기능 사용
    * hypothesis = x_train.matmul(W) + b
    * 이 방법은 더 간결하고, x의 길이가 바뀌어도 코드를 바꿀 필요가 없고, 속도도 더 빠릅니다.
* 그 외 cost 계산 방법과 학습 방법은 선형 회귀와 동일합니다.
* 프로그램을 돌려보면, cost가 점점 줄어들고 hypothesis가 y에 가까워지는 것을 확인할 수 있습니다.


```python
x_train = torch.FloatTensor([[73,  80,  75], 
                             [93,  88,  93], 
                             [89,  91,  80], 
                             [96,  98,  100],   
                             [73,  66,  70]])  
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

W = torch.zeros((3, 1), requires_grad=True) # x가 3개이기 때문에 3 X 1 tensor 생성
b = torch.zeros(1, requires_grad=True)

optimiser = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    hypothesis = x_train.matmul(W) + b # W1 * x1 + W2 * x2 + W3 * x3 + b와 같은 효과

    cost = torch.mean((hypothesis - y_train) ** 2)

    optimiser.zero_grad()
    cost.backward()
    optimiser.step()

    print('Epoch {:4d}/{}   hypothesis: {}   Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
```

    Epoch    0/20   hypothesis: tensor([0., 0., 0., 0., 0.])   Cost: 29661.800781
    Epoch    1/20   hypothesis: tensor([66.7178, 80.1701, 76.1025, 86.0194, 61.1565])   Cost: 9537.694336
    Epoch    2/20   hypothesis: tensor([104.5421, 125.6208, 119.2478, 134.7861,  95.8280])   Cost: 3069.590820
    Epoch    3/20   hypothesis: tensor([125.9858, 151.3882, 143.7087, 162.4333, 115.4844])   Cost: 990.670288
    Epoch    4/20   hypothesis: tensor([138.1429, 165.9963, 157.5768, 178.1071, 126.6283])   Cost: 322.481873
    Epoch    5/20   hypothesis: tensor([145.0350, 174.2780, 165.4395, 186.9928, 132.9461])   Cost: 107.717064
    Epoch    6/20   hypothesis: tensor([148.9423, 178.9730, 169.8976, 192.0301, 136.5279])   Cost: 38.687496
    Epoch    7/20   hypothesis: tensor([151.1574, 181.6346, 172.4254, 194.8856, 138.5585])   Cost: 16.499043
    Epoch    8/20   hypothesis: tensor([152.4131, 183.1435, 173.8590, 196.5043, 139.7097])   Cost: 9.365656
    Epoch    9/20   hypothesis: tensor([153.1250, 183.9988, 174.6723, 197.4217, 140.3625])   Cost: 7.071114
    Epoch   10/20   hypothesis: tensor([153.5285, 184.4835, 175.1338, 197.9415, 140.7325])   Cost: 6.331847
    Epoch   11/20   hypothesis: tensor([153.7572, 184.7582, 175.3958, 198.2360, 140.9424])   Cost: 6.092532
    Epoch   12/20   hypothesis: tensor([153.8868, 184.9138, 175.5449, 198.4026, 141.0613])   Cost: 6.013817
    Epoch   13/20   hypothesis: tensor([153.9602, 185.0019, 175.6299, 198.4969, 141.1288])   Cost: 5.986785
    Epoch   14/20   hypothesis: tensor([154.0017, 185.0517, 175.6785, 198.5500, 141.1671])   Cost: 5.976325
    Epoch   15/20   hypothesis: tensor([154.0252, 185.0798, 175.7065, 198.5800, 141.1888])   Cost: 5.971208
    Epoch   16/20   hypothesis: tensor([154.0385, 185.0956, 175.7229, 198.5966, 141.2012])   Cost: 5.967835
    Epoch   17/20   hypothesis: tensor([154.0459, 185.1045, 175.7326, 198.6059, 141.2082])   Cost: 5.964969
    Epoch   18/20   hypothesis: tensor([154.0501, 185.1094, 175.7386, 198.6108, 141.2122])   Cost: 5.962291
    Epoch   19/20   hypothesis: tensor([154.0524, 185.1120, 175.7424, 198.6134, 141.2145])   Cost: 5.959664
    Epoch   20/20   hypothesis: tensor([154.0536, 185.1134, 175.7451, 198.6145, 141.2158])   Cost: 5.957089
    

# PyTorch 모듈 활용

## nn.Module
* nn.Module을 상속(inheritance)해서 모델을 생성하고, nn.Linear에 입력 차원과 출력 차원을 알려주고 hypothesis를 어떻게 계산하는지 forward 함수에서 알려주면 됩니다.


```python
import torch.nn as nn

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 3이 입력차원, 1이 출력차원
    def forward(self, x):
        return self.linear(x)

# model = MultivariateLinearRegressionModel()
# hypothesis = model(x_train)
```

## F.mse_loss
* torch.nn.functional의 장점:
  * 다른 cost function으로 바꿀 때 편리합니다.
  * cost function을 계산하면서 생길 오류가 없어 디버깅하기 편리합니다.


```python
import torch.nn.functional as F

cost = F.mse_loss(hypothesis, y_train)
```

## 전체 코드


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x_train = torch.FloatTensor([[73,  80,  75], 
                             [93,  88,  93], 
                             [89,  91,  80], 
                             [96,  98,  100],   
                             [73,  66,  70]])  
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = MultivariateLinearRegressionModel()

optimiser = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20

for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train) # model(x_train)은 model.forward(x_train)와 동일함

    cost = F.mse_loss(hypothesis, y_train)

    optimiser.zero_grad()
    cost.backward()
    optimiser.step()

    print('Epoch {:4d}/{}   hypothesis: {}   Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
```

    Epoch    0/20   hypothesis: tensor([ -94.6430, -112.7596, -107.9365, -121.3467,  -85.9796])   Cost: 77016.921875
    Epoch    1/20   hypothesis: tensor([12.8703, 16.4313, 14.6992, 17.2704, 12.5715])   Cost: 24758.568359
    Epoch    2/20   hypothesis: tensor([73.8229, 89.6734, 84.2257, 95.8565, 68.4431])   Cost: 7962.174805
    Epoch    3/20   hypothesis: tensor([108.3788, 131.1964, 123.6430, 140.4091, 100.1184])   Cost: 2563.631348
    Epoch    4/20   hypothesis: tensor([127.9697, 154.7369, 145.9905, 165.6670, 118.0761])   Cost: 828.480347
    Epoch    5/20   hypothesis: tensor([139.0763, 168.0826, 158.6604, 179.9863, 128.2568])   Cost: 270.782867
    Epoch    6/20   hypothesis: tensor([145.3730, 175.6484, 165.8440, 188.1040, 134.0285])   Cost: 91.530701
    Epoch    7/20   hypothesis: tensor([148.9427, 179.9376, 169.9171, 192.7059, 137.3007])   Cost: 33.914764
    Epoch    8/20   hypothesis: tensor([150.9666, 182.3690, 172.2269, 195.3146, 139.1557])   Cost: 15.394239
    Epoch    9/20   hypothesis: tensor([152.1139, 183.7473, 173.5369, 196.7933, 140.2074])   Cost: 9.439394
    Epoch   10/20   hypothesis: tensor([152.7644, 184.5285, 174.2801, 197.6313, 140.8035])   Cost: 7.523253
    Epoch   11/20   hypothesis: tensor([153.1331, 184.9712, 174.7021, 198.1062, 141.1415])   Cost: 6.905225
    Epoch   12/20   hypothesis: tensor([153.3422, 185.2219, 174.9418, 198.3751, 141.3330])   Cost: 6.704421
    Epoch   13/20   hypothesis: tensor([153.4607, 185.3639, 175.0783, 198.5273, 141.4416])   Cost: 6.637683
    Epoch   14/20   hypothesis: tensor([153.5279, 185.4442, 175.1562, 198.6133, 141.5031])   Cost: 6.614072
    Epoch   15/20   hypothesis: tensor([153.5659, 185.4896, 175.2009, 198.6618, 141.5379])   Cost: 6.604333
    Epoch   16/20   hypothesis: tensor([153.5875, 185.5151, 175.2268, 198.6890, 141.5576])   Cost: 6.599013
    Epoch   17/20   hypothesis: tensor([153.5998, 185.5293, 175.2421, 198.7042, 141.5688])   Cost: 6.595149
    Epoch   18/20   hypothesis: tensor([153.6067, 185.5372, 175.2513, 198.7125, 141.5750])   Cost: 6.591750
    Epoch   19/20   hypothesis: tensor([153.6106, 185.5415, 175.2570, 198.7169, 141.5786])   Cost: 6.588500
    Epoch   20/20   hypothesis: tensor([153.6128, 185.5437, 175.2608, 198.7191, 141.5805])   Cost: 6.585258
    

# 많은 양의 데이터 분석

## Minibatch Gradient Descent
* 미니 배치 학습을 하게되면, 데이터를 미니 배치만큼만 가져가서 미니 배치에 대한 대한 cost를 계산하고, Gradient Descent를 수행합니다.
* 장점:
  * 업데이트를 조금 더 빨리 할 수 있습니다.
* 단점:
  * 전체 데이터를 쓰지 않아, 잘못된 방향으로 업데이트를 할 수 있습니다.

## PyTorch Dataset
* __len__(): 데이터셋의 총 데이터 수를 반환합니다.
* __getitem__(): 어떠한 index를 받았을 때, 그에 상응하는 입출력 데이터를 반환합니다.


```python
from torch.utils.data import Dataset

class CustomDataset(Dataset): # torch.utils.data.Dataset 상속 (inheritance)
    def __init__(self):
        self.x_data = [[73,  80,  75], 
                       [93,  88,  93], 
                       [89,  91,  80], 
                       [96,  98,  100],   
                       [73,  66,  70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        return x, y
    
dataset = CustomDataset()

# TensorDataset을 이용할 수도 있음
# from torch.utils.data import TensorDataset
# dataset = TensorDataset(x_train, y_train)
```

## PyTorch DataLoader
* batch_size:
  * 각 미니배치의 크기를 의미합니다.
  * 통상적으로 2의 제곱수로 설정합니다(16, 32, 64, 128, ...).
* shuffle:
  * Epoch마다 데이터 세트를 섞어서 데이터가 학습되는 순서를 바꿉니다.
  * 일반적으로, True로 설정하는걸 권장합니다.


```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True,
)
```

## 전체 코드
* enumerator(dataloader): 미니배치의 index와 데이터를 받습니다.
* len(dataloader): 한 epoch당 미니배치의 개수를 의미합니다.
* Epoch가 늘어남에 따라 cost가 점점 줄어드는 것을 확인할 수 있습니다.
* 훈련을 마친 후, [73, 80, 75]라는 임의의 값을 입력하면 152.6124라는 값을 예측하는 것을 볼 수 있습니다.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

dataset = CustomDataset()
dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True,
)

model = nn.Linear(3,1)
optimiser = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    x_train, y_train = samples
    
    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimiser.zero_grad()
    cost.backward()
    optimiser.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
```

    Epoch    0/20 Batch 1/3 Cost: 68962.632812
    Epoch    0/20 Batch 2/3 Cost: 26021.757812
    Epoch    0/20 Batch 3/3 Cost: 12828.721680
    Epoch    1/20 Batch 1/3 Cost: 1747.355225
    Epoch    1/20 Batch 2/3 Cost: 788.609619
    Epoch    1/20 Batch 3/3 Cost: 83.814842
    Epoch    2/20 Batch 1/3 Cost: 61.407188
    Epoch    2/20 Batch 2/3 Cost: 47.154518
    Epoch    2/20 Batch 3/3 Cost: 1.573085
    Epoch    3/20 Batch 1/3 Cost: 0.528568
    Epoch    3/20 Batch 2/3 Cost: 11.780762
    Epoch    3/20 Batch 3/3 Cost: 0.155980
    Epoch    4/20 Batch 1/3 Cost: 7.185496
    Epoch    4/20 Batch 2/3 Cost: 0.588306
    Epoch    4/20 Batch 3/3 Cost: 0.296059
    Epoch    5/20 Batch 1/3 Cost: 6.641660
    Epoch    5/20 Batch 2/3 Cost: 2.504662
    Epoch    5/20 Batch 3/3 Cost: 0.066988
    Epoch    6/20 Batch 1/3 Cost: 6.969504
    Epoch    6/20 Batch 2/3 Cost: 0.817236
    Epoch    6/20 Batch 3/3 Cost: 2.177309
    Epoch    7/20 Batch 1/3 Cost: 0.195941
    Epoch    7/20 Batch 2/3 Cost: 8.973341
    Epoch    7/20 Batch 3/3 Cost: 0.379227
    Epoch    8/20 Batch 1/3 Cost: 1.039827
    Epoch    8/20 Batch 2/3 Cost: 0.015303
    Epoch    8/20 Batch 3/3 Cost: 14.998128
    Epoch    9/20 Batch 1/3 Cost: 3.316312
    Epoch    9/20 Batch 2/3 Cost: 6.385233
    Epoch    9/20 Batch 3/3 Cost: 0.718314
    Epoch   10/20 Batch 1/3 Cost: 6.037750
    Epoch   10/20 Batch 2/3 Cost: 2.898014
    Epoch   10/20 Batch 3/3 Cost: 0.164086
    Epoch   11/20 Batch 1/3 Cost: 0.025057
    Epoch   11/20 Batch 2/3 Cost: 0.453884
    Epoch   11/20 Batch 3/3 Cost: 16.476767
    Epoch   12/20 Batch 1/3 Cost: 2.390849
    Epoch   12/20 Batch 2/3 Cost: 4.634770
    Epoch   12/20 Batch 3/3 Cost: 6.022240
    Epoch   13/20 Batch 1/3 Cost: 0.059228
    Epoch   13/20 Batch 2/3 Cost: 7.548706
    Epoch   13/20 Batch 3/3 Cost: 0.374786
    Epoch   14/20 Batch 1/3 Cost: 1.053912
    Epoch   14/20 Batch 2/3 Cost: 7.379338
    Epoch   14/20 Batch 3/3 Cost: 0.805108
    Epoch   15/20 Batch 1/3 Cost: 0.278761
    Epoch   15/20 Batch 2/3 Cost: 6.308988
    Epoch   15/20 Batch 3/3 Cost: 4.007633
    Epoch   16/20 Batch 1/3 Cost: 0.370973
    Epoch   16/20 Batch 2/3 Cost: 8.348041
    Epoch   16/20 Batch 3/3 Cost: 0.428824
    Epoch   17/20 Batch 1/3 Cost: 1.168038
    Epoch   17/20 Batch 2/3 Cost: 0.003512
    Epoch   17/20 Batch 3/3 Cost: 14.635855
    Epoch   18/20 Batch 1/3 Cost: 2.698539
    Epoch   18/20 Batch 2/3 Cost: 5.919295
    Epoch   18/20 Batch 3/3 Cost: 1.349824
    Epoch   19/20 Batch 1/3 Cost: 1.579122
    Epoch   19/20 Batch 2/3 Cost: 6.934173
    Epoch   19/20 Batch 3/3 Cost: 0.609065
    Epoch   20/20 Batch 1/3 Cost: 1.379900
    Epoch   20/20 Batch 2/3 Cost: 7.033577
    Epoch   20/20 Batch 3/3 Cost: 0.879879
    


```python
# 임의의 값을 넣어 예측값 확인
new_var =  torch.FloatTensor([[73, 80, 75]]) 
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
```

    훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[152.6124]], grad_fn=<AddmmBackward0>)
    


```python

```
