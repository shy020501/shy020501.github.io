---
layout: single
title:  "[PyTorch] PyTorch 기본 연산"
categories: PyTorch
tag: [Python, PyTorch, Machine Learning]
toc: True
---

<br>

# PyTorch Tensor

## 1D Array 만들기


```python
import torch

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.]) # Numpy와 비슷하게, torch.FloatTensor 함수에 python list를 넣어주면 됨
print(t)
```

    tensor([0., 1., 2., 3., 4., 5., 6.])
    


```python
print(t.dim()) # 몇 차원 tensor인지 출력
print(t.shape) # Tensor의 형태 출력
print(t.size()) # Tensor의 형태 출력
print(t[0], t[2], t[-1]) # 특정 원소 출력
print(t[1:4], t[:2]) # Slicing 후 출력
```

    1
    torch.Size([7])
    torch.Size([7])
    tensor(0.) tensor(2.) tensor(6.)
    tensor([1., 2., 3.]) tensor([0., 1.])
    

## 2D Array 만들기


```python
t = torch.FloatTensor([[0., 1., 2.],
                       [3., 4., 5.],
                       [6., 7., 8.],
                       [9., 10., 11.]
                       ])

print(t)
```

    tensor([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]])
    


```python
print(t.dim())
print(t.shape) 
print(t.size()) 
print(t[0])
print(t[:, 1].size()) # 첫 번째 차원에서는 다 가져오고, 두 번째 차원에서는 1번 index에 위치한 값만 가져오기
print(t[:, :-1])# 첫 번째 차원에서는 다 가져오고, 두 번째 차원에서는 뒤에서 첫번째 index에 위치한 값만 빼고 가져오기
```

    2
    torch.Size([4, 3])
    torch.Size([4, 3])
    tensor([0., 1., 2.])
    torch.Size([4])
    tensor([[ 0.,  1.],
            [ 3.,  4.],
            [ 6.,  7.],
            [ 9., 10.]])
    

<br>

# Broadcasting
* 두 tensor의 크기가 다를 때, 자동적으로 크기를 조정해 연산을 도와주는 기능을 broadcast라고 합니다.
* 다만, 실수로 크기를 잘못 설정하였을 때도 broadcasting이 실행되면 오류가 뜨지 않고, 이상한 결과값이 나오는 상태로 프로그램이 실행될 수도 있으니 주의해야합니다.

## 같은 크기일 때


```python
t1 = torch.FloatTensor([3, 3])
t2 = torch.FloatTensor([2, 2])

print(t1 + t2)
```

    tensor([5., 5.])
    

## 백터와 스칼라의 합
* 원래대로라면, 둘은 곱하지 못합니다.
* 그러나 PyTorch가 스칼라(Scalar)를 자동으로 같은 크기의 행렬로 만들어 계산이 가능하게끔 만들어줍니다.


```python
t1 = torch.FloatTensor([1, 2])
t2 = torch.FloatTensor([3]) # [3] -> [[3, 3]]

print(t1 + t2)
```

    tensor([4., 5.])
    

## 크기가 다른 백터의 합


```python
t1 = torch.FloatTensor([1, 2]) # 2 X 1 백터
t2 = torch.FloatTensor([3], [4]) # 1 X 2 백터

print(t1 + t2) # 둘 다 2 X 2 백터로 만들어서 계산
```

## Multiplication vs Matrix Multiplication
* Muliplication(*, .mul): 아다마르 곱을 수행합니다(각 행렬의 원소끼리만 곱함).
* Matrix Multiplication(.matmul): 행렬 곱을 수행합니다.


```python
t1 = torch.FloatTensor([[1, 2],
                        [3, 4]])
t2 = torch.FloatTensor([[1], [2]])

print("Multiplication:")
print(t1 * t2) # t1과 t2의 아다마르 곱 (t2가 2 X 2 행렬로 broadcasting됨)
print(t1.mul(t2))

print("Matrix Multiplication:")
print(t1.matmul(t2)) # t1과 t2의 행렬 곱


```

    Multiplication:
    tensor([[1., 2.],
            [6., 8.]])
    tensor([[1., 2.],
            [6., 8.]])
    
    Matrix Multiplication:
    tensor([[ 5.],
            [11.]])
    

<br>

# 평균값

## 평균 구하기
* Long 형태의 tensor에서는 평균을 구할 수 없습니다.


```python
t = torch.FloatTensor([1, 2])
print(t.mean())
```

    tensor(1.5000)
    


```python
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)
```

    mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
    

## 원하는 차원에 대해서만
* 특정 차원에 대해서 평균을 구하고 싶다고 했을 때, 해당 차원을 없앤다고 생각하면 됩니다.
* 예를 들어 2 X 2 행렬이 있을 때, 첫 번째 차원에 대해서 평균을 구하면 1 X 2 행렬이 나옵니다.


```python
t = torch.FloatTensor([[1, 2], 
                       [3, 4]])

print(t)
print(t.mean()) # 4개의 원소에 대해서 평균을 구함
print(t.mean(dim = 0)) # 첫 번째 차원에 대한 평균을 구함 (2 X 2) -> (1 X 2)
print(t.mean(dim = 1)) # 두 번째 차원에 대한 평균을 구함 (2 X 2) -> (2 X 1)
print(t.mean(dim = -1))
```

    tensor([[1., 2.],
            [3., 4.]])
    tensor(2.5000)
    tensor([2., 3.])
    tensor([1.5000, 3.5000])
    tensor([1.5000, 3.5000])
    

### 합도 마찬가지


```python
t = torch.FloatTensor([[1, 2], 
                       [3, 4]])

print(t)
print(t.sum()) # 4개의 원소에 대해서 평균을 구함
print(t.sum(dim = 0)) # 첫 번째 차원에 대한 평균을 구함 (2 X 2) -> (1 X 2)
print(t.sum(dim = 1)) # 두 번째 차원에 대한 평균을 구함 (2 X 2) -> (2 X 1)
print(t.sum(dim = -1))
```

    tensor([[1., 2.],
            [3., 4.]])
    tensor(10.)
    tensor([4., 6.])
    tensor([3., 7.])
    tensor([3., 7.])
    

<br>

# Max와 Argmax
* Max는 tensor에 대해서 가장 큰 값을 의미합니다.
* Argmax는 가장 큰 값의 index를 의미합니다.
* max() 메서드를 인자(argument) 없이 호출하면, max 값만 반환해줍니다.
* max() 메서드에 특정 차원을 인자로 넣어주면, max값과 argmax를 반환해줍니다.


```python
t = torch.FloatTensor([[1, 2], 
                       [3, 4]])

print(t.max()) # Max 값만 return
```

    tensor(4.)
    


```python
t = torch.FloatTensor([[1, 2], 
                       [3, 4]])

print(t.max(dim = 0)) # Max 값과 argmax 값 return

print("Max: ", t.max(dim = 0)[0])
print("Argmax: ", t.max(dim = 0)[1])
```

    torch.return_types.max(
    values=tensor([3., 4.]),
    indices=tensor([1, 1]))
    
    Max:  tensor([3., 4.])
    Argmax:  tensor([1, 1])
    

<br>

# 그 외 연산들

## View
* view() 메서드는 tensor의 크기를 변경시켜주는(reshape) 역할을 합니다.
* -1은 보통 변동이 가장 심한 배치 사이즈에 사용할 수 있습니다.


```python
t = torch.FloatTensor([[[0, 1, 2], 
                        [3, 4, 5]], 
                      
                       [[6, 7, 8], 
                        [9, 10, 11]]
                      ]) # 2 X 2 X 3 행렬

print(t)
print(t.shape)
```

    tensor([[[ 0.,  1.,  2.],
             [ 3.,  4.,  5.]],
    
            [[ 6.,  7.,  8.],
             [ 9., 10., 11.]]])
    torch.Size([2, 2, 3])
    


```python
print(t.view([-1, 3])) # -1은 정확한 사이즈를 모르겠다는 뜻
print(t.view([-1, 3]).shape)
```

    tensor([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]])
    torch.Size([4, 3])
    


```python
print(t.view([-1, 1, 3]))
print(t.view([-1, 1, 3]).shape)
```

    tensor([[[ 0.,  1.,  2.]],
    
            [[ 3.,  4.,  5.]],
    
            [[ 6.,  7.,  8.]],
    
            [[ 9., 10., 11.]]])
    torch.Size([4, 1, 3])
    

## Squeeze
* 특정 차원에 1개의 값 밖에 없으면 해당 차원을 없애줍니다.


```python
t = torch.FloatTensor([[0], [1], [2]])

print(t)
print(t.shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    


```python
print(t.squeeze())
print(t.squeeze().shape)
```

    tensor([0., 1., 2.])
    torch.Size([3])
    


```python
print(t.squeeze(dim = 0)) # 첫 번째 차원은 3개의 값이 있기 때문에 squeeze해도 변하지 않습니다.
print(t.squeeze(dim = 0).shape)

print(t.squeeze(dim = 1)) # 두 번째 차원은 1개의 값 밖에 없기 때문에 squeeze히면 사라집니다.
print(t.squeeze(dim = 1).shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    
    tensor([0., 1., 2.])
    torch.Size([3])
    

## Unsqueeze
* Squeeze와 반대로 원하는 차원을 하나 추가해줍니다.
* 따라서, 호출할 때 인자를 꼭 넣어주어야 합니다.


```python
t = torch.FloatTensor([0, 1, 2]) # (3, ) tensor
print(t.shape)
```

    torch.Size([3])
    


```python
print(t.unsqueeze(0)) # (3, ) -> (1, 3) tensor
print(t.unsqueeze(0).shape)

# View로도 똑같은 기능 구현 가능
print(t.view(1, -1))
print(t.view(1, -1).shape)
```

    tensor([[0., 1., 2.]])
    torch.Size([1, 3])
    
    tensor([[0., 1., 2.]])
    torch.Size([1, 3])
    


```python
print(t.unsqueeze(1)) # (3, ) -> (3, 1) tensor
print(t.unsqueeze(1).shape)

print(t.unsqueeze(-1))
print(t.unsqueeze(-1).shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    
    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    

## Type Casting


```python
long_t = torch.LongTensor([1, 2, 3, 4])

print(long_t)
print(long_t.float())
```

    tensor([1, 2, 3, 4])
    tensor([1., 2., 3., 4.])
    


```python
byte_t = torch.ByteTensor([True, False, False, True])

print(byte_t)
print(byte_t.long())
print(byte_t.float())
```

    tensor([1, 0, 0, 1], dtype=torch.uint8)
    tensor([1, 0, 0, 1])
    tensor([1., 0., 0., 1.])
    

## Concatenate
* 특정 차원을 인자로 넘겨서 호출해주면 해당 차원을 늘려 두 개의 tensor을 합쳐 줍니다.


```python
t1 = torch.FloatTensor([[1, 2], [3, 4]])
t2 = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([t1, t2], dim = 0)) # 첫 번째 차원이 늘어남
print(torch.cat([t1, t2], dim = 1)) # 두 번째 차원이 늘어남
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.]])
    tensor([[1., 2., 5., 6.],
            [3., 4., 7., 8.]])
    

## Stacking
* Concatenate와 비슷한 기능을 하지만 더 편리합니다.
* Stack 쌓듯이 


```python
t1 = torch.FloatTensor([1,4])
t2 = torch.FloatTensor([2,5])
t3 = torch.FloatTensor([3,6])

print(torch.stack([t1, t2, t3]))
print(torch.stack([t1, t2, t3], dim = 1))
```

    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    tensor([[1., 2., 3.],
            [4., 5., 6.]])
    


```python
print(torch.cat([t1, t2, t3], dim = 0))
print(torch.cat([t1.unsqueeze(0), t2.unsqueeze(0), t3.unsqueeze(0)], dim = 0)) # Stack과 같은 기능
```

    tensor([1., 4., 2., 5., 3., 6.])
    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    


```python
t1 = torch.FloatTensor([[1, 2], [3, 4]])
t2 = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([t1, t2], dim = 0))

print(torch.stack([t1, t2]))
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.]])
    
    tensor([[[1., 2.],
             [3., 4.]],
    
            [[5., 6.],
             [7., 8.]]])
    

## Ones와 Zeros
* 각자 같은 크기의 1과 0으로 이루어진 tensor를 만들어줍니다.


```python
t = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(t)

print(torch.ones_like(t))
print(torch.zeros_like(t))
```

    tensor([[0., 1., 2.],
            [2., 1., 0.]])
    
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    

## In-place 연산
* 메모리에 새로 선언하지 않고, 정답값을 기존 tensor 값에 넣어줍니다.
* PyTorch에서 garbage collector가 효율적으로 잘 설계 돼있어서, in-place 연산을 수행해도 속도 면에서는 큰 이점이 없을 수도 있습니다.


```python
t = torch.FloatTensor([[1, 2], [3, 4]])

print(t.mul(2.))
print(t)

print(t.mul_(2.))
print(t)
```

    tensor([[2., 4.],
            [6., 8.]])
    tensor([[1., 2.],
            [3., 4.]])
    
    tensor([[2., 4.],
            [6., 8.]])
    tensor([[2., 4.],
            [6., 8.]])
    


<br>

출처 | "모두를 위한 딥러닝 시즌2", Deep Learning Zero To All, https://www.youtube.com/playlist?list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv
