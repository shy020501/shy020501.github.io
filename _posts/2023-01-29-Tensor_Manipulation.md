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
    

# Mean

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
    

# Max와 Argmax
* Max는 tensor에 대해서 가장 큰 값을 의미합니다.
* Argmax는 가장 큰 값의 index를 의미합니다.
* .max() 함수를 인자(argument) 없이 호출하면, max 값만 반환해줍니다.
* .max() 함수에 특정 차원을 인자로 넣어주면, max값과 argmax를 반환해줍니다.


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
    


```python

```
