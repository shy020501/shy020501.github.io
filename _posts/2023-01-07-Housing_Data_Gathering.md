---
layout: single
title:  "[ML] 1. 데이터 가져오기 - Housing Price"
categories: Machine_Learning
tag: [Python, Machine Learning, Numpy, Matplotlib, Pandas, SciKit-Learn]
toc: True
---

<br>

# 활용 데이터
* StatLib 저장소에 있는 캘리포니아 주택 가격 (California Housing Price) 데이터 세트를 사용하였습니다.
* 이 데이터 세트는 1990년 캘리포니아 인구조사 데이터를 기반으로 합니다.

![California Housing Price](../../images/2023-01-07-Housing_Data_Gathering/california_housing_price.png)

<br>

# 데이터 다운로드
* CSV(Comma-Separated Value) 파일인 housing.csv를 압축한 tar 파일(.tgz)을 내려받아 데이터 수집합니다.


```python
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing") # housing.tgz 파일을 저장할 다이렉토리 (datasets/housing)
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz" # housing.tgz 파일을 다운 받을 경로

def fetching_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True) # housing_path 다이렉토리 생성
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path) # housing_url에서 tgz_path로 housing.tgz 파일 다운
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path) # housing_path 다이렉토리에 housing.tgz 파일 압축 해제
    housing_tgz.close()
    
fetching_housing_data()
```
<br>

# 데이터 읽어오기

## Pandas로 csv 파일 읽기


```python
import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head() # 첫 다섯 행 출력
housing.info() # 데이터에 대한 간략한 설명

housing["ocean_proximity"].value_counts() # 유일하게 데이터 타입이 실수형이 아닌 ocean_proximity 필드의 카테고리 확인

housing.describe() # 숫자형 필드의 특성 요약
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Matplotlib으로 히스토그램 출력하기

* 표 분석:
  * 'Median income'은 달러가 아니라 대략 수만 달러를 나타냅니다(1.5 ≈ $15,000).
  * 스케일을 조정하고 상한이 15,0001, 하한이 0.4999가 되도록 만든 것입니다.
    * 머신러닝에서는 이렇듯 전처리 된 데이터를 다루는 경우가 흔하지만, 데이터가 어떻게 계산되었는 지는 이해해야합니다.
  * 'Housing median age'와 'median housing value' 역시 상한과 하한을 제한해두었습니다.
  * 이중 'median housing value'은 레이블로 사용되기 때문에 큰 문제가 될 수도 있습니다(가격이 상한이 넘지 않도록 학습될 수도 있기 때문).
  * 이를 해결할 수 있는 방법은 2가지가 있습니다:
    * 상한값 밖의 구역에 대한 정확한 레이블 구하기
    * 훈련 세트에서 상한값 밖의 데이터 제거
* 주의할 점:
  * 많은 히스토그램은 가운데에서 왼쪽보다 오른쪽으로 더 멀리 뻗어 있습니다.
  * 이러한 일부 머신러닝 알고리즘으로 하여금 패턴을 찾기 어렵게 만듭니다.
  * 테스트 세트를 들여다 보면 겉으로 드러나는 어떤 패턴에 속아 특정 머신러닝 모델을 선택할 수도 있습니다.
  * 이 세트로 일반화 오차를 추정하면 매우 낙천적으로 추정이 되어 론칭을 했을 때 에상한 성능이 나오지 않을 수도 있습니다(데이터 스누핑 편향).


```python
# 브라우저에서 바로 도표를 볼 수 있게 해줌
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins = 50, figsize = (20,15)) # bins: 가로축 구간 개수, figsize: 도표 크기
plt.show()
```


    
![Housing Histogram](../../images/2023-01-07-Housing_Data_Gathering/housing_histogram.png)
    

<br>

# 테스트 세트 만들기

## 무작위 셔플
* 해당 함수를 다시 돌리면 새로운 테스팅 세트가 생성됩니다.
* 이를 여러번 계속하면 결국 전체 데이터셋을 보게 됨으로, 이러한 상황은 지양해야합니다.
* 함수 설명:
  * np.random.permutation(): 무작위로 섞인 배열 생성하는 함수
  * .iloc(): 행/열을 숫자로 location을 나타내서 selecting/indexing하는 함수


```python
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) # Data의 index를 랜덤하게 shuffle
    test_set_size = int(len(data) * test_ratio)  
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))
```

    16512
    4128
    

## 샘플의 식별자 사용하기
* 샘플의 고유하고 변경 불가능한 식별자(있다고 가정하고)를 사용하여 테스트 세트에 보낼지 말지 설정할 수 있습니다.
* 예를 들어, 식별자의 해시값을 계산해 해시 최댓값의 20% 이하의 샘플만 테스트 세트에 보낼 수도 있습니다.
* 이러한 방법을 이용하면, 데이터 세트가 갱신되더라도 테스트 세트가 동일하게 유지되고 이전에 훈련 세트에 있던 샘플은 확인하지 않을 것입니다.
* Housing 데이터 세트에는 식별자 칼럼이 없기에 행의 index를 ID로 사용합니다.
* crc32:
  * crc32는 어떠한 파일이나 id에 대해 해쉬값을 만들어줍니다.
  * crc32는 8개의 16진수(Hexadecimal)로 표현됩니다(따라서, 약 42.9억 분의 1의 확률로 해쉬값이 같을 수는 있습니다).
* 함수/연산자들 설명:
  * '&' 연산자: 비트 값이 둘 다 1일 때만 1을 반환하는 연산자 (i.e. 1100 0110 & 0101 1110 => 0100 0110)
  * .apply(): 행 또는 열 또는 전체 셀에 원하는 연산을 지원
  * .loc(): label이나 조건표현으로 선택하는 함수
  * '~' 연산자: 비트를 반전시키는 비트 보수 연산자 (i.e. ~ 0111 1010 => 1000 0101)
  * .reset_index(): 인덱스를 열로 변환하는 함수


```python
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio)) # test_set_check를 통해 id에 대해 해쉬값 생성 (그 중 20%만 True 반환)
    return data.loc[~in_test_set], data.loc[in_test_set] # in_test_set에 속한 20%는 test_set에, 나머지 80%는 train_set에 들어감 (True가 반환된 index는 test_set에 들어가는 것)

housing_with_id = housing.reset_index() # 인덱스가 주어진 housing 데이터 세트를 housing_with_id 변수에 저장
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index") # id를 기반으로 80%는 train_set에, 20%는 test_set에 저장

print(len(train_set))
print(len(test_set))
```

    16512
    4128
    

## Sklearn 사용하기
* 난수 초기값을 지정해줄 수 있습니다(random_state).
* 행의 개수가 같은 여러 데이터 세트를 넘겨서 같은 인덱스를 기반으로 나눌 수 있습니다.


```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

print(len(train_set))
print(len(test_set))
```

    16512
    4128
    

## StratifiedShuffleSplit 사용하기

### 계층 나누기
* 무작위 샘플링 방식은 샘플링 편향이 생길 위험성이 있습니다.
* 모집단의 데이터 분포 비율을 유지하면서 샘플링을 하는 것을 계층적 샘플링이라고 합니다.
* 'Median income'이 'median house value'를 예측하는데 매우 중요하다고 가정하면, 테스트 세트가 전체 데이터 세트에 있는 여러 소득 카테고리의 비율을 잘 반영해야합니다.
* 이를 달성하기 위해, pd.cut() 함수를 사용하여 5개로 나누어진 소득 카테고리 특성을 만들 수 있습니다(카테고리 1은 median income에서 0~1.5, 카테고리 2는 median income에서 1.5~3, ...).


```python
housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels = [1,2,3,4,5])

housing["income_cat"].hist()
```

    
![Housing Histogram](../../images/2023-01-07-Housing_Data_Gathering/income_cat_histogram.png)
    


### StratifiedShuffleSplit 적용
* 계층적 샘플링과 랜덤 샘플링을 합친 것입니다.
* 함수 설명:
  * .split(): 훈련 세트와 테스트 세트로 분할하기 위한 인덱스를 생성해 반환하는 함수
  * .value_counts(): 지정된 열의 값에 대한 발생 횟수를 반환하는 함수


```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]): # income_cat 레이블을 기준으로 train_index와 test_index 반환
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
strat_test_set["income_cat"].value_counts() / len(strat_test_set) # 5개의 카테고리로 나눠진 income_cat 레이블을 기준으로 각 카테고리가 몇 번 발생하였나 세고, 전체 세트 사이즈로 나눠서 발생 빈도를 측정
```



    # 위에 있는 히스토그램의 분포와 일치하는 것을 확인 할 수 있음
    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114341
    1    0.039971
    Name: income_cat, dtype: float64



### Income_cat 삭제
* drop() 함수는 행 또는 열을 삭제합니다.
* 매개변수 axis의 값이 0일 때는 행, 1일 때는 열을 삭제합니다.
* 매개변수 inplace의 값을 True로 설정하면(기본값은 False), 호출된 데이터프레임 자체를 수정하고 아무런 값도 반환하지 않습니다.


```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)
```

<br>

출처 | 오렐리앙 제롱, 「핸즈온 머신러닝 2판」, 박해선, 한빛미디어(2020)