# 전 내용 요약
* 바뀐 점:
  * 예측 변수와 타겟값에 같은 변형을 적용시키지 않기 위해 둘을 분리해두었습니다.


```python
%matplotlib inline  

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
   
housing = strat_train_set.drop("median_house_value", axis = 1) # drop() 메서드는 데이터 복사본을 만들어 strat_train_set에 영향을 주지 않음
housing_labels = strat_train_set["median_house_value"].copy() # 타겟값만 따로 분리하여 복사
```

# 데이터 준비
* 머신러닝 알고리즘을 위해 데이터를 준비할 때, 해당 작업을 수동으로 하는 대신 함수로 자동화해야 하는 이유는 아래와 같습니다:
  * 어떤 데이터 세트에 대해서도 변환을 손쉽게 반복 가능
  * 향후 프로젝트에서 사용 가능한 변환 라이브러리를 점진적으로 구축 가능
  * 실제 시스템에서 모델(알고리즘)에 새 데이터를 주입시키기 전에 해당 함수 사용 가능
  * 여러 가지 데이터 변환 시도 및 최적의 조합 파악 가능

# 데이터 정제
* 앞서 total_bedrooms 특성에 값이 없는 경우를 확인헀었는데, 이를 해결하는 방법은 세 가지가 있습니다:
  * [옵션1] 해당 구역을 제거
  * [옵션2] 전체 특성을 삭제
  * [옵션3] 임의의 값으로 채우기 (0, 평균, 중간값 등)

## 데이터 프레임 메서드 사용


```python
median = housing["total_bedrooms"].median()

housing.dropna(subset = ["total_bedrooms"]) # 옵션 1
housing.drop("total_bedrooms", axis = 1) # 옵션 2
housing["total_bedrooms"].fillna(median, inplace = True) # 옵션 3
```

## SimpleImputer
* SimpleImputer는 누락된 값을 손 쉽게 다루게 해줍니다.
* 해당 예제의 경우, 누락된 값을 중간값(median)으로 대체하도록 하였습니다.
* Ocean_proximity를 제거한(drop) 이유는 중간값이 수치형 특성에서만 계산될 수 있기 때문에 텍스트형 특성을 제거한 복사본을 제작한 것입니다.
* 해당 예제에서는 'total_bedrooms' 특성에만 누락된 값이 있지만, 나중에 시스템이 서비스 될 때 데이터에서 어떤 값이 누락될지 확신할 수 없으므로 모든 수치형 특서에 imputer
* 메서드 설명:
  * fit(): 누락된 값에 대한 학습을 진행
  * transform(): 실체 누락된 값을 처리하고 변환된 데이터 세트를 반환
  * 둘을 연달아 호출하는 fit_transform() 메서드가 좀 더 효율적입니다.


```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median") # 중간값(median)으로 대체하도록 지정
housing_num = housing.drop("ocean_proximity", axis = 1) # 텍스트형 특성 제거한 복사본 생성
imputer.fit(housing_num) # 훈련 데이터에 적용 (imputer가 각 특성의 중간값을 계산에서 결과를 객체의 statistics_ 속성에 저장)
# print(imputer.statistics_) 각 특성의 중간값 출력

result = imputer.transform(housing_num) # 누락된 값을 중간값으로 대체해서 반환
```

* 이를 Pandas 데이터프레임을 이용해 다시 되돌릴 수도 있습니다.


```python
housing_tr = pd.DataFrame(result, columns = housing_num.columns, 
                          index = housing_num.index)
```

# 텍스트와 범주형 특성

## OrdinalEncoder
* 대부분 머신러닝 알고리즘은 숫자를 다루기 때문에 텍스트 특성은 숫자로 변환해야합니다.
* 이를 위해, 사이킷런의 OrdinalEncoder 클라스를 사용할 수 있습니다.
* 범주형 특성마다 카테고리들의 일차원 배열을 담은 리스트가 반환됩니다.
* categories_ 인스턴스 변수를 이용해 카테고리 목록을 확인할 수 있습니다.


```python
from sklearn.preprocessing import OrdinalEncoder

housing_cat = housing[["ocean_proximity"]] 
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(housing_cat.head(10))
print(housing_cat_encoded[:10]) # 카테고리 별로 숫자가 지정된 것을 확인할 수 있음

print(ordinal_encoder.categories_)
```

          ocean_proximity
    12655          INLAND
    15502      NEAR OCEAN
    2908           INLAND
    14053      NEAR OCEAN
    20496       <1H OCEAN
    1481         NEAR BAY
    18125       <1H OCEAN
    5830        <1H OCEAN
    17989       <1H OCEAN
    4861        <1H OCEAN
    [[1.]
     [4.]
     [1.]
     [4.]
     [0.]
     [3.]
     [0.]
     [0.]
     [0.]
     [0.]]
    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
          dtype=object)]
    

## 원-핫 인코딩
* 기존에 사용된 인코더의 표현 방식의 문제는 머신러닝 알고리즘이 가까이 있는 두 값이 떨어져 있는 두 값보다 더 비슷하다고 생각하는 것입니다.
  * 예를 들어, OrdinalEncoder 예제의 카테고리를 보면 실제로는 '<1H OCEAN' 카테고리와 'NEAR OCEAN' 카테고리가 훨씬 비슷하지만, 머신러닝 알고리즘은 '<1H OCEAN' 카테고리와 'INLAND' 카테고리가 더 비슷하다고 생각하게 됩니다.
  * 다만, 'bad', 'average', 'good', 'exellent' 같이 순서가 있는 카테고리의 경우는 문제가 없습니다. 
* 이 문제는 주로 카테고리별 이진 특성을 만들어 해결합니다.
  * 카테고리가 '<1H OCEAN'일 때 한 특성이 1이고(그 외에는 0), 카테고리가 'INLAND'일 때 다른 한 특성이 1이 되는 식입니다.
  * 한 특성만 1이고(핫) 나머지는 0이므로, 이를 원-핫 인코딩이라고 합니다.
 * 카테고리 특성이 담을 수 있는 카테고리 수가 많다면 원-핫 인코딩은 많은 수의 입력 특성을 만들 수 있습니다.
   * 이는 훈련을 느리게하고 성능을 감소시킬 수 있습니다.
   * 이런 현상이 발생하면  범주형 입력값을 숫자형 특성으로 바꾸거나(예를 들어 ocean_proximity 특성을 해안까지의 거리로), 임베딩(embedding)이라 부르는 학습 가능한 저차원 벡터로 바꿀 수 있습니다.


```python
from sklearn.preprocessing import OneHotEncoder

housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

print(housing_cat.head(10))
print(housing_cat_1hot.toarray()[:10]) # OrdinalEncoder 예제에서 각 카테고리별로 지정된 숫자의 index만 1이 된 것을 확인할 수 있음
```

          ocean_proximity
    12655          INLAND
    15502      NEAR OCEAN
    2908           INLAND
    14053      NEAR OCEAN
    20496       <1H OCEAN
    1481         NEAR BAY
    18125       <1H OCEAN
    5830        <1H OCEAN
    17989       <1H OCEAN
    4861        <1H OCEAN
    [[0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0.]
     [1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0.]]
    

# 나만의 변환기
* 특별한 정제 작업이나 어떤 특성을 조합하는 등의 작업을 위해서는 사이킷런에서 제공하는 변환기가 아닌 나만의 변환기를 만들 필요가 있습니다.
* 사이킷런은 덕 타이핑을 지원하므로 fit(), transform(), fit_transform 메서드를 구현한 파이썬 클라스를 만들면 나만의 변환기를 (파이프라인 같은) 사이킷런의 기능과 매끄럽게 연동할 수 있습니다.
  * 덕 타이핑은 상속이나 인터페이스 구현이 아니라 객체의 속성이나 메서드가 객체의 유형을 결정하는 방식입니다.
  * TransformerMixin을 상속하면 fit_transform 메서드가 자동으로 생성됩니다.
    * 파이썬에서 이름에 Mixin이 있으면, 객체의 기능을 확정하려는 목적으로 만들어진 클라스입니다.
    * TransformerMixin은 fit_transform() 메서드 하나를 가지고 있으며, 이를 상속하는 모든 파이썬 클라스에 이 메서드를 제공합니다.
  * BaseEstimator를 상속하면, 하이퍼파라미터 튜닝에 필요한 두 메서드 get_params()와 set_params()를 추가로 얻게 됩니다.
    * get_params()와 set_params()는 사이킷런의 파이프라인과 그리드 탐색에 꼭 필요한 메서드이기에 모든 추정기와 변환기는 BaseEstimator를 상속해야합니다.
    * 이 두 메서드는 생성자에 명시된 매개면수만을 참조하기에 *args나 **kargs는 사용할 수 없습니다.
  


```python
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # *args나 **kargs가 아니어야함
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X):
        return self  # 더 할 일이 없습니다.
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix] # [:, n]: 모든 행에서 n번째 열의 정보를 가져다 달라는 뜻 (세로줄)
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if(self.add_bedrooms_per_room):
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room] # numpy.c_: 두개 이상의 배열을 가로로 연결
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

# 특성 스케일링
* 특성 스케일링은 데이터에 적용할 가장 중요한 변환들 중 하나입니다.
* 머신러닝 알고리즘들은 대부분 (트리 기반 알고리즘을 제외하고는) 입력한 숫자 특성들의 스케일이 많이 다르면 잘 작동하지 않습니다.
* 모든 특성의 범위를 같도록 만들어주는 방법은 주로 2가지가 사용됩니다:
  * min-max 스케일링:
    * 0~1 범위 안에 들도록 값을 이동하고 스케일을 조정합니다.
    * 사이킷런에서 제공하는 MinMaxScaler 변환기를 사용할 수 있습니다.
    * 범위를 바꾸고 싶다면 feature_range 매개번수로 범위를 변경할 수 있습니다.
    * 종종 정규화(normalisation)라고 불러지기도 하지만 이는 여러 의미로 다양하게 사용됨으로 혼동해서는 안됩니다.
  * 표준화(standardisation):
    * 각 값에서 평균을 뺀 후, 표준편차로 나누어 결과 분포의 분산이 1이 되도록합니다.
    * 사이킷런에서 제공하는 StandardScalar 변환기를 사용할 수 있습니다.
    * min-max 스케일링과 달리 상/하한이 정해져있지 않아 특정 알고리즘에서는 문제가 될 수도 있습니다(i.e. 0 ~ 1 사이의 입력값을 기대하는 신경망).
    * 이상치에 영향을 덜 받습니다
    * 예를 들어 0 ~ 15 사이에 위치한 값들 사이에 실수로 100이라는 이상치가 입력되면 max-min 스케일링에서는 0 ~ 15 사이의 값들이 0 ~ 0.15로 만들어지나, 표준화는 크게 영향받지 않습니다.

## 변환 파이프라인 (Pipeline)
* 사이킷런에는 연속된 변환을 순서대로 처리하기 위한 PIpeline 클라스가 있습니다.
  * 파이프라인(pipeline)은 한 데이터 처리의 출력이 다음 단계의 입력으로 이어지는 형태로 연결된 구조를 의미합니다.
* Pipeline은 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력으로 받습니다.
* 마지막 단계에서는 변환기와 추정기를 모두 사용할 수 있고, 그 외에는 모두 변환기여야합니다.
  * 즉, fit_transfer() 메서드나 fit() 메서드, transfer() 메서드를 가지고 있어야합니다.
  * 예제에서 마지막 추정기는 StandardScaler 변환기로 transfer() 메서드를 가지고 있습니다.


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy = "median")), 
                         ('attribs_adder', CombinedAttributesAdder()), 
                         ('std_scalar', StandardScaler())
                        ])

housing_num_tr = num_pipeline.fit_transform(housing_num) # housing_num은 수치형 열만 가지고 있는 복사본
```

## ColumnTransformer
* ColumnTransfer 클라스를 통해 하나의 변환기로 각 열마다 적절한 변환을 적용하여 모든 열을 처리할 수 있습니다(수치형 열이던, 범주형 열이던).
* 해당 예제에서 수치형 열은 앞서 정의한 num_pipeline을, 범주형 열은 OneHotEncoder을 사용해 변환됩니다.
* 여기서, num_pipeline은 희소 행렬을, OneHotEncoder()은 밀집 행렬을 반환합니다.
  * 희소 행렬(Sparse Matrix): 대부분의 값이 0으로 채워진 행렬
  * 밀집 행렬(Dense Matrix): 대부분의 값이 0이 아닌 값으로 채워진 행렬
* 이렇듯, 희소 행렬과 밀집 행렬이 섞여 있을 때, ColumnTransformer는 최종 행렬의 밀집도(0이 아닌 원소의 비율)를 추정합니다.
  * 밀집도가 임곗값(기본적으로는 0.3)보다 낮으면 희소 행렬, 그렇지 않으면 밀집 행렬을 반환합니다.
  * 해당 예제에서는 밀집 행렬이 반환됩니다.


```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), 
                                   ("cat", OneHotEncoder(), cat_attribs)
                                  ])

housing_prepared = full_pipeline.fit_transform(housing)
```
