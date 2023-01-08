# 전 내용 요약


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
```

# 지리적 데이터 시각화

## 데이터 시각화


```python
housing = strat_train_set.copy() # 훈련 세트를 손상시키지 않기 위해 복사
housing.plot(kind = "scatter", x = "longitude", y = "latitude") # 위도, 경도를 이용하여 산점도 제작
```




    <AxesSubplot: xlabel='longitude', ylabel='latitude'>




    
![png](output_4_1.png)
    


## 밀집도 추가


```python
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)
```




    <AxesSubplot: xlabel='longitude', ylabel='latitude'>




    
![png](output_6_1.png)
    


## 주택 가격 추가
* s: 원의 반지름 (구역의 인구)
* c: 색상 (가격)


```python
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1,
            s = housing["population"]/100, label = "population", figsize = (10,7),
            c = "median_house_value", colormap = "jet", colorbar = True,
            )
plt.legend()
```




    <matplotlib.legend.Legend at 0x28854603f40>




    
![png](output_8_1.png)
    


# 데이터 상관관계 조사

## 표준 상관계수
* 데이터 세트가 너무 크지 않기 때문에 corr() 함수를 이용해 특성들 간의 표준 상관계수(standard correlation coefficient)를 계산할 수 있습니다.
* 1에 가까울 수록 강한 양의 상관관계, -1에 가까울 수록 강한 음의 상관관계를 나타냅니다.


```python
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending = False)
```

    C:\Users\shy02\AppData\Local\Temp\ipykernel_54800\3968440067.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      corr_matrix = housing.corr()
    




    median_house_value    1.000000
    median_income         0.687151
    total_rooms           0.135140
    housing_median_age    0.114146
    households            0.064590
    total_bedrooms        0.047781
    population           -0.026882
    longitude            -0.047466
    latitude             -0.142673
    Name: median_house_value, dtype: float64



## 산점도 (scatter_matrix) 사용

### 산점도
* 존재하는 숫자형 특성 11개에 대한 산점도, 즉 총 121개의 그래프를 전부 그릴 수 없음으로, 중간 주택 가격(median house value)와 상관관계가 높아 보이는 특성 몇 개만 사용하였습니다.
* 대각선(좌측 상단에서 우측 하단)은 자기 자신에 대한 그래프임으로 직선이 되어 유용하지 않음으로, 각 특성의 히스토그램으로 대체 되었습니다.


```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize = (12,8))
```




    array([[<AxesSubplot: xlabel='median_house_value', ylabel='median_house_value'>,
            <AxesSubplot: xlabel='median_income', ylabel='median_house_value'>,
            <AxesSubplot: xlabel='total_rooms', ylabel='median_house_value'>,
            <AxesSubplot: xlabel='housing_median_age', ylabel='median_house_value'>],
           [<AxesSubplot: xlabel='median_house_value', ylabel='median_income'>,
            <AxesSubplot: xlabel='median_income', ylabel='median_income'>,
            <AxesSubplot: xlabel='total_rooms', ylabel='median_income'>,
            <AxesSubplot: xlabel='housing_median_age', ylabel='median_income'>],
           [<AxesSubplot: xlabel='median_house_value', ylabel='total_rooms'>,
            <AxesSubplot: xlabel='median_income', ylabel='total_rooms'>,
            <AxesSubplot: xlabel='total_rooms', ylabel='total_rooms'>,
            <AxesSubplot: xlabel='housing_median_age', ylabel='total_rooms'>],
           [<AxesSubplot: xlabel='median_house_value', ylabel='housing_median_age'>,
            <AxesSubplot: xlabel='median_income', ylabel='housing_median_age'>,
            <AxesSubplot: xlabel='total_rooms', ylabel='housing_median_age'>,
            <AxesSubplot: xlabel='housing_median_age', ylabel='housing_median_age'>]],
          dtype=object)




    
![png](output_14_1.png)
    


### 중간 소득 (median income) 산점도
* 두 특성은 상관관계가 매우 강합니다(위쪽으로 향하는 경향이 있으며, 각 점들이 너무 멀리 퍼져있지 않습니다).
* 앞서 본 상한가 제한(1. 데이터 가져오기 참고)이 $500,000에서 수평선으로 잘 드러나고 있습니다.
* 그 외에도 $280,000, $350,000, $450,000 근처에서 수평선이 보입니다.
  * 알고리즘이 데이터에서 이런 이상한 형태를 학습하지 않도록 해당 구역을 제거하는 것이 좋습니다.


```python
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)
```




    <AxesSubplot: xlabel='median_income', ylabel='median_house_value'>




    
![png](output_16_1.png)
    


# 특성 조합
* 머신러닝 알고리즘 용 데이터를 준비하기 위해 마지막으로 할 수 있는 것은 여러가지 특성들을 조합해보는 것입니다.
* 예를 들어, 특정 구역의 방 개수는 해당 지역의 가구 수를 모르면 그다지 유용하지 않습니다. 따라서, 가구 수와 방 개수 특성을 조합해보면 가구당 방 개수라는 유용한 데이터를 도출해낼 수 있습니다.
* 이 탐색 단계는 완벽하지 않습니다. 따라서, 결과를 관찰하고 분석하여 더 많은 통찰을 얻고 다시 이 탐색 단계로 돌아오는 반복적인 과정을 거쳐야합니다.


```python
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"] # 가구 당 방개수
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"] # 방 당 침대수 (침대/방 비율)
housing["population_per_household"] = housing["population"] / housing["households"] # 가구당 인원수

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)
```

    C:\Users\shy02\AppData\Local\Temp\ipykernel_54800\3891066511.py:5: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      corr_matrix = housing.corr()
    




    median_house_value          1.000000
    median_income               0.687151
    rooms_per_household         0.146255
    total_rooms                 0.135140
    housing_median_age          0.114146
    households                  0.064590
    total_bedrooms              0.047781
    population_per_household   -0.021991
    population                 -0.026882
    longitude                  -0.047466
    latitude                   -0.142673
    bedrooms_per_room          -0.259952
    Name: median_house_value, dtype: float64




```python

```
