---
layout: post
title: "Random Bitcoin Analysis"
categories:
  - Investments
tags:
  - bitcoin
  - futures
  - deribit
  - python
  - jupyter
  - arbitrage
---


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('downloads/BTC-PERPETUAL.txt')
```


```python
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.index = df['date']
df = df.drop(columns=['timestamp', 'date', 'instrument_name'])
df.columns = ['perpetual', 'index']
```


```python
df.head()
```




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
      <th>timestamp</th>
      <th>instrument_name</th>
      <th>price</th>
      <th>index_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1534242862267</td>
      <td>BTC-PERPETUAL</td>
      <td>6035.0</td>
      <td>6034.91</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1534243101608</td>
      <td>BTC-PERPETUAL</td>
      <td>6043.0</td>
      <td>6037.37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1534243156847</td>
      <td>BTC-PERPETUAL</td>
      <td>6035.5</td>
      <td>6038.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1534243161625</td>
      <td>BTC-PERPETUAL</td>
      <td>6043.0</td>
      <td>6038.58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1534243288687</td>
      <td>BTC-PERPETUAL</td>
      <td>6035.0</td>
      <td>6031.53</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = df.iloc[0:20, 2]
df1
```




    0     6035.0
    1     6043.0
    2     6035.5
    3     6043.0
    4     6035.0
    5     6030.0
    6     6025.5
    7     6028.0
    8     6026.0
    9     6026.0
    10    6026.0
    11    6026.0
    12    6025.0
    13    6015.5
    14    6013.0
    15    6013.0
    16    6015.5
    17    6017.0
    18    6021.0
    19    6027.0
    Name: price, dtype: float64




```python
sns.kdeplot(df1, label='1')
sns.kdeplot(df1.resample('S').mean().interpolate(), label='2')
sns.kdeplot(df1.resample('1Min').mean(), label='3')
sns.kdeplot(df1.resample('1Min').mean().interpolate(), label='4')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ff9b0c2cc0>




![png](/assets/images/bitcoin-futures-arbitrage-part-5_files/bitcoin-futures-arbitrage-part-5_6_1.png)



```python
df2 = df1.resample('S').mean().interpolate()
plt.scatter(x=df2.index, y=df2)
#plt.plot(df1.resample('S').mean().interpolate(), label='2')
#plt.legend()
#plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-92-790b471c3bb3> in <module>
    ----> 1 df2 = df1.resample('S').mean().interpolate()
          2 plt.scatter(x=df2.index, y=df2)
          3 #plt.plot(df1.resample('S').mean().interpolate(), label='2')
          4 #plt.legend()
          5 #plt.show()


    ~\.conda\envs\data\lib\site-packages\pandas\core\generic.py in resample(self, rule, how, axis, fill_method, closed, label, convention, kind, loffset, limit, base, on, level)
       8153                      axis=axis, kind=kind, loffset=loffset,
       8154                      convention=convention,
    -> 8155                      base=base, key=on, level=level)
       8156         return _maybe_process_deprecations(r,
       8157                                            how=how,


    ~\.conda\envs\data\lib\site-packages\pandas\core\resample.py in resample(obj, kind, **kwds)
       1248     """
       1249     tg = TimeGrouper(**kwds)
    -> 1250     return tg._get_resampler(obj, kind=kind)
       1251 
       1252 


    ~\.conda\envs\data\lib\site-packages\pandas\core\resample.py in _get_resampler(self, obj, kind)
       1378         raise TypeError("Only valid with DatetimeIndex, "
       1379                         "TimedeltaIndex or PeriodIndex, "
    -> 1380                         "but got an instance of %r" % type(ax).__name__)
       1381 
       1382     def _get_grouper(self, obj, validate=True):


    TypeError: Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'RangeIndex'



```python

```
