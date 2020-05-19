---
layout: post
title: "Perpetual Funding Rate ARIMA Models"
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
import matplotlib.pyplot as plt
import scipy
from matplotlib import style
import seaborn as sns
from IPython.core.pylabtools import figsize
import warnings
import statsmodels.api as sm

%matplotlib inline
plt.style.use('ggplot')
```


```python
np.random.seed(2019)
```


```python
# Read all historical sales data and index data
df = pd.read_csv('downloads/BTC-PERPETUAL.txt')
```


```python
# Properly index the pandas dataframe
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.index = df['date']
df = df.drop(columns=['timestamp', 'date'])
df.columns = ['date', 'perpetual', 'index']
```


```python
df['ratio'] = df['perpetual'] / df['index']
```


```python
df.describe()
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
      <th>perpetual</th>
      <th>index</th>
      <th>ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.739576e+07</td>
      <td>2.739576e+07</td>
      <td>2.739576e+07</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.793806e+03</td>
      <td>8.790548e+03</td>
      <td>1.000265e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.473821e+03</td>
      <td>2.470238e+03</td>
      <td>2.642148e-03</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.121500e+03</td>
      <td>3.126330e+03</td>
      <td>8.501664e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.703750e+03</td>
      <td>7.701290e+03</td>
      <td>9.996183e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.444000e+03</td>
      <td>9.442560e+03</td>
      <td>1.000240e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.056100e+04</td>
      <td>1.055535e+04</td>
      <td>1.000883e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.391450e+04</td>
      <td>1.385573e+04</td>
      <td>1.251092e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_D = df.resample('D').last().interpolate()
```


```python
df.resample('8H').last().interpolate().describe()
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
      <th>perpetual</th>
      <th>index</th>
      <th>ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6772.631726</td>
      <td>6771.830643</td>
      <td>0.999992</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2595.348401</td>
      <td>2592.762525</td>
      <td>0.000808</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3140.000000</td>
      <td>3141.320000</td>
      <td>0.995980</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3991.250000</td>
      <td>3992.470000</td>
      <td>0.999481</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6431.250000</td>
      <td>6433.255000</td>
      <td>0.999993</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8747.250000</td>
      <td>8739.232500</td>
      <td>1.000493</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12937.500000</td>
      <td>12926.800000</td>
      <td>1.003497</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_D.to_csv('daily_perpetual_spot_ratio.csv')
```


```python
figsize(7,3)
df_D.plot(y=['perpetual', 'index'])
plt.ylabel('Price ($)')
plt.title('Deribit Perpetual Sales Prices')
plt.show()
```


![png](/assets/images/2019-05-28-funding-rate-arima-model_files/2019-05-28-funding-rate-arima-model_10_0.png)



```python
figsize(6,3)
df_D.plot(y='ratio', legend=False)
plt.ylabel('Ratio')
plt.title('Perpetural Index Ratio vs Time')
plt.show()
```


![png](/assets/images/2019-05-28-funding-rate-arima-model_files/2019-05-28-funding-rate-arima-model_11_0.png)



```python
sm.graphics.tsa.plot_acf(df_D['ratio'], lags=40)
plt.show()
```


![png](/assets/images/2019-05-28-funding-rate-arima-model_files/2019-05-28-funding-rate-arima-model_12_0.png)



```python
sm.graphics.tsa.plot_pacf(df_D['ratio'], lags=40)
plt.show()
```


![png](/assets/images/2019-05-28-funding-rate-arima-model_files/2019-05-28-funding-rate-arima-model_13_0.png)



```python
sm.tsa.stattools.adfuller(df_D['ratio'], maxlag=10, regression='ct')
```




    (-6.314735984634745,
     4.678722300365411e-07,
     3,
     443,
     {'1%': -3.9793522915426682,
      '5%': -3.4204471749378556,
      '10%': -3.1329068263745925},
     -5234.242543485603)




```python
df_D_diff = df_D['ratio'].diff()[1:]
```


```python
sm.tsa.stattools.adfuller(df_D_diff, maxlag=10, regression='ct')
```




    (-10.033224693489473,
     1.962353349225152e-15,
     8,
     437,
     {'1%': -3.9796369453725298,
      '5%': -3.4205845422066594,
      '10%': -3.1329875260529962},
     -5197.00904872605)



First difference is clearly stationary. Some evidence that no diff is not stationary. DF test c, ct, and ctt all reject null, but not DF test with nc.

Estimate Models with Both


```python
res = sm.tsa.arma_order_select_ic(df_D['ratio'], max_ar=5, max_ma=5, ic=['aic', 'bic'], fit_kw={'method':'css-mle'})
print(res)
```

    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\tsa\kalmanf\kalmanfilter.py:649: RuntimeWarning: divide by zero encountered in true_divide
      R_mat, T_mat)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\tsa\tsatools.py:607: RuntimeWarning: invalid value encountered in true_divide
      (1+np.exp(-params))).copy()
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\tsa\tsatools.py:609: RuntimeWarning: invalid value encountered in true_divide
      (1+np.exp(-params))).copy()
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:508: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\tsa\tsatools.py:650: RuntimeWarning: invalid value encountered in true_divide
      newparams = ((1-np.exp(-params))/(1+np.exp(-params))).copy()
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\tsa\tsatools.py:651: RuntimeWarning: invalid value encountered in true_divide
      tmp = ((1-np.exp(-params))/(1+np.exp(-params))).copy()
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:508: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:508: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)


    {'aic':              0            1            2            3            4  \
    0 -5180.704496 -5280.219359 -5311.576438 -5322.052563 -5330.564203   
    1 -5333.072907 -5357.503554 -5358.480995 -5356.524349          NaN   
    2 -5346.739381 -5358.438353 -5356.503612 -5354.708097 -5270.102621   
    3 -5351.237785 -5356.547647 -5354.439180          NaN -5327.036950   
    4 -5355.017719 -5355.169046 -5353.183672 -5358.069382          NaN   
    5 -5354.311319 -5351.017978 -5351.998921 -5356.288051          NaN   
    
                 5  
    0 -5335.781545  
    1 -5353.277258  
    2          NaN  
    3 -5323.770297  
    4 -5355.820691  
    5 -4386.224784  , 'bic':              0            1            2            3            4  \
    0 -5172.499379 -5267.911683 -5295.166203 -5301.539770 -5305.948851   
    1 -5320.765231 -5341.093320 -5337.968202 -5331.908998          NaN   
    2 -5330.329146 -5337.925560 -5331.888260 -5325.990186 -5237.282153   
    3 -5330.724992 -5331.932295 -5325.721270          NaN -5290.113923   
    4 -5330.402367 -5326.451136 -5320.363203 -5321.146355          NaN   
    5 -5325.593408 -5318.197509 -5315.075894 -5315.262465          NaN   
    
                 5  
    0 -5307.063635  
    1 -5320.456789  
    2          NaN  
    3 -5282.744711  
    4 -5310.692546  
    5 -4336.994081  , 'aic_min_order': (1, 2), 'bic_min_order': (1, 1)}


    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\Users\chris\.conda\envs\data\lib\site-packages\statsmodels\base\model.py:508: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)



```python
best_aic_model = None
best_bic_model = None
lowest_AIC = float('inf')
lowest_BIC = float('inf')
for p in range(15):
    for q in range(15):
        for d in range(2):
            try:
                model = sm.tsa.ARIMA(df_D['ratio'], (p,d,q)).fit()
                if model.aic < lowest_AIC:
                    best_aic_model = model
                    lowest_AIC = model.aic
                if model.bic < lowest_BIC:
                    best_bic_model = model
                    lowest_BIC = model.bic
            except:
                pass
```


```python
sm.tsa.ARIMA(df_D['ratio'], (20,0,1)).fit().summary()
```


```python
model = sm.tsa.ARIMA(df_D['ratio'], (5,1,2)).fit(disp=False)
model.summary()
```


```python
df.loc['2018-12-01':'2018-12-07']['ratio'].resample('1Min').last().interpolate().describe()
```




    count    10080.000000
    mean         0.998998
    std          0.000896
    min          0.993191
    25%          0.998457
    50%          0.998994
    75%          0.999573
    max          1.002621
    Name: ratio, dtype: float64


