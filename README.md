
# Abalone, abalone, give me the formuoli

We will be looking at a dataset from the UCI machine learning repository called the Abalone Data Set. It can be found here: https://archive.ics.uci.edu/ml/datasets/abalone. Each data point contains multiple physical characteristics of a single abalone, and the goal is to develop a model to predict the number of rings. The number of rings dictates the age of the abalone, so the problem is to develop a model to predict the age of an abalone. Since rings come in integer values, both classification and regression are viable options. In this notebook we will use regression. All lengths in the data set are in millimeters, while all weights are grams. According to the data set description on the website, all continuous values have been scaled down by a factor of 200. 

We will adopt OSEMN pipeline strategy:

1) Obtain the data
2) Scrubbing or cleaning the data. This includes data imputation (filling in missing values) and adjusting column names.
3) Explore the data. Look for outliers or weird data. Explore the relationship between features and output varaibles. Construct a correlation matrix.
4) Model the data (ML, etc).
5) iNterpret the data. What conclusions can we make? What are the most important factors (features)? How are the varaibles related to each other? 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
```

## 1) Obtain the Data
I downloaded the data from the UCI website, saved it as a CSV, and added the headers myself. Now we just load the data.


```python
data = pd.read_csv("abalone.csv")
```

## 2) Clean The Data
I already added column names (hence the typo in one of the columns!) and missing values were removed before the dataset was added to the UCI repository.


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4177 entries, 0 to 4176
    Data columns (total 9 columns):
    Sex               4177 non-null object
    Length            4177 non-null float64
    Diameter          4177 non-null float64
    Height            4177 non-null float64
    Whole_Weight      4177 non-null float64
    Shucked_Weight    4177 non-null float64
    Viscera_Weight    4177 non-null float64
    SHell_Weight      4177 non-null float64
    Rings             4177 non-null int64
    dtypes: float64(7), int64(1), object(1)
    memory usage: 293.8+ KB


## 3) Explore the Data
Let's actually look at the data now.


```python
data.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Length</th>
      <th>Diameter</th>
      <th>Height</th>
      <th>Whole_Weight</th>
      <th>Shucked_Weight</th>
      <th>Viscera_Weight</th>
      <th>SHell_Weight</th>
      <th>Rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4177.000000</td>
      <td>4177.000000</td>
      <td>4177.000000</td>
      <td>4177.000000</td>
      <td>4177.000000</td>
      <td>4177.000000</td>
      <td>4177.000000</td>
      <td>4177.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.523992</td>
      <td>0.407881</td>
      <td>0.139516</td>
      <td>0.828742</td>
      <td>0.359367</td>
      <td>0.180594</td>
      <td>0.238831</td>
      <td>9.933684</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.120093</td>
      <td>0.099240</td>
      <td>0.041827</td>
      <td>0.490389</td>
      <td>0.221963</td>
      <td>0.109614</td>
      <td>0.139203</td>
      <td>3.224169</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.075000</td>
      <td>0.055000</td>
      <td>0.000000</td>
      <td>0.002000</td>
      <td>0.001000</td>
      <td>0.000500</td>
      <td>0.001500</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.450000</td>
      <td>0.350000</td>
      <td>0.115000</td>
      <td>0.441500</td>
      <td>0.186000</td>
      <td>0.093500</td>
      <td>0.130000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.545000</td>
      <td>0.425000</td>
      <td>0.140000</td>
      <td>0.799500</td>
      <td>0.336000</td>
      <td>0.171000</td>
      <td>0.234000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.615000</td>
      <td>0.480000</td>
      <td>0.165000</td>
      <td>1.153000</td>
      <td>0.502000</td>
      <td>0.253000</td>
      <td>0.329000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.815000</td>
      <td>0.650000</td>
      <td>1.130000</td>
      <td>2.825500</td>
      <td>1.488000</td>
      <td>0.760000</td>
      <td>1.005000</td>
      <td>29.000000</td>
    </tr>
  </tbody>
</table>
</div>



Two things stands out: first, the minimum height is 0, which must be a typo. Second, the smallest weight is (after rescaling) significantly less than a gram. Let's sort the first few values sorted by height.


```python
data.sort_values(by=["Height"]).head(15)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Length</th>
      <th>Diameter</th>
      <th>Height</th>
      <th>Whole_Weight</th>
      <th>Shucked_Weight</th>
      <th>Viscera_Weight</th>
      <th>SHell_Weight</th>
      <th>Rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3996</th>
      <td>I</td>
      <td>0.315</td>
      <td>0.230</td>
      <td>0.000</td>
      <td>0.1340</td>
      <td>0.0575</td>
      <td>0.0285</td>
      <td>0.3505</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>I</td>
      <td>0.430</td>
      <td>0.340</td>
      <td>0.000</td>
      <td>0.4280</td>
      <td>0.2065</td>
      <td>0.0860</td>
      <td>0.1150</td>
      <td>8</td>
    </tr>
    <tr>
      <th>236</th>
      <td>I</td>
      <td>0.075</td>
      <td>0.055</td>
      <td>0.010</td>
      <td>0.0020</td>
      <td>0.0010</td>
      <td>0.0005</td>
      <td>0.0015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2169</th>
      <td>I</td>
      <td>0.165</td>
      <td>0.115</td>
      <td>0.015</td>
      <td>0.0145</td>
      <td>0.0055</td>
      <td>0.0030</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1174</th>
      <td>F</td>
      <td>0.635</td>
      <td>0.495</td>
      <td>0.015</td>
      <td>1.1565</td>
      <td>0.5115</td>
      <td>0.3080</td>
      <td>0.2885</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3902</th>
      <td>I</td>
      <td>0.160</td>
      <td>0.120</td>
      <td>0.020</td>
      <td>0.0180</td>
      <td>0.0075</td>
      <td>0.0045</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>694</th>
      <td>I</td>
      <td>0.165</td>
      <td>0.110</td>
      <td>0.020</td>
      <td>0.0190</td>
      <td>0.0065</td>
      <td>0.0025</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>I</td>
      <td>0.160</td>
      <td>0.110</td>
      <td>0.025</td>
      <td>0.0195</td>
      <td>0.0075</td>
      <td>0.0050</td>
      <td>0.0060</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2381</th>
      <td>M</td>
      <td>0.155</td>
      <td>0.115</td>
      <td>0.025</td>
      <td>0.0240</td>
      <td>0.0090</td>
      <td>0.0050</td>
      <td>0.0075</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3190</th>
      <td>I</td>
      <td>0.200</td>
      <td>0.145</td>
      <td>0.025</td>
      <td>0.0345</td>
      <td>0.0110</td>
      <td>0.0075</td>
      <td>0.0100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>720</th>
      <td>I</td>
      <td>0.160</td>
      <td>0.110</td>
      <td>0.025</td>
      <td>0.0180</td>
      <td>0.0065</td>
      <td>0.0055</td>
      <td>0.0050</td>
      <td>3</td>
    </tr>
    <tr>
      <th>719</th>
      <td>I</td>
      <td>0.150</td>
      <td>0.100</td>
      <td>0.025</td>
      <td>0.0150</td>
      <td>0.0045</td>
      <td>0.0040</td>
      <td>0.0050</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2171</th>
      <td>I</td>
      <td>0.190</td>
      <td>0.130</td>
      <td>0.030</td>
      <td>0.0295</td>
      <td>0.0155</td>
      <td>0.0150</td>
      <td>0.0100</td>
      <td>6</td>
    </tr>
    <tr>
      <th>238</th>
      <td>I</td>
      <td>0.110</td>
      <td>0.090</td>
      <td>0.030</td>
      <td>0.0080</td>
      <td>0.0025</td>
      <td>0.0020</td>
      <td>0.0030</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2172</th>
      <td>I</td>
      <td>0.215</td>
      <td>0.150</td>
      <td>0.030</td>
      <td>0.0385</td>
      <td>0.0115</td>
      <td>0.0050</td>
      <td>0.0100</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



It's just two entries with height 0, so I think it's safe to drop them. While we're here entry 1174 is a clear outlier. Let's handle that later.


```python
data = data.loc[data["Height"] != 0.0]
```


```python
data.sort_values(by=["Whole_Weight"]).head(15)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Length</th>
      <th>Diameter</th>
      <th>Height</th>
      <th>Whole_Weight</th>
      <th>Shucked_Weight</th>
      <th>Viscera_Weight</th>
      <th>SHell_Weight</th>
      <th>Rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>236</th>
      <td>I</td>
      <td>0.075</td>
      <td>0.055</td>
      <td>0.010</td>
      <td>0.0020</td>
      <td>0.0010</td>
      <td>0.0005</td>
      <td>0.0015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>238</th>
      <td>I</td>
      <td>0.110</td>
      <td>0.090</td>
      <td>0.030</td>
      <td>0.0080</td>
      <td>0.0025</td>
      <td>0.0020</td>
      <td>0.0030</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2114</th>
      <td>I</td>
      <td>0.130</td>
      <td>0.095</td>
      <td>0.035</td>
      <td>0.0105</td>
      <td>0.0050</td>
      <td>0.0065</td>
      <td>0.0035</td>
      <td>4</td>
    </tr>
    <tr>
      <th>237</th>
      <td>I</td>
      <td>0.130</td>
      <td>0.100</td>
      <td>0.030</td>
      <td>0.0130</td>
      <td>0.0045</td>
      <td>0.0030</td>
      <td>0.0040</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>I</td>
      <td>0.140</td>
      <td>0.105</td>
      <td>0.035</td>
      <td>0.0140</td>
      <td>0.0055</td>
      <td>0.0025</td>
      <td>0.0040</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3899</th>
      <td>I</td>
      <td>0.140</td>
      <td>0.105</td>
      <td>0.035</td>
      <td>0.0145</td>
      <td>0.0050</td>
      <td>0.0035</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2169</th>
      <td>I</td>
      <td>0.165</td>
      <td>0.115</td>
      <td>0.015</td>
      <td>0.0145</td>
      <td>0.0055</td>
      <td>0.0030</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>719</th>
      <td>I</td>
      <td>0.150</td>
      <td>0.100</td>
      <td>0.025</td>
      <td>0.0150</td>
      <td>0.0045</td>
      <td>0.0040</td>
      <td>0.0050</td>
      <td>2</td>
    </tr>
    <tr>
      <th>526</th>
      <td>M</td>
      <td>0.155</td>
      <td>0.110</td>
      <td>0.040</td>
      <td>0.0155</td>
      <td>0.0065</td>
      <td>0.0030</td>
      <td>0.0050</td>
      <td>3</td>
    </tr>
    <tr>
      <th>696</th>
      <td>I</td>
      <td>0.155</td>
      <td>0.105</td>
      <td>0.050</td>
      <td>0.0175</td>
      <td>0.0050</td>
      <td>0.0035</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>720</th>
      <td>I</td>
      <td>0.160</td>
      <td>0.110</td>
      <td>0.025</td>
      <td>0.0180</td>
      <td>0.0065</td>
      <td>0.0055</td>
      <td>0.0050</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3902</th>
      <td>I</td>
      <td>0.160</td>
      <td>0.120</td>
      <td>0.020</td>
      <td>0.0180</td>
      <td>0.0075</td>
      <td>0.0045</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>694</th>
      <td>I</td>
      <td>0.165</td>
      <td>0.110</td>
      <td>0.020</td>
      <td>0.0190</td>
      <td>0.0065</td>
      <td>0.0025</td>
      <td>0.0050</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>I</td>
      <td>0.160</td>
      <td>0.110</td>
      <td>0.025</td>
      <td>0.0195</td>
      <td>0.0075</td>
      <td>0.0050</td>
      <td>0.0060</td>
      <td>4</td>
    </tr>
    <tr>
      <th>239</th>
      <td>I</td>
      <td>0.160</td>
      <td>0.120</td>
      <td>0.035</td>
      <td>0.0210</td>
      <td>0.0075</td>
      <td>0.0045</td>
      <td>0.0050</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Looking at this table I suspect that the low weight entry is real, since it only has one ring and has small length, diameter, and height. 

Let's construct a correlation matrix and a heatmap for our data.


```python
data.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Length</th>
      <th>Diameter</th>
      <th>Height</th>
      <th>Whole_Weight</th>
      <th>Shucked_Weight</th>
      <th>Viscera_Weight</th>
      <th>SHell_Weight</th>
      <th>Rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Length</th>
      <td>1.000000</td>
      <td>0.986802</td>
      <td>0.828108</td>
      <td>0.925217</td>
      <td>0.897859</td>
      <td>0.902960</td>
      <td>0.898419</td>
      <td>0.556464</td>
    </tr>
    <tr>
      <th>Diameter</th>
      <td>0.986802</td>
      <td>1.000000</td>
      <td>0.834298</td>
      <td>0.925414</td>
      <td>0.893108</td>
      <td>0.899672</td>
      <td>0.906084</td>
      <td>0.574418</td>
    </tr>
    <tr>
      <th>Height</th>
      <td>0.828108</td>
      <td>0.834298</td>
      <td>1.000000</td>
      <td>0.819886</td>
      <td>0.775621</td>
      <td>0.798908</td>
      <td>0.819596</td>
      <td>0.557625</td>
    </tr>
    <tr>
      <th>Whole_Weight</th>
      <td>0.925217</td>
      <td>0.925414</td>
      <td>0.819886</td>
      <td>1.000000</td>
      <td>0.969389</td>
      <td>0.966354</td>
      <td>0.955924</td>
      <td>0.540151</td>
    </tr>
    <tr>
      <th>Shucked_Weight</th>
      <td>0.897859</td>
      <td>0.893108</td>
      <td>0.775621</td>
      <td>0.969389</td>
      <td>1.000000</td>
      <td>0.931924</td>
      <td>0.883129</td>
      <td>0.420597</td>
    </tr>
    <tr>
      <th>Viscera_Weight</th>
      <td>0.902960</td>
      <td>0.899672</td>
      <td>0.798908</td>
      <td>0.966354</td>
      <td>0.931924</td>
      <td>1.000000</td>
      <td>0.908186</td>
      <td>0.503562</td>
    </tr>
    <tr>
      <th>SHell_Weight</th>
      <td>0.898419</td>
      <td>0.906084</td>
      <td>0.819596</td>
      <td>0.955924</td>
      <td>0.883129</td>
      <td>0.908186</td>
      <td>1.000000</td>
      <td>0.627928</td>
    </tr>
    <tr>
      <th>Rings</th>
      <td>0.556464</td>
      <td>0.574418</td>
      <td>0.557625</td>
      <td>0.540151</td>
      <td>0.420597</td>
      <td>0.503562</td>
      <td>0.627928</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, cmap=sns.diverging_palette(220, 10, as_cmap=True))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1581bc88>




![png](output_15_1.png)


Almost everything is strongly correlated with everything else, except ring count! :( Assuming body proportions don't vary much between different abalone this makes sense. 

All of the features features associated with weight are pretty much perfectly correlated with each other, with pearson coefficients > 0.95 for whole weight. This isn't surprising since they're just the weights of different parts of the same abalone. I think we can safely drop the shucked weight, viscera weight, and shell weight and just keep whole weight (PCA is probably overkill).

Length and diameter are just about perfectly correlated. The circumfrence of an ellipse is proportional to its length with the constant of proportionality determined by its eccentricity, so no surprise there. Height is strongly correlated with the remaining features, but not as strongly as, say, length with diameter. We'll replace these three features with their geometric mean. While keeping length, diameter, and height can only help with prediction, I think the benefits gained of an easier to interpret model outweight the marginal gain of predictability from keeping them.


```python
data_trunc = pd.DataFrame({ "Whole_Weight": data["Whole_Weight"],
                           "Char_Len": np.cbrt(data["Length"]*data["Diameter"]*data["Height"]),
                           "Rings": data["Rings"],
                          "Sex": data["Sex"]})
```

The column name Char_Len means characteristic length.


```python
data_trunc.corr()
#data_trunc["Log_Rings"] = np.log(data_trunc["Rings"])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Char_Len</th>
      <th>Rings</th>
      <th>Whole_Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Char_Len</th>
      <td>1.000000</td>
      <td>0.593123</td>
      <td>0.931968</td>
    </tr>
    <tr>
      <th>Rings</th>
      <td>0.593123</td>
      <td>1.000000</td>
      <td>0.540151</td>
    </tr>
    <tr>
      <th>Whole_Weight</th>
      <td>0.931968</td>
      <td>0.540151</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Whole_Weight and Char_Len are very strongly correlated. We could probably afford to drop one of these as well. Let's plot them against each other.


```python
plt.scatter(data_trunc["Whole_Weight"], data_trunc["Char_Len"])
```




    <matplotlib.collections.PathCollection at 0x1a18d82dd8>




![png](output_21_1.png)


The shape suggests the characteristic length, $\ell$, is related to the weight, $w$, by a power law: $\ell \propto w^{\alpha}$, with $\alpha < 1$. Let's see how well $\alpha = 1/3$ works.


```python
#data_trunc.plot(kind='scatter', x='Whole_Weight', y='Char_Len' )
plt.scatter(np.cbrt(data_trunc["Whole_Weight"]), data_trunc["Char_Len"])
```




    <matplotlib.collections.PathCollection at 0x1a18e61630>




![png](output_23_1.png)


This looks quite linear. We could try fiddling around with the power but this is working well enough that I don't think it's necessary. Rather than keep weight, define $x \equiv w^{1/3}$ and just use that.


```python
data_trunc["x"] = np.cbrt(data_trunc["Whole_Weight"])
```

Let's see how x is distributed by sex.


```python
dm = data_trunc.loc[data_trunc["Sex"] == 'M']
df = data_trunc.loc[data_trunc["Sex"] == 'F']
di = data_trunc.loc[data_trunc["Sex"] == 'I']
sns.distplot(dm["x"])
sns.distplot(df["x"])
sns.distplot(di["x"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a18d8de10>




![png](output_27_1.png)


It seems that male and female are quite similar physically, while infants are different from either. Let's combine the M and F categories and repllace the sex column with a binary Is_Infant column.


```python
data_trunc["Is_Infant"] = data_trunc["Sex"].map(lambda x: 0 if x=="M" or x=="F"  else 1  )
```


```python
data_trunc.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Char_Len</th>
      <th>Rings</th>
      <th>Sex</th>
      <th>Whole_Weight</th>
      <th>x</th>
      <th>Is_Infant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.250809</td>
      <td>15</td>
      <td>M</td>
      <td>0.5140</td>
      <td>0.801040</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.202855</td>
      <td>7</td>
      <td>M</td>
      <td>0.2255</td>
      <td>0.608670</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.310899</td>
      <td>9</td>
      <td>F</td>
      <td>0.6770</td>
      <td>0.878071</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.271781</td>
      <td>10</td>
      <td>M</td>
      <td>0.5160</td>
      <td>0.802078</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.188820</td>
      <td>7</td>
      <td>I</td>
      <td>0.2050</td>
      <td>0.589637</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(data_trunc.loc[data_trunc["Is_Infant"] == 0 ]["x"],
            data_trunc.loc[data_trunc["Is_Infant"] == 0 ]["Rings"], color="green")
plt.scatter(data_trunc.loc[data_trunc["Is_Infant"] == 1 ]["x"],
            data_trunc.loc[data_trunc["Is_Infant"] == 1 ]["Rings"], color="blue")
plt.ylabel("Ring Count")
```




    <matplotlib.text.Text at 0x1a19132b70>




![png](output_31_1.png)


We see that there is a fanning effect on ring count, where the scales of the residuals increase as the fitted values increase. To combat this, we wil take the log of ring count.


```python
data_trunc["Log_Rings"] = np.log(data_trunc["Rings"])
```


```python
plt.scatter(data_trunc["x"], data_trunc["Log_Rings"], color="red", alpha=0.4, marker='+')
plt.ylabel("Log of Ring Count")
```




    <matplotlib.text.Text at 0x1a191bca90>




![png](output_34_1.png)


Alright, let's get to modelling the data.

## 4) Model The Data

As a first pass we will model log of ring count with a spline fit by minizing the RSS.

SKlearn doesn't have a method for fitting splines, but SciPy does. The UnivariateSpline function will fit a curve to an input of (x,y) pairs. We can specify the degree of the fitting polynomial, how the knots are chosen, and the smoothing parameter. Let's construct separate splines for the infant and non-infant values.


```python
non_inf_data = data_trunc.loc[data_trunc["Is_Infant"] == 0]
non_inf_data = non_inf_data.sort_values(by=["x"], ascending=False)

inf_data = data_trunc.loc[data_trunc["Is_Infant"] == 1]
inf_data = inf_data.sort_values(by=["x"], ascending=False)



X_ni = non_inf_data["x"].as_matrix().reshape(-1,1)
y_ni = non_inf_data["Log_Rings"].as_matrix().reshape(-1,1)

X_i = inf_data["x"].as_matrix().reshape(-1,1)
y_i = inf_data["Log_Rings"].as_matrix().reshape(-1,1)
```


```python
from sklearn.model_selection import train_test_split

X_ni_train, X_ni_test, y_ni_train, y_ni_test = train_test_split(X_ni,y_ni, test_size=0.2, random_state=42)

X_i_train, X_i_test, y_i_train, y_i_test = train_test_split(X_i,y_i, test_size=0.2, random_state=42)
```


```python
from scipy.interpolate import UnivariateSpline

spl_ni_3 = UnivariateSpline(X_ni_train, y_ni_train, k=3)
spl_ni_5 = UnivariateSpline(X_ni_train, y_ni_train, k=5)

spl_i_3 = UnivariateSpline(X_i_train, y_i_train, k=3)
spl_i_5 = UnivariateSpline(X_i_train, y_i_train, k=5)
```


```python
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.scatter(X_ni, y_ni, color="purple", alpha=0.1, label="Non-Infant Data")
ax.plot(X_ni, spl_ni_3(X_ni), color="blue", label="Third Order Fit")
ax.plot(X_ni, spl_ni_5(X_ni), color="orange", label="Fifth Order Fit")
plt.ylabel("Log of Ring Count")
plt.xlabel("x")
ax.legend(loc="best")
```




    <matplotlib.legend.Legend at 0x1a19600a90>




![png](output_39_1.png)



```python
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.scatter(X_i, y_i, color="purple", alpha=0.1, label="Infant Data")
ax.plot(X_i, spl_i_3(X_i), color="blue", label="Third Order Fit")
ax.plot(X_i, spl_i_5(X_i), color="orange", label="Fifth Order Fit")
plt.ylabel("Log of Ring Count")
plt.xlabel("x")
ax.legend(loc="best")
```




    <matplotlib.legend.Legend at 0x1a19891198>




![png](output_40_1.png)


For both cases the third order and fifth order fits match each other very closely away from the boundaries. Since a cubic polynomial is simpler we should stick with that. Also the fifth order fit is concave down for large values of x, which makes no physical sense.

We can also adjust the smoothing factor, $\lambda$. Ideally we should choose the value of the smoothing factor using a cross validation set. However when I set the smoothing factor and plot the resulting splines certain values of $x$ don't show up. See the plots below.


```python
spl_i_3.set_smoothing_factor(0.1)
spl_i_5.set_smoothing_factor(0.1)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.scatter(X_i, y_i, color="purple", alpha=0.1, label="Infant Data")
ax.plot(X_i, spl_i_3(X_i), color="blue", label="Third Order Fit")
ax.plot(X_i, spl_i_5(X_i), color="orange", label="Fifth Order Fit")
plt.ylabel("Log of Ring Count")
plt.xlabel("x")
ax.legend(loc="best")
```




    <matplotlib.legend.Legend at 0x1a199f7d68>




![png](output_42_1.png)



```python
spl_ni_3.set_smoothing_factor(0.1)
spl_ni_5.set_smoothing_factor(0.1)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.scatter(X_ni, y_ni, color="purple", alpha=0.1, label="Non-Infant Data")
ax.plot(X_ni, spl_ni_3(X_ni), color="blue", label="Third Order Fit")
ax.plot(X_ni, spl_ni_5(X_ni), color="orange", label="Fifth Order Fit")
plt.ylabel("Log of Ring Count")
plt.xlabel("x")
ax.legend(loc="best")
```




    <matplotlib.legend.Legend at 0x1a19b71cf8>




![png](output_43_1.png)


Let's try using a GLM for this problem. Rather than treating ring count (or log of ring count) as a continuous variable, let's actually take the fact that they are discrete in account in the model. Denote $\lambda = E(Y|X=x)$ for the conditional expectation value of $Y$, where $Y$ is the ring count. Let's assume that the conditional distribution is Poisson, and we'll use a log link. Then, writing $\lambda = e^{f(x)}$, where $f$ is our model, we have

\begin{equation}
p(y|x) = \frac{\lambda^{y}e^{-\lambda}}{y !}
\end{equation}

It's easy to check that the log-likelihood $\ell$ is given by

\begin{equation}
\ell = \sum_{i} y^{(i)}f(x^{(i)}) - e^{f(x^{(i)})} - \log (y^{(i)}!)
\end{equation}

Taking a derivative with respect to the parameter $\theta$ gives

\begin{equation}
\partial_{\theta}\ell = \sum_{i} \left[ y^{(i)} - e^{f(x^{(i)})} \right] \frac{\partial f}{\partial \theta}(x^{(i)})
\end{equation}

For simplicty, let's take $f(x) = \theta_0 + \theta_1 x$. We could use a spline again, but looking at the plots above I think a linear model should be fine. If we solve the problem with gradient descent, the update procedure is
\begin{gather}
\theta_0 \leftarrow \theta_0 + \frac{\alpha}{m}\sum_{i} \left[ y^{(i)} - e^{f(x^{(i)})} \right] \\
\theta_1 \leftarrow \theta_1 + \frac{\alpha}{m}\sum_{i} \left[ y^{(i)} - e^{f(x^{(i)})} \right]x^{(i)}
\end{gather}

Just for fun, let's see how it does on the infant case (which looks the most linear).




```python
theta0 = 1.0
theta1 = 2.5
#theta2 = 0.5
#theta3 = 0.7

X = inf_data["x"].as_matrix()
Y = inf_data["Rings"].as_matrix()

m = len(X)
alpha = 0.0001

def f(x):
    return theta0 + theta1 * x 

for j in range(10000):
    delta0 = (alpha/m)*np.sum(Y - np.exp(f(X)))
    delta1 =(alpha/m)*np.dot(Y - np.exp(f(X)), X)
    theta0 += delta0
    theta1 += delta1


```


```python
plt.plot(X, np.exp(f(X)) , alpha=0.5, color='red')
plt.scatter(X, Y, alpha = 0.2, color='purple')
```




    <matplotlib.collections.PathCollection at 0x1a1b78fe48>




![png](output_46_1.png)



```python
plt.plot(X, f(X), alpha=0.5, color='red')
plt.scatter(X, np.log(Y), alpha = 0.2, color='purple')
```




    <matplotlib.collections.PathCollection at 0x1a1b97d630>




![png](output_47_1.png)


It seems to be doing okay, but I think the extra flexibility and local optimization used in the spline is why it worked well, as opposed to finding the most appropriate cost function.

## 5) Interpret the Data
Let's summarize what I did first.

The initial dataset consisted of 7 continuous predictors and a single categorical predictor. The 7 continuous predictors were found to be highly correlated, so I thought it would be a useful simplification to reduce the number of predictors. Three of the predictors were some length scale while the remaining four were weights associated with different processing steps of the abalone. I replaced the three length scales with their geometric mean, which I called $\ell$. I argued that because the whole weight, $w$, was nearly perfectly correlated with the remaining three weights, and because $w$ is the most natural weight scale, we can remove the other three values. Since weight scales with volume, or in this case $w \sim \ell^{3}$, we can keep a single predictor $x \equiv w^{1/3}$ (and sex). Finally, for the predictors considered here, sexual dysmorphia didn't seem to be present, and so I replaced the sex predictor with a binary "is infant".  

When I created a scatter plot for ring count versus $x$, I found that there was a flaring effect present, where the values became increasingly spread out as $x$ increased. To deal with this I took the logarithm of the ring count as the dependent variable. I then partitioned the data into two parts according to the "is infant" variable, and fit separate spline curves to each part. I wanted to add a cross validation set to tune the smoothing paramter but the code was being weird >:( I fit third order and fifth order splines and for both parts found the fifth order fit to have physically undesirable features.

---------------------------------------------------------

One could argue that since the original data set is relatively small and there are only a few features it's unnecessary to trim data out. After all more data gives more predictive power. There are a few reasons why I think it's a good idea to remove some features. First, the whole point of the problem is to predict the number of rings in abalone (and hence the age) in as few steps as possible. Second, since the features are so strongly correlated, any errors in the measurements process could mean an overall loss of information. We can avoid this by just keeping the most information-rich features. Since overall weight is the easiest feature to obtain, I decided to just keep that. Finally, fewer input variables means an easier to interpret model. Even though a spline fit is more complicated than, say, a linear fit, it's still easy to understand what it's doing just by looking at a plot of the curve overlayed on the data.

From the plots above we see that there is still significant variance between the model and the data. I doubt keeping more features would significantly improve this. The authors of the original data set suggest that more information (weather patterns, location, or other features) are necessary to get a really predictive model. Still, we can at least get a qualitative understanding of the relationship between age and weight. 





