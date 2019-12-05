
# ANOVA  - Lab

## Introduction

In this lab, you'll get some brief practice generating an ANOVA table (AOV) and interpreting its output. You'll also perform some investigations to compare the method to the t-tests you previously employed to conduct hypothesis testing.

## Objectives

In this lab you will: 

- Use ANOVA for testing multiple pairwise comparisons 
- Interpret results of an ANOVA and compare them to a t-test

## Load the data

Start by loading in the data stored in the file `'ToothGrowth.csv'`: 


```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv('ToothGrowth.csv')
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
      <th>len</th>
      <th>supp</th>
      <th>dose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4.2</td>
      <td>VC</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11.5</td>
      <td>VC</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7.3</td>
      <td>VC</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5.8</td>
      <td>VC</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td>4</td>
      <td>6.4</td>
      <td>VC</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



## Generate the ANOVA table

Now generate an ANOVA table in order to analyze the influence of the medication and dosage:  


```python
formula = 'len ~ C(supp) + dose'
lm = ols(formula, df).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                   sum_sq    df           F        PR(>F)
    C(supp)    205.350000   1.0   11.446768  1.300662e-03
    dose      2224.304298   1.0  123.988774  6.313519e-16
    Residual  1022.555036  57.0         NaN           NaN


## Interpret the output

Make a brief comment regarding the statistics and the effect of supplement and dosage on tooth length: 


```python
# Dosage and Supplement both appear to be influential, with dosage having the most substantial effect on tooth length.
```

## Compare to t-tests

Now that you've had a chance to generate an ANOVA table, its interesting to compare the results to those from the t-tests you were working with earlier. With that, start by breaking the data into two samples: those given the OJ supplement, and those given the VC supplement. Afterward, you'll conduct a t-test to compare the tooth length of these two different samples: 


```python
vc_df = df[df['supp'] == 'VC']
oj_df = df[df['supp'] == 'OJ']
```


```python
oj_teeth = np.array(oj_df['len'])
vc_teeth = np.array(vc_df['len'])
oj_teeth, vc_teeth
```




    (array([15.2, 21.5, 17.6,  9.7, 14.5, 10. ,  8.2,  9.4, 16.5,  9.7, 19.7,
            23.3, 23.6, 26.4, 20. , 25.2, 25.8, 21.2, 14.5, 27.3, 25.5, 26.4,
            22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23. ]),
     array([ 4.2, 11.5,  7.3,  5.8,  6.4, 10. , 11.2, 11.2,  5.2,  7. , 16.5,
            16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5, 23.6, 18.5,
            33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5]))



Now run a t-test between these two groups and print the associated two-sided p-value: 


```python
# Calculate the 2-sided p-value for a t-test comparing the two supplement groups
import numpy as np
import scipy.stats as stats
from flatiron_stats import welch_t, welch_df, p_value_welch_ttest

p_value_welch_ttest(oj_teeth, vc_teeth, two_sided=True)
```




    0.06063450788093383




```python
#alternative method

from scipy import stats

stats.ttest_ind(oj_teeth, vc_teeth, equal_var=False)[1]
```




    0.06063450788093387



## A 2-Category ANOVA F-test is equivalent to a 2-tailed t-test!

Now, recalculate an ANOVA F-test with only the supplement variable. An ANOVA F-test between two categories is the same as performing a 2-tailed t-test! So, the p-value in the table should be identical to your calculation above.

> Note: there may be a small fractional difference (>0.001) between the two values due to a rounding error between implementations. 


```python
# Your code here; conduct an ANOVA F-test of the oj and vc supplement groups.
# Compare the p-value to that of the t-test above. 
# They should match (there may be a tiny fractional difference due to rounding errors in varying implementations)

formula = 'len ~ C(supp)'
lm = ols(formula, df).fit()
table = sm.stats.anova_lm(lm)
print(table)
```

                df       sum_sq     mean_sq         F    PR(>F)
    C(supp)    1.0   205.350000  205.350000  3.668253  0.060393
    Residual  58.0  3246.859333   55.980333       NaN       NaN


## Run multiple t-tests

While the 2-category ANOVA test is identical to a 2-tailed t-test, performing multiple t-tests leads to the multiple comparisons problem. To investigate this, look at the various sample groups you could create from the 2 features: 


```python
for group in df.groupby(['supp', 'dose'])['len']:
    group_name = group[0]
    data = group[1]
    print(group_name)
```

    ('OJ', 0.5)
    ('OJ', 1.0)
    ('OJ', 2.0)
    ('VC', 0.5)
    ('VC', 1.0)
    ('VC', 2.0)


While bad practice, examine the effects of calculating multiple t-tests with the various combinations of these. To do this, generate all combinations of the above groups. For each pairwise combination, calculate the p-value of a 2-sided t-test. Print the group combinations and their associated p-value for the two-sided t-test.


```python
# Your code here; reuse your t-test code above to calculate the p-value for a 2-sided t-test
# for all combinations of the supplement-dose groups listed above. 
# (Since there isn't a control group, compare each group to every other group.)

OJ_05 = df.query("supp == 'OJ' and dose == 0.5")
OJ_10 = df.query("supp == 'OJ' and dose == 1.0")
OJ_20 = df.query("supp == 'OJ' and dose == 2.0")
VC_05 = df.query("supp == 'VC' and dose == 0.5")
VC_10 = df.query("supp == 'VC' and dose == 1.0")
VC_20 = df.query("supp == 'VC' and dose == 2.0")
```


```python
oj5 = np.array(OJ_05['len'])
oj1 = np.array(OJ_10['len'])
oj2 = np.array(OJ_20['len'])
vc5 = np.array(VC_05['len'])
vc1 = np.array(VC_10['len'])
vc2 = np.array(VC_20['len'])
```


```python
p_vals = []
p_vals.append(p_value_welch_ttest(oj5, oj1, two_sided=True))
p_vals.append(p_value_welch_ttest(oj5, oj2, two_sided=True))
p_vals.append(p_value_welch_ttest(oj2, oj1, two_sided=True))
p_vals.append(p_value_welch_ttest(vc5, vc1, two_sided=True))
p_vals.append(p_value_welch_ttest(vc5, vc2, two_sided=True))
p_vals.append(p_value_welch_ttest(vc2, vc1, two_sided=True))
p_vals.append(p_value_welch_ttest(vc5, oj5,two_sided=True))
p_vals.append(p_value_welch_ttest(vc5, oj1,two_sided=True))
p_vals.append(p_value_welch_ttest(vc5, oj2,two_sided=True))
p_vals.append(p_value_welch_ttest(vc1, oj5,two_sided=True))
p_vals.append(p_value_welch_ttest(vc1, oj1,two_sided=True))
p_vals.append(p_value_welch_ttest(vc1, oj2,two_sided=True))
p_vals.append(p_value_welch_ttest(vc2, oj5,two_sided=True))
p_vals.append(p_value_welch_ttest(vc2, oj1,two_sided=True))
p_vals.append(p_value_welch_ttest(vc2, oj2,two_sided=True))
p_vals
```




    [8.784919055160323e-05,
     1.323783877626994e-06,
     0.03919514204624397,
     6.811017703167721e-07,
     4.681577414622495e-08,
     9.155603056631989e-05,
     0.00635860676409683,
     3.655206737285255e-08,
     1.3621326289126046e-11,
     0.04601033257637566,
     0.0010383758722998238,
     2.3610742028168374e-07,
     7.196253523966689e-06,
     0.09652612338267019,
     0.9638515887233756]




```python
# faster method
from itertools import combinations

groups = [group[0] for group in df.groupby(['supp', 'dose'])['len']]
combos = combinations(groups, 2)
for combo in combos:
    supp1 = combo[0][0]
    dose1 = combo[0][1]
    supp2 = combo[1][0]
    dose2 = combo[1][1]
    sample1 = df[(df.supp == supp1) & (df.dose == dose1)]['len']
    sample2 = df[(df.supp == supp2) & (df.dose == dose2)]['len']
    p = stats.ttest_ind(sample1, sample2, equal_var=False)[1]
    print(combo, p)

```

    (('OJ', 0.5), ('OJ', 1.0)) 8.784919055161479e-05
    (('OJ', 0.5), ('OJ', 2.0)) 1.3237838776972294e-06
    (('OJ', 0.5), ('VC', 0.5)) 0.006358606764096813
    (('OJ', 0.5), ('VC', 1.0)) 0.04601033257637553
    (('OJ', 0.5), ('VC', 2.0)) 7.196253524006043e-06
    (('OJ', 1.0), ('OJ', 2.0)) 0.039195142046244004
    (('OJ', 1.0), ('VC', 0.5)) 3.6552067303259103e-08
    (('OJ', 1.0), ('VC', 1.0)) 0.001038375872299884
    (('OJ', 1.0), ('VC', 2.0)) 0.09652612338267014
    (('OJ', 2.0), ('VC', 0.5)) 1.3621396478988818e-11
    (('OJ', 2.0), ('VC', 1.0)) 2.3610742020468435e-07
    (('OJ', 2.0), ('VC', 2.0)) 0.9638515887233756
    (('VC', 0.5), ('VC', 1.0)) 6.811017702865016e-07
    (('VC', 0.5), ('VC', 2.0)) 4.6815774144921145e-08
    (('VC', 1.0), ('VC', 2.0)) 9.155603056638692e-05


## Summary

In this lesson, you implemented the ANOVA technique to generalize testing methods to multiple groups and factors.
