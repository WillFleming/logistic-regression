# logistic-regression

This is an implementation of logistic regression in Python using only NumPy. Maximum likelihood estimation is performed using the method of [iteratively re-weighted least squares (IRLS)](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares).

For a detailed walkthrough of the algorithm and math behind logistic regression, [view the Jupyter notebook](https://nbviewer.jupyter.org/github/WillFleming/logistic-regression/blob/master/Logistic-Regression-with-IRLS.ipynb).

To simply run the regression:

```python
from models.logistic_regression import LogisticModel, read_dataset

# read_dataset() returns an instance of LogisticModel
model = read_dataset('datasets/juice.csv', name='juice').fit()
model.summary()
```
Which prints a summary of the fitted model:
```
juice: logistic regression

Growth ~ pH + Nisin + Temp + Brix

Coefficient     Estimate       
--------------- ---------------
<Intercept>     -7.246333840412728
Brix            -0.31173234862940336
Nisin           -0.066276262143294
Temp            0.1104223950074471
pH              1.885950986253122

Converged in 8 iterations (IRLS)
```

To predict with new data:
```python
test_X = read_dataset('datasets/juice_test.csv', name='logistic', data_only = True)[1]
model.predict(test_X, use_probability=True)
```

Which returns an array:
```
array([ 0.96195407,  0.97240465,  0.35067621,  0.02940531,  0.56176085,
        0.75815753,  0.13940303,  0.07039647,  0.01166177,  0.00191276,
        0.10981691,  0.00337951,  0.13385971,  0.24290651,  0.02448657,
        0.75615076,  0.01934562,  0.29819405,  0.21170375])
```


##### Sources
1. [*Machine Learning: A Probabilistic Perspective* by Kevin R Murphy](https://www.cs.ubc.ca/~murphyk/MLbook/)
2. [GLM Lecture Notes by Germán Rodríguez](http://data.princeton.edu/wws509/notes)
3. [Logistic Regression and Newton’s Method Lecture Notes by Cosma Shalizi](http://www.stat.cmu.edu/~cshalizi/402/lectures/14-logistic-regression/lecture-14.pdf)
