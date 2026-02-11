Here is the translation of your text into English, preserving your structure and code:

**Outlier Detection** is an important topic in machine learning. Algorithms of this type are relevant and used everywhere:
Cybersecurity, Banking systems, data preprocessing, medicine, log analysis, quality control, and this is just a small part of the list.
Today we will get acquainted with two such algorithms, compare them, and look at the results of our work.
In our study, we will evaluate the algorithms using the metrics **Recall** (the real share of those correctly marked as an anomaly) and **Precision** (Shows the share of true positives among all those the model marked as positive).

I took a sufficiently complex and voluminous dataset from Kaggle: [Link to dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/code?datasetId=310&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false)
It provides data on monetary transactions; our task is to identify anomalies (potential fraudulent operations).

### 1. IQR: Starting with the basics

IQR (Interquartile Range) is a quick and effective way to find deviations that does not require model training in the usual sense of the word.
The fact is that IQR is merely a statistical method of analysis based on "boundaries of normality."
To do this, we build a certain barrier in space; everything beyond this barrier is considered an anomaly.

Mathematically, we find  — the median of our values,  — the point below which 25% of the data lies, and  — the point below which 75% of the data lies.


Then:
Lower bound: 
Upper bound: 

Now, let's implement this using Pandas:

```python
import pandas as pd
import numpy as np

def detect_outliers_iqr(data: pd.Series):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return outliers, lower_bound, upper_bound

```

Now let's try running it on our data:

```bash
Name: Amount, Length: 31904, dtype: float64,

```

We obtained these results, but are all these operations anomalies for our tasks?
Of course not, because in our algorithm we were able to use only a one-dimensional array showing the sum of operations; this is the most important disadvantage of this algorithm. It is an elementary and fast way to find anomalies, but far from the best.

Moreover, let's look at the algorithm's results in percentage terms:
8.93% of all data are highlighted as anomalous! It would seem, why so many? But this is intuitively understandable,
because if we set fixed frames for the algorithm that simply cut off part of the data, based on nothing other than the fact
that they are at a significant distance from the center, we must be ready for such a result.

Let's evaluate our algorithm by metrics:
recall: 0.185
precision: 0.003
It turns out that among all anomalies found by the model, only 0.3 percent are actually them.
But, we were able to find 18.5% of the total share of anomalies!

Such a gap between metrics clearly illustrates the weakness of the one-dimensional statistical approach.

### 2. Now let's move on to more complex methods, one of which is Isolation Forest.

Its ready-made version is in the **scikit-learn** library:

```python
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pandas as pd


model = IsolationForest()

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(x_train)
predict = model.predict(x_val)
predict[predict == 1] = 0
predict[predict == -1] = 1

```

We initialized the model and started training. The algorithm works quite quickly even on a large amount of data;
in 3 seconds everything is ready. Now it's time to look at the obtained metrics:

recall: 0.82
precision: 0.039
We see significant progress compared to the statistical method!
The algorithm found 82% of all anomalies! BUT, only 3.9 percent of the anomalies predicted by the model are actually anomalies.
The result is quite good.

```python
def get_path_length(X, tree, current_height):
    if tree.is_leaf:
        return current_height + c(len(X))
    if X[tree.feature] < tree.split_value:
        return get_path_length(X, tree.left, current_height + 1)
    else:
        return get_path_length(X, tree.right, current_height + 1)

```

Here we see how Isolation Forest works from the inside:
If we have reached a leaf node, we return the length + correction, otherwise we go further, and the smaller the number we return, the higher the probability that this leaf is an anomaly.
If the feature value is less than the threshold, we recursively go left, otherwise to the right.

It works on the principle—the easier it is to separate data from the general group, the more suspicious it is.
We can say that we are talking about an n-dimensional feature space; the model determines a line in a random place that separates the data into two groups,
and for each group recursively carries out this algorithm; eventually, one element remains in each group. The fewer recursion calls needed, the higher the probability that
this element is an anomaly (i.e., the fewer lines were needed).
Intuitively, this is easy to imagine: if a point in n-dimensional space lies far from the main cluster, it can be cut off by literally a couple of random "cuts".
If the point is inside a dense group, the algorithm will need many more iterations to isolate it into a separate leaf.

This algorithm has a huge number of advantages: the number of features is not so important to it, it does not overfit, it has linear complexity,
but there are also disadvantages; the main one, I believe, is that if there is little data in the sample, the algorithm may consider them an anomaly.

It is time to sum up our experiment; we tried two methods: statistical (IQR) and a machine learning method,
based on an ensemble of trees (Isolation Forest). We see a colossal advantage of the ML method over the statistical one, but why?
The most obvious reason is that the statistical method works on one feature, which is extremely insufficient
in the context of our task. If we choose Amount as a feature, then our potentially fraudulent operation could be hidden among,
say, 10 consecutive operations for small amounts (while one big one would be recognized as an anomaly). And it turns out
that if a user spent several times more money than usual, then this is an anomalous operation.
But let's turn to the metrics, consider precision for IsolationForest. We can say that out of 1000 operations marked by us as fraudulent,
only 39 are actually them. And in fact, this is not as bad a result as it might seem at first, because in this task it is better to consider an extra operation "suspicious" than to miss a fraudster.