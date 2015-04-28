# kNN
k-Nearest Neighbors algorithm (aka kNN) is a non-parametric method used for classification and regression

* Pros: High accuracy, insensitive to outliers, no assumptions about data 
* Cons: Computationally expensive, requires a lot of memory
* Works with: Numeric values, nominal values

## Basic steps

* Collect: Any method.
* Prepare: Numeric values are needed for a distance calculation. A structured data format is best.
* Analyze: Any method.
* Train: Does not apply to the kNN algorithm.
* Test: Calculate the error rate.
* Use: This application needs to get some input data and output structured numeric values. 
Next, the application runs the kNN algorithm on this input data and determines which class the 
input data should belong to. The application then takes some action on the calculated class.

### Normalize values
When comparing features which values that lie in different ranges it is recommended to normalize them. 
Common ranges to normalize are 0 to 1 or -1 to 1. To normalize from 0 to 1:

```python
new_value = (old_value - min)/(max - min)
```

Where min and max are the smallest and biggest value per feature.

# Decision trees
Take a set of data and provide insights from it, distilling data into knowledge.

* Pros: computationally cheap to use, easy for humans to understand learned results, missing values OK, can deal with 
 irrelevant features.
* Cons: Prone to overfitting
* Works with: Numeric values, nominal values

## Basic steps
* Collect: Any method.
* Prepare: This tree-building algorithm works only on nominal values, so quantized the rest
* Analyze: Any method.
* Train: Construct a tree structure.
* Test: Calculate the error rate.
* Use: Supervised learning. To better understand the data. 

## Information theory applied to split a dataset
Decide on which feature to use to split the data. this means finding the best feature. The process is done one and
again until all data is classified.

The function that creates every branch in the tree, must be recursive.

It is a way of organizing a dataset, because it is a way of measuring the amount of information in every feature.

### ID3

Stands for *Iterative Dichotomiser 3*

Defines the way of selecting the starting attribute used to generate a decision tree:
**Iterate over all the attributes in the dataset and select the attribute with smallest entropy or greatest information gain.**

[More info on ID3 algorithm](http://en.wikipedia.org/wiki/ID3_algorithm)

It's a recursive algorithm, after every execution, takes the resulting branch dataset and selects a new starting attribute
among the never selected before. Every execution "consumes" an attribute.

Recursion stops when:

* Every element in a subset belongs to the same class.
* There are no more attributes to be selected. It all the elements of the subset do not belong to the same one, the most
common one should be selected.

### Measuring information: Information theory
Information gain: the change in information before and after the split.

**Shannon entropy**

Entropy is the expected value of the information: I(xi) = log2p(xi), where p(xi) is the probability of choosing 
this class.

    H = -sum(n, i=1)[p(xi)log2p(xi)]

n is the number of classes.

The higher the entropy, the higher the information gain of the data.

**Gini impurity**

Probability of choosing an item from the set and the probability of that item in being misclassified:

    GI(t) = sum(n, i!= j)[p(i|t)p(j|t)]


