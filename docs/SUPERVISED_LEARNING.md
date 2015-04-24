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