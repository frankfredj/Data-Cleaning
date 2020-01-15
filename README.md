# Inputs

This function takes 6 inputs, namely:

* train.path: the location of the training file (csv)
* test.path: the location of the testing file (csv)
* numeric.max.missing: the maximum % of missing values to be tolerated amongst numeric predictors (in format 0. ... )
* unique.vals: the cut-off for transforming a numeric predictor into a factor predictor (i.e. : length(unique(...)) <= unique.vals)
* scale.numerics: should the numeric predictors be scaled?
* id.col: is the first column an identifier (i.e.: not a predictor)

Note that the last column of the training file should contain the target variable (i.e.: y)


# General pipeline

Numerics and factors are separated, and the later will be one-hot encoded. Missing factor values will be treated as their own category.
Missing numeric values will be imputed via a Random Forest algorithm. This is done separately, without merging the training and testing
files.

A linear model data frame is then created by using a Box-Cox transform on the numeric predictors with a skewness >= 0.75. Lambdas are
calculated via cross-validation on the training set, and then applied to both the training and testing set. The column space basis is
then tested for singularity (i.e.: some columns can be expressed as linear combination of other columns). If the basis is singular, then
some columns will be dropped in order to remediate the issue. Finally, Box-Cox is applied to Y and stored into a separate variable.


# Outputs

* train.clean: the clean training set
* test.clean: the clean testing set
* y: the target variable
* train.clean.LINEAR: the transformed clean training set to be used for linear regression
* test.clean.LINEAR: the transformed clean testing set to be used for linear regression
* y.L : the Box-Cox transformed y to be used for linear regression


#Additional details

The function will use all but two of your machine's cores. The tasks to be executed in parallel processing are:

* One-hot encoding of factors
* Imputating via Random Forest
* Scaling of numeric predictors
* Box-Cox lambdas CV
* Box-Cox transforms


#Performances

Using the House Price dataset from Kaggle (https://www.kaggle.com/c/house-prices-advanced-regression-techniques), containing the following
files:

* 1460 x 79 train file
* 1459 x 79 test file

And using an i7-3630qm laptop processor, we achieve the following result:

![](https://i.imgur.com/RqDBHcL.png)





