# Loan Default

This repo is trying to predict if a person is going to default on their loan.

- Each person in the data is identified by the SK_ID_CURR column which is an ID for each person because releasing names is never a good idea.
- The TARGET column is if the person defaulted on the loan. 1 for yes and 0 for no.
- The main data is in application_train.csv and application_test.csv, but the other CSV files can be processed and joined to the data to make a better dataset for use. This is called feature engineering and is very important.

# Notebooks

- **Data_Exploration.ipynb** <br/>
  Data_Exploration is a notebook that explores the data. It found that 97% of the rows have at least one column with a null value. Tree models such as Decision Trees and Random Forests cannot handle null values in their training data.
  How will we handle null values? This lead to using [imputation](https://stats.idre.ucla.edu/wp-content/uploads/2016/02/multipleimputation.pdf) to fill null values in the data. It also found that the data was imbalanced, which means that there are a lot more cases of non-defaulting loans than defaulting loans. Imbalance in the dataset can negatively affect tree based models. A model with imbalanced training data may achieve a high test acurracy even though the model is bad. For example the model could predict false everytime, and the model would still get an accuracy of over ninety percent because over ninety percent of the classes are false in the imbalanced dataset. This is why ROC score is used in combination with test accuracy. The model will still get a low ROC score with imbalanced data because the ROC score measures false positives and false negative, and not the total amount of correct guesses like test accuracy. To fix the data imbalance a technique called [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) was used.
- **Data_Exploration_FullSet.ipynb** <br/>
  Data_Exploration notebook only looked at a select amount of columns from application_train.csv. Data_Exploration_FullSet notebook takes all the columns from application_train.csv and does label encoding on the columns to convert from categorical to numerical data. The notebook also uses imputation and SMOTE on the full set of data. The notebook finally finds the optimal DecisionTreeClassifier using GridSearchCV.
- **Feature_Eng.ipynb** <br/>
  This notebook is an attempt at doing feature engineering.
  Tries to transform the other CSV files to make new meaningful columns and then join the new columns to the dataset. Saves the new dataset to csv for Submission_Notebook.ipynb to use.
- **Model_Comparison.ipynb** <br/>
  Model_Comparison compares models by preparing the data and models to make submissions to be graded by the kaggle system.
  The kaggle scoring system takes a csv of (SK_ID_CURR, TARGET) where SK_ID_CURR is the persons ID and TARGET is the percent likelihood that person will default on their loan(ranging between 0.00 and 1.00).
- **LightGBM.ipynb** <br/>
  LightGBM is the state of the art gradient boosting model. LightGBM works with null values in the dataset, so no need to use imputation. LightGBM can also handle imbalanced data, using a classification balance technique such as SMOTE is not necessary. By combining a multitude of weak hypotheses on important features, this model can find nuanced relationships to increase the accuracy of predictions. 
- **LGBM_hyperparameterization.ipynb** <br/>
  LightGBM is the state of the art gradient boosting model. LightGBM works with null values in the dataset, so no need to use imputation.
  Here we label/onehot encode the input data, visualize the effects of a few different parameters on the LGBM classifier, and then create a final LGBM classifier using those parameters and produce the lgbm_sub.csv submission document with the classified application_test data.
- **Model Performance.ipynb** <br/>
  This notebook compares popular models such as Decision Trees, Random Forests, Logistic Regression and gradient boosting models, then plots them all on a single AUC-ROC graph to determine the baseline optimal model for our project. We determined that LightGBM was the model to go for, and we also included the features which this model found to be the most important.

### Installation

```sh
$ conda create --name=Loan-Default
$ conda activate Loan-Default
$ conda install scikit-learn
$ conda install matplotlib
$ conda install pandas
$ conda install numpy
$ conda install scipy
$ conda install seaborn
$ conda install jupyter
$ conda install -c conda-forge imbalanced-learn
$ conda install -c conda-forge lightgbm
$ jupyter notebook
```

### Libraries and Techniques

#### [SMOTE](https://arxiv.org/pdf/1106.1813.pdf)

The dataset for the loan defautlts is highly class-imbalanced, since most people do not default on their loans. This creates a problem with training the data. This can be addressed by creating synthetic data in the minority class (the defaulting cases). SMOTE does this by connecting each of the existing examples in the dataset to its closest k examples, with k defaulting to 5 in the library. it then produces synthetic examples that fit on the lines connecting the points. Doing this iteratively produces enough synthetic data to balance the set.

#### [Imputation](https://stats.idre.ucla.edu/wp-content/uploads/2016/02/multipleimputation.pdf)

97% percent of the dataset examples are missing values from at least one feature. A basic strategy for handling this situation would be to drop any columns that have any missing values, but this would reduce the size of the training set by 97%, down to only 11351 examples from the original 307511, losing valuable data. To mitigate this, Imputation attempts to fill in the missing information by infering them from the known examples. scikit-learn's SimpleImputer allows for constant value filling, as well as statistical filling. For this project, the filling is done using the mean of the known examples. Ideally, we would use a more sophisticated technique for imputation such as k-NN, but with LightGBM being the model we went ahead with, imputation was no longer necessary.

#### [Decision Tree](https://hunch.net/~coms-4771/quinlan.pdf)

This model imposes a multitude of split criterion over a specified amount of nodes, which begins by determining the most important of the available features in the dataset. At each split, the model finds the next important feature to split on by using impurity measurements such as entropy or the Gini index, and continues until the model reaches the specified number of leaves or features.

#### [Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

A random forest model is composed of a set of decision trees, where it takes the majority vote or average prediction, depending on if the problem is a classification or regression problem.

#### [LightGBM]([https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)

Simplistically, a gradient boosting model uses a series of weak hypotheses on a set of features to create nuanced meshes, like a set of linear separators. More information on gradient boosting models and specifically the LightGBM model can be found in the above link.
