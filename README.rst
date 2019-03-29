##################################
Kaggle : Home Credit Default Risk
##################################

Goal
=======

Fetch the Kaggle competition data from the `Home Credit Default Risk Competition <https://www.kaggle.com/c/home-credit-default-risk>`_,
generate numeric and categorical features then build models using Tensorflow, Scikit-Learn and XGBoost.


Features
========
The `competition dataset <https://www.kaggle.com/c/home-credit-default-risk/data>`_ contains 8 raw data files.
The files include a "loan application" file that information collected at the time of the application. The other 7 files
contain supplementary data collect about the loan application from 3rd parties.


Numeric Features
~~~~~~~~~~~~~~~~
For each file in the dataset, we generate min, max, median, mean for each numeric feature. This resulted in ``213 features``.


Category Count Features
~~~~~~~~~~~~~~~~~~~~~~~

For each file in the dataset, we convert all the categorical features into one-hot encoded features. There is a
1:N relationship between the loan application data and supplementary datasets. So there can be many
supplementary rows per 1 loan application row. We simply sum the categorical {0, 1} values per
loan application id to get the count per category. We then join this dataset to the loan application dataset using the application ID.

This resulted in ``191`` features


Final Training Set
~~~~~~~~~~~~~~~~~~

Final training set include numeric and categorical count features from all the data files. The final dataset contained
``651`` features.


Model
=====

We trained a `fully-connected dense neural network using Tensorflow <https://www.tensorflow.org/>`_,
a `random forest model using scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
and a `XG Boost Model <https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn>`_

I didn't perform a parameter sweep to find optimal model parameters. I used common values seen in kernels where
the parameters could run on my consumer-grade laptop in under 20 minutes.



Best Model Performance
~~~~~~~~~~~~~~~~~~~~~~

* xgboost_kaggle_higgs_exp/main_plus_all/submissions.csv
* AUC Score 0.68446 (best competition model scored 0.81471)