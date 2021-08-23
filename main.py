"""
    MLStart
    Machine Learning in Python

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn


def describe_dataset(dataset):
    print("\nKeys of iris_dataset:\n", dataset.keys())
    print("\nTarget names:", dataset['target_names'])
    print("\nFeature names:\n", dataset['feature_names'])
    print("\nFirst five rows of data:\n", dataset['data'][:10])


def main():
    iris_dataset = load_iris()
    describe_dataset(iris_dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)
    # create dataframe from data in X_train
    # label the columns using the strings in iris_dataset.feature_names
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # create a scatter matrix from the dataframe, color by y_train
    pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8, cmap=mglearn.cm3)


if __name__ == '__main__':
    print("MLStart v.0.1")
    print("pandas version:", pd.__version__)
    main()
