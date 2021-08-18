"""
    MLStart
    Machine Learning in Python

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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
    print("\nTrain data:")
    print(" X_train shape:", X_train.shape)
    print(" y_train shape:", y_train.shape)


if __name__ == '__main__':
    print("MLStart v.0.1")
    main()
