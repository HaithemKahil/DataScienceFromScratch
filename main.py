# ----------------------------------------------------------
# KNN Classifier Tests -------------------------------------
# ----------------------------------------------------------

import requests
from typing import List,Dict, Tuple
from collections import defaultdict
from Utils import LinearAlgebraUtils
import csv
from matplotlib import pyplot as plt
import random
from KNNChapter import KNNClassifier


def parse_dataset_rows(row: List[str]) -> LinearAlgebraUtils.LabeledPoint:
    features = [float(value) for value in row[:-1]]
    label = row[-1].split("-")[-1]
    return LinearAlgebraUtils.LabeledPoint(features, label)

# Downloading the iris flowers dataset


# data_set = requests.get("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
# with open('iris.dat','w') as file:
#     file.write(data_set.text)

with open("DataSets/iris.dat") as dataset_file :
    reader = csv.reader(dataset_file)
    iris_data = [parse_dataset_rows(row) for row in reader]

points_by_species: Dict[str,List[float]] = defaultdict(list)
for element in iris_data:
    points_by_species[element.label].append(element.point)

# Simple data exploration
# in this part we will plot the different pair combinations and see how
# well features are separating data points

metrics = ['sepal length','sepal width','petal length','petal width']
plot_pairs = [(i,j) for i in range(4) for j in range(4) if i<j]
marks = ['+','.','x']
fig, ax = plt.subplots(2,3)

for row in range(2):
    for column in range(3):
        i,j = plot_pairs[3 * row + column]
        ax[row][column].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize = 8)
        ax[row][column].set_xticks([])
        ax[row][column].set_yticks([])
        for mark, (species, points) in zip(marks,points_by_species.items()):
            x_values = [point[i] for point in points]
            y_values = [point[j] for point in points]
            ax[row][column].scatter(x_values, y_values, marker = mark, label = species)

ax[-1][-1].legend(loc = 'lower right', prop = {'size':6})
plt.show()

# Splitting Data to training and testing datasets

random.seed(10)
random.shuffle(iris_data)
cut_point = int(0.7 * len(iris_data))
iris_train, iris_test = iris_data[:cut_point],iris_data[cut_point:]

# KNN Classification and calculating number of correct classified points

confusion_matrix : Dict[Tuple[str,str], int] = defaultdict(int)
correctly_classified = 0

for element in iris_test:
    predicted_class = KNNClassifier.classify(5,iris_train,element)
    actual = element.label
    if predicted_class == actual :
        correctly_classified += 1

    confusion_matrix[(predicted_class, actual)] += 1

pct_correct = correctly_classified / len(iris_test)
print(pct_correct,confusion_matrix)
