# ----------------------------------------------------------
# KNN Classifier Tests -------------------------------------
# ----------------------------------------------------------

import requests
from typing import List,Dict
from collections import defaultdict
from Utils import LinearAlgebraUtils
import csv


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

print(points_by_species)