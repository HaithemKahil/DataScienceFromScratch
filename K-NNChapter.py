from typing import List
from collections import Counter
from Utils import LinearAlgebraUtils


# Implementing a basic KNN classifier
class KNNClassifier:
    # In case many labels have an equal number of votes, we have 3 alternative approaches to choose between
    #   1 - Pick up the winner randomly
    #   2 - Weight the votes by distance and pick the weighted winner
    #   3 - reduce k until one winner finding only one winner

    # first approach
    def raw_majority_votes(labels:List[str]) -> str:
        votes = Counter(labels)
        winner, votes_number = votes.most_common(1)[0]
        print(winner,  votes_number)
        return winner

    # third approach
    def majority_votes(labels: List[str]) -> str:
        votes = Counter(labels)
        winner, winner_count = votes.most_common(1)[0]
        num_winners = len([num for num in votes.values() if num == winner_count])
        if num_winners == 1 :
            return winner
        else:
            return KNNClassifier.majority_votes(labels[:-1])

    def classify(k:int, labeled_points: List[LinearAlgebraUtils.LabeledPoint], new_point:LinearAlgebraUtils.LabeledPoint):
        ordered_by_distance = sorted(labeled_points,key=lambda lp:LinearAlgebraUtils.distace(lp,new_point))
        k_nearest_labels = [lp.label for lp in ordered_by_distance[:k]]
        return KNNClassifier.majority_votes(k_nearest_labels)
