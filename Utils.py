from typing import NamedTuple,List
import math


class LinearAlgebraUtils:
    class LabeledPoint(NamedTuple):
        point: List[float]
        label: str

    def distace(self,lp_one,lp_two:LabeledPoint):
        result = 0
        sum_squared_vector = sum([(x - y)**2 for x,y in zip(lp_one.point,lp_two.point)])
        return math.sqrt(sum_squared_vector)
