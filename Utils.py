from typing import NamedTuple,List


class LabeledPoint(NamedTuple):
    point: List[float]
    label: str