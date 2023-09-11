from typing import Union, Any
from cv2 import Mat


# Interface
class Interface:
    Methods = list[str, int, Any]
    Images = Union[Mat, list[Mat]]
    Bboxs = list[list[int]]
    W = Union[int, float]
    H = Union[int, float]
    ReadFormat = tuple[Mat, W, H, Bboxs, list[Union[str, int]]]
