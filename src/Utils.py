from src.Structs import *
import math
import numpy as np

def nodeEclDst(a : Node, b : Node) -> float:
    """
    Calculates the euclidean between two nodes.
    """
    return math.sqrt((a.pos.x - b.pos.x)**2 + (a.pos.y - b.pos.y)**2)


def dpNodeEclDst(dp : TwoDimVector, b : Node) -> float:
    """
    Calculates the euclidean between two nodes.
    """
    return math.sqrt((dp.x - b.pos.x)**2 + (dp.y - b.pos.y)**2)
