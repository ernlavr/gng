from dataclasses import dataclass

# Fix for forward declaration for type hinting below
class Edge:
    pass
class Node:
    pass


@dataclass
class TwoDimVector():
    """ Utility for representing a two dimensional vector. """

    """ Pos X """
    x : int
    """ Pos Y """
    y : int

@dataclass
class Edge():
    """
    Edge class for the Growing Neural Gas algorithm
    """

    """ Source ID """
    source : Node
    """ Target ID """
    target : Node
    """ Weight """
    age : int

@dataclass
class Node():
    """ 
    Node class for the Growing Neural Gas algorithm
    """

    """ ID of the node"""
    id      : int
    """ Position of a node in space """
    pos     : TwoDimVector
    """ Local accumulated error """
    error   : float
    """ Set of edges that define the topological neighbours of this node """
    edges   : list[Edge]

