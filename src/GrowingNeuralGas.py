from operator import attrgetter
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.Structs import *
from src.Utils import *

class GrowingNeuralGas():
    """ Implements the growing neural gas algorithm as described by Bernd Fritzke """

    def __init__(self, input):
        """
        Initializes the Growing Neural Gas algorithm.
        """
        self.input          : np.ndarray                = input
        self.visualization  : np.ndarray                = input
        self.nodes          : list[Node]                = []
        self.edges          : list[Edge]                = []
        self.dotLocations   : np.ndarray[TwoDimVector]  = []
        self.borders        : tuple[float]              = input.shape
        self.max_iterations : int                       = 75000
        self.maxNodes       : int                       = 50
        self.newNodeEvery   : int                       = 5
        self.max_error      : float                     = 0.1
        self.max_distance   : int                       = 140
        self.maxAge         : int                       = 10
        self.minDistance    : int                       = 10
        self.learningRate   : float                     = 0.5
        self.threshold      : float                     = 0.1
        self.numNodes       : int                       = 0

        self.setDataPoints()
        self.createInitialNodes()
        self.train()

    
    def setDataPoints(self):
        """ Get centre points of each blob in the image """
        data = []
        for (x, y), value in np.ndenumerate(self.input):
            if value == 0:
                data.append(TwoDimVector(x, y))
        self.dotLocations = np.array(data)

    def createNode(self, pos):
        """
        Creates a new node at the given position.
        """
        node = Node(self.numNodes, pos, 0, [])
        self.nodes.append(node)
        self.numNodes += 1
        return node


    def removeNode(self, nodeId):
        """
        Removes the given node from the network. Return true if successful
        """
        for node in self.nodes:
            if node.id == nodeId:
                # Remove the node and decrement the numNodes
                self.nodes.remove(node)
                self.numNodes -= 1
                return True
        return False


    def connectTwoNodes(self, src, dst):
        """
        Creates an edge between the two given nodes.
        """
        # edgeSrc/Dst are there to simply ensure that src is always the source node's id for aesthetic purposes
        edgeSrc = Edge(src, dst, 0)
        self.edges.append(edgeSrc)
        


    def removeConnectionOfTwoNodes(self, src, dst):
        """
        Removes the edge between the two given nodes.
        """
        # Check if src and dst are connected, if so then remove the edge entry for both data structures, also self.edges
        for edge in src.edges:
            if edge.source.id == src.id and edge.target.id == dst.id:
                self.edges.remove(edge)
                return True
            if edge.target.id == src.id and edge.source.id == dst.id:
                self.edges.remove(edge)
                return True
            
                
        
        return False


    def visualize(self):
        """
        Illustrates the network by adding nodes and edges to the self.visualization array and display it using matplotlib.
        """
        # Show the visualization array in matplotlib
        plt.imshow(self.visualization, cmap='gray')

        # # with a single-pixel dot, illustrate self.dotLocations
        for dot in self.dotLocations:
            plt.scatter(dot.x, dot.y, color='cyan', s=1)

        # with a small red dot add all nodes
        for node in self.nodes:
            plt.plot(node.pos.x, node.pos.y, 'r.')
        
        # with a thin green, transparent line add all edges
        for edge in self.edges:
            plt.plot([edge.source.pos.x, edge.target.pos.x], [edge.source.pos.y, edge.target.pos.y], 'g-', alpha=0.5, linewidth=0.5)            
        
        plt.show()

    def createInitialNodes(self):
        """
        Creates the initial nodes at random positions.
        """
        # Create two nodes with a random position
        nodes = []
        for i in range(2):
            randomX = random.randint(0, self.borders[0] - 1)
            randomY = random.randint(0, self.borders[1] - 1)
            node = self.createNode(TwoDimVector(randomX, randomY))
            nodes.append(node)


    def findTwoClosestNodes(self, dataPoint : TwoDimVector):
        """
        Finds the node with the highest activation value.
        """
        class Distance():
            def __init__(self, node, distance):
                self.node = node
                self.distance = distance

        tmp = [None] * len(self.nodes) # preallocate memory
        tmp = np.array(tmp)
        for i, node in enumerate(self.nodes):
            dist = dpNodeEclDst(dataPoint, node)
            tmp[i] = Distance(node, dist)
        
        # Sort the array
        sortedByDistance = sorted(tmp, key=lambda tup: tup.distance)
        return sortedByDistance[0], sortedByDistance[1]


    def train(self):
        """
        Trains the network for the given number of iterations.
        """
        # Iterate over all nodes and edges for the number of max iters
        for i in range(self.max_iterations):
            print(f"Iteration {i} / {self.max_iterations}")

            # Take a random data point
            randomDataPointIndex = random.randint(0, len(self.dotLocations) - 1)
            randomDataPoint = self.dotLocations[randomDataPointIndex]

            # Find the two closest nodes
            first, second = self.findTwoClosestNodes(randomDataPoint)
            if first.distance < self.minDistance:
                print("No nearby nodes")
                continue

            # Update error distance and positions
            first.node.error += first.distance
            first.node.pos.x += self.learningRate * (randomDataPoint.x - first.node.pos.x)
            first.node.pos.y += self.learningRate * (randomDataPoint.y - first.node.pos.y)

            # Connect the two nodes
            self.connectTwoNodes(first.node, second.node)

            # Update edge ages
            for edge in self.edges:
                if edge.source.id == first.node.id or edge.target.id == first.node.id:
                    edge.age += 1

                    # Remove the oldies
                    if edge.age > self.maxAge:
                        self.removeConnectionOfTwoNodes(edge.source, edge.target)

            # Check if we should add a new node
            if i % self.newNodeEvery == 0 and self.numNodes < self.maxNodes:
                
                largestErrorNode = max(self.nodes, key=attrgetter('error'))
                # Get the largest error neighbour that largestErrorNode has by querying self.edges
                tmp = 0
                largestNeighbour = None
                for edge in self.edges:
                    if edge.source.id == largestErrorNode.id:
                        if edge.target.error > tmp:
                            tmp = edge.target.error
                            largestNeighbour = edge.target
                    elif edge.target.id == largestErrorNode.id:
                        if edge.source.error > tmp:
                            tmp = edge.source.error
                            largestNeighbour = edge.source



                

                randomX = random.randint(0, self.borders[0] - 1)
                randomY = random.randint(0, self.borders[1] - 1)
                self.createNode(TwoDimVector(randomX, randomY))
                       
        self.visualize()