import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.Structs import *
from src.Utils import *

class GrowingNeuralGas():
    """ Implements the growing neural gas algorithm as described by Bernd Fritzke """

    def __init__(self, input, trainingIterations):
        """
        Initializes the Growing Neural Gas algorithm.
        """
        self.input          : np.ndarray                = input
        self.visualization  : np.ndarray                = input
        self.nodes          : list[Node]                = []
        self.edges          : list[Edge]                = []
        self.dotLocations   : np.ndarray[TwoDimVector]  = []
        self.borders        : tuple[float]              = input.shape
        self.max_iterations : int                       = trainingIterations
        self.maxNodes       : int                       = 200
        self.lmbda          : int                       = 100
        self.alpha          : int                       = 0.5
        self.beta           : int                       = 0.00005
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
        counter = 0
        for (x, y), value in np.ndenumerate(self.input):
            if value == 0:
                counter += 1
                if counter % 5 == 0: # decrease the amount of data points
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
        src.neighbours.append(dst)
        dst.neighbours.append(src)
        self.edges.append(Edge(src, dst, 0))
        

    def removeConnectionOfTwoNodes(self, src, dst):
        """
        Removes the edge between the two given nodes.
        """

        # Remove the neighbours
        if src in dst.neighbours and dst in src.neighbours:
            src.neighbours.remove(dst)
            dst.neighbours.remove(src)

            # Update the local edges
            for edge in self.edges:
                if edge.src == src and edge.dst == dst:
                    self.edges.remove(edge)
                if edge.src == dst and edge.dst == src:
                    self.edges.remove(edge)

        return False


    def visualize(self):
        """
        Illustrates the network by adding nodes and edges to the self.visualization array and display it using matplotlib.
        """
        print("Visializing...")
        # Show the visualization array in matplotlib
        #plt.imshow(self.visualization, cmap='gray')

        # # with a single-pixel dot, illustrate self.dotLocations
        for dot in self.dotLocations:
            plt.scatter(dot.x, -dot.y, color='cyan', s=1)

        # with a small red dot add all nodes
        for node in self.nodes:
            plt.plot(node.pos.x, -node.pos.y, 'r.')
        
        # with a thin green, transparent line add all edges
        for edge in self.edges:
            plt.plot([edge.src.pos.x, edge.dst.pos.x], [-edge.src.pos.y, -edge.dst.pos.y], 'g-', alpha=0.5, linewidth=0.5)            
        
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
                if edge.src.id == first.node.id or edge.dst.id == first.node.id:
                    edge.age += 1

                    # Remove the oldies
                    if edge.age > self.maxAge:
                        self.removeConnectionOfTwoNodes(edge.src, edge.dst)

            # Check if we should add a new node
            if i % self.lmbda == 0 and self.numNodes < self.maxNodes:
                # Find the node with the highest error
                maxErrorNode = None
                maxError = 0
                for node in self.nodes:
                    if node.error > maxError:
                        maxError = node.error
                        maxErrorNode = node

                # FInd the maxErrorNode neighbour with the largest error
                maxErrorNeighbour = None
                maxErrorNeighbourError = 0
                for neighbour in maxErrorNode.neighbours:
                    if neighbour.error >= maxErrorNeighbourError:
                        maxErrorNeighbourError = neighbour.error
                        maxErrorNeighbour = neighbour
                
                # Insert a new node between the maxErrorNeighbour and maxErrorNode
                newNode = self.createNode(TwoDimVector((maxErrorNode.pos.x + maxErrorNeighbour.pos.x) / 2, (maxErrorNode.pos.y + maxErrorNeighbour.pos.y) / 2))
                self.connectTwoNodes(maxErrorNode, newNode)
                self.connectTwoNodes(newNode, maxErrorNeighbour)
                self.removeConnectionOfTwoNodes(maxErrorNode, maxErrorNeighbour)

                # Decrease the error of the maxErrorNode and maxErrorNeighbour
                maxErrorNode.error *= self.alpha
                maxErrorNeighbour.error *= self.alpha
                newNode.error = maxErrorNode.error

                # Decrease all other node errors
                for node in self.nodes:
                    if node != maxErrorNode and node != maxErrorNeighbour:
                        node.error *= self.beta

        self.visualize()