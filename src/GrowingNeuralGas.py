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
        self.max_iterations : int                       = 1000
        self.max_neighbours : int                       = 10
        self.max_error      : float                     = 0.1
        self.max_distance   : int                       = 140
        self.min_distance   : int                       = 1
        self.learning_rate  : float                     = 0.1
        self.threshold      : float                     = 0.1
        self.numNodes       : int                       = 0

        self.getCentrePoints()
        self.createInitialNodes()
        self.train()

    
    def getCentrePoints(self):
        """ Get centre points of each blob in the image """
        im2, contours = cv2.findContours(self.input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in im2:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # ignore centre of the image
                if cx != self.borders[0] // 2 - 1 and cy != self.borders[1] // 2 - 1:
                    self.dotLocations.append(TwoDimVector(cx, cy))
        
        self.dotLocations = np.array(self.dotLocations)

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
        edgeDst = Edge(dst, src, 0)
        src.edges.append(edgeSrc)
        dst.edges.append(edgeDst)
        self.edges.append(edgeSrc)


    def removeConnectionOfTwoNodes(self, src, dst):
        """
        Removes the edge between the two given nodes.
        """
        # Check if src and dst are connected, if so then remove the edge entry for both data structures, also self.edges
        edgetoRemoveFromSrcAndThis = None
        edgetoRemoveFromDst = None
        srcFound = False
        dstFound = False
        for edge in src.edges:
            if dst.id == edge.target.id:
                edgetoRemoveFromSrcAndThis = edge
                srcFound = True
                
        for edge in dst.edges:
            if dst.id == edge.source.id:
                edgeToRemoveFromDst = edge
                dstFound = True
                
        if srcFound is True and dstFound is True:
            src.edges.remove(edgetoRemoveFromSrcAndThis)
            self.edges.remove(edgetoRemoveFromSrcAndThis)
            dst.edges.remove(edgeToRemoveFromDst)
            return True

        elif srcFound is True and dstFound is False:
            print(f"Attempting to remove edges {src} -> {dst} but {dst} is not connected to {src}. Something is wrong with pointer maintenance")
            return False
        elif srcFound is False and dstFound is True:
            print(f"Attempting to remove edges {dst} -> {src} but {src} is not connected to {dst}. Something is wrong with pointer maintenance")
            return False
            
        else:
            return False

    def visualize(self):
        """
        Illustrates the network by adding nodes and edges to the self.visualization array and display it using matplotlib.
        """
        # Show the visualization array in matplotlib
        plt.imshow(self.visualization, cmap='gray')

        # with a single-pixel dot, illustrate self.dotLocations
        for dot in self.dotLocations:
            plt.scatter(dot.x, dot.y, color='cyan', s=1)

        # with a small red dot add all nodes
        for node in self.nodes:
            plt.plot(node.pos.x, node.pos.y, 'r.')
        
        # with a thin green, transparent line add all edges
        for edge in self.edges:
            plt.plot([edge.source.pos.x, edge.target.pos.x], [edge.source.pos.y, edge.target.pos.y], 'g-', alpha=0.5, linewidth=0.5)            
        
        # show the plot
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
        
        # Connect the two nodes
        self.connectTwoNodes(nodes[0], nodes[1])


    def findWinner():
        pass


    def train(self):
        """
        Trains the network for the given number of iterations.
        """
        # Iterate over all nodes and edges for the number of max iters
        for i in range(self.max_iterations):
            for node in self.nodes:
                for edge in node.edges:
                    # get find the two nodes that are the nearest to any point in self.dotLocations
                    

                    # Compute the distance between the source and target node
                    dst = nodeEclDst(node, edge.target)
                    # If the distance is greater than the max_distance then remove the edge
                    if dst > self.max_distance:
                        self.removeConnectionOfTwoNodes(edge.source, edge.target)
                    # If the distance is smaller than the min_distance then add a new node
                    elif dst < self.min_distance:
                        # Create a new node at the middle of the source and target node
                        middleX = (edge.source.pos.x + edge.target.pos.x) / 2
                        middleY = (edge.source.pos.y + edge.target.pos.y) / 2
                        middleNode = self.createNode(TwoDimVector(middleX, middleY))
                        # Connect the new node to the source and target node
                        self.connectTwoNodes(edge.source, middleNode)
                        self.connectTwoNodes(edge.target, middleNode)
                        # Connect the new node to the source and target node
                        self.connectTwoNodes(middleNode, edge.target)
                        self.connectTwoNodes(middleNode, edge.source)
                    # If the distance is between the min_distance and max_distance then update the error
                    else:
                        # Compute the error
                        error = self.max_error - dst / self.max_distance
                        # Update the error of the edge
                        edge.weight += error * self.learning_rate
                        # Update the error of the source node
                        edge.source.error += error * self.learning_rate
                        # Update the error of the target node
            
            self.visualize()
        
        
