import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
'''
        Initializes a node with properties: value, index, and connections.
        Value represents the node's associated value, index serves as its unique identifier,
        and connections is a list of indices of nodes connected to it. If no connections are specified during initialization,
        it defaults to an empty list.
'''

class Node:
    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections if connections is not None else []
        self.value = value

class Network:
    def __init__(self, nodes=None):
        '''
        Initializes a network with a list of nodes, initializing an empty list if no nodes are provided.
        '''
        self.nodes = nodes if nodes is not None else []

    def get_mean_degree(self):
        total_connection = sum(len(node.connections) for node in self.nodes)
        mean_degree = total_connection / len(self.nodes)
        return mean_degree
         '''
        Calculates the average number of connections each node has in the network.
        It sums the lengths of connections lists for all nodes and divides by the total number of nodes to find the mean degree.
        This metric helps assess the network's overall connectivity.
        '''

    def get_mean_clustering(self):
        total_clustering = 0

        for node in self.nodes:
            number_neighbour = len(node.connections)
            if number_neighbour < 2:
                continue

            actual_connection = 0
            for neighbour_index in node.connections:
                for other_node in self.nodes:
                    if neighbour_index in other_node.connections:
                        actual_connection += 1

            clustering_coefficient = actual_connection / (number_neighbour * (number_neighbour - 1) / 2)
            total_clustering += clustering_coefficient

        mean_clustering_coefficient = total_clustering / len(self.nodes)
        return mean_clustering_coefficient

    '''
        Calculates the average tendency for nodes to cluster together by computing the clustering coefficient for each node.
        It measures how closely neighbors are connected and averages these coefficients across all nodes to assess clustering in the network.
   '''

    def get_mean_path_length(self):
        total_mean_path_length = 0

        for node in self.nodes:
            total_path_length = 0
            visited = set()
            queue = [(self.nodes.index(node), 0)]

            while queue:
                current_node, path_length = queue.pop(0)
                visited.add(current_node)
                total_path_length += path_length

                for neighbor_index in self.nodes[current_node].connections:
                    if neighbor_index not in visited:
                        queue.append((neighbor_index, path_length + 1))

            num_neighbors = len(node.connections)
            if num_neighbors > 0:
                total_mean_path_length += total_path_length / num_neighbors

        return total_mean_path_length / len(self.nodes)
        '''
        Computes the average distance between nodes in the network by finding shortest paths between all pairs of nodes.
        '''
