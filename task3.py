import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Node:
    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections if connections is not None else []
        self.value = value

class Network:
    def __init__(self, nodes=None):
        self.nodes = nodes if nodes is not None else []

    def get_mean_degree(self):
        total_connection = sum(len(node.connections) for node in self.nodes)
        mean_degree = total_connection / len(self.nodes)
        return mean_degree

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
