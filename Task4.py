import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

class NetworkSimulation:
    def __init__(self):
        """
        Initializes the NetworkSimulation class.
        """
        pass

    def create_ring_network(self, n, k=1):
        """
        Creates a ring network where each node is connected to its k nearest neighbors on both sides.
        Parameters:
            n (int): Number of nodes in the network.
            k (int): Each node is connected to k neighbors on each side.
        Returns:
            tuple: A tuple containing the list of nodes and the list of edges.
        """
        nodes = list(range(n))
        edges = []
        for node in nodes:
            for i in range(1, k + 1):
                edges.append((node, (node - i) % n))  # Connect to k previous neighbors (circular)
                edges.append((node, (node + i) % n))  # Connect to k next neighbors (circular)
        return nodes, edges

    def create_small_world_network(self, n, p=0.2, k=2):
        """
        Generates a small-world network starting from a ring network by randomly re-wiring edges.
        Parameters:
            n (int): Number of nodes in the network.
            p (float): Probability of re-wiring each edge.
            k (int): Each node is initially connected to k neighbors on each side.
        Returns:
            tuple: A tuple containing the list of nodes and the list of edges.
        """
        nodes, edges = self.create_ring_network(n, k)
        for edge in list(edges):
            if random.random() < p/2:
                new_node = random.choice(nodes)
                # Ensure new edge does not create self-loops or duplicate edges
                while new_node == edge[0] or (edge[0], new_node) in edges or (new_node, edge[0]) in edges:
                    new_node = random.choice(nodes)
                edges.append((edge[0], new_node))  # Add the new edge
        return nodes, edges

    def plot_network(self, nodes, edges, title):
        """
        Plots the network using matplotlib.
        Parameters:
            nodes (list): List of nodes in the network.
            edges (list): List of edges in the network.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(8, 8))
        theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)  # Compute angular positions of nodes
        pos = {node: (np.cos(t), np.sin(t)) for node, t in zip(nodes, theta)}  # Position nodes in a circle

        # Plot nodes
        for node, position in pos.items():
            plt.scatter(*position, c='blue', s=100, zorder=5)
            plt.text(position[0], position[1], str(node), color='white', ha='center', va='center', zorder=10)

        # Plot edges
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            plt.plot([x0, x1], [y0, y1], 'gray', zorder=0)
        plt.title(title)
        plt.axis('off')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate and plot different types of networks.")
    parser.add_argument('-ring_network', type=int, help='Create a ring network with a specified number of nodes.')
    parser.add_argument('-small_world', type=int, help='Create a small-world network with specified parameters.')
    parser.add_argument('-re_wire', type=float, default=0.2, help='Rewiring probability for the small world network.')

    args = parser.parse_args()

    network_sim = NetworkSimulation()

    if args.ring_network is not None:
        nodes, edges = network_sim.create_ring_network(args.ring_network)
        network_sim.plot_network(nodes, edges, f"Ring Network with N={args.ring_network}")
    elif args.small_world is not None:
        nodes, edges = network_sim.create_small_world_network(args.small_world, args.re_wire)
        network_sim.plot_network(nodes, edges, f"Small World Network with N={args.small_world} and Rewiring Probability {args.re_wire}")

if __name__ == "__main__":
    main()
