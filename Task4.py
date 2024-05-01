import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

class NetworkSimulation:
    def __init__(self):
        pass

    def make_ring_network(self, N, neighbour_range=1):
        nodes = list(range(N))
        edges = []
        for node in nodes:
            for i in range(1, neighbour_range + 1):
                edges.append((node, (node - i) % N))  # Connect to k previous neighbors (circular)
                edges.append((node, (node + i) % N))  # Connect to k next neighbors (circular)
        return nodes, edges

    def make_small_world_network(self, N, re_wire_prob=0.2):
        nodes, edges = self.make_ring_network(N, 2)
        for edge in list(edges):
            if random.random() < re_wire_prob/2:
                new_node = random.choice(nodes)
                # Ensure new edge does not create self-loops or duplicate edges
                while new_node == edge[0] or (edge[0], new_node) in edges or (new_node, edge[0]) in edges:
                    new_node = random.choice(nodes)
                edges.append((edge[0], new_node))  # Add the new edge
        return nodes, edges

    def plot(self, nodes, edges, title):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(nodes)
        network_radius = num_nodes * 10

        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        pos = {node: (network_radius * np.cos(t), network_radius * np.sin(t)) for node, t in zip(nodes, theta)}

        # Plot nodes using Circle
        for node, position in pos.items():
            circle = plt.Circle(position, 0.3*num_nodes, color='blue', zorder=5)
            ax.add_patch(circle)
            ax.text(position[0], position[1], str(node), color='white', ha='center', va='center', zorder=10)

        # Plot edges
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            ax.plot([x0, x1], [y0, y1], 'gray', zorder=0)

        plt.title(title)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate and plot different types of networks.")
    parser.add_argument('-ring_network', type=int, help='Create a ring network with a specified number of nodes.')
    parser.add_argument('-small_world', type=int, help='Create a small-world network with specified parameters.')
    parser.add_argument('-re_wire', type=float, default=0.2, help='Rewiring probability for the small world network.')

    args = parser.parse_args()
    network_sim = NetworkSimulation()

    if args.ring_network is not None:
        nodes, edges = network_sim.make_ring_network(args.ring_network)
        network_sim.plot(nodes, edges, f"Ring Network with N={args.ring_network}")
    elif args.small_world is not None:
        nodes, edges = network_sim.make_small_world_network(args.small_world, args.re_wire)
        network_sim.plot(nodes, edges, f"Small World Network with N={args.small_world} and Rewiring Probability {args.re_wire}")

if __name__ == "__main__":
    main()
