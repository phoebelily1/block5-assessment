import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse

class NetworkSimulation:
    def __init__(self):
        pass

    def create_ring_network(self, n, k=1):
        """
        A ring network is generated, with each node connected to k nearest neighbor nodes on either side.
        """
        G = nx.Graph()
        nodes = list(range(n))
        G.add_nodes_from(nodes)
        for node in nodes:
            neighbors = [(node - i) % n for i in range(1, k+1)] + [(node + i) % n for i in range(1, k+1)]
            G.add_edges_from((node, neighbor) for neighbor in neighbors)
        return G

    def create_small_world_network(self, n, p=0.2, k=2):
        """
        从环形网络开始，通过随机“重连”边来生成小世界网络。
        """
        G = self.create_ring_network(n, k)
        nodes = list(G.nodes())
        edges = list(G.edges())
        for edge in edges:
            if random.random() < p:
                new_node = random.choice(nodes)
                while new_node == edge[0] or G.has_edge(edge[0], new_node):
                    new_node = random.choice(nodes)
                G.add_edge(edge[0], new_node)
        return G

    def plot_network(self, G, title):
        """
        Use matplotlib to plot the network.
        """
        plt.figure(figsize=(8, 8))
        colors = [plt.cm.viridis(i / len(G.nodes())) for i in range(len(G.nodes()))]
        nx.draw_circular(G, node_color=colors, node_size=700, with_labels=True, font_weight='bold')
        plt.title(title)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate and plot different types of networks.")
    parser.add_argument('-ring_network', type=int, help='Create a ring network with a range of 1 and a size of N')
    parser.add_argument('-small_world', type=int, help='Create a small-worlds network with default parameters')
    parser.add_argument('-re_wire', type=float, default=0.2, help='Rewiring probability for the small world network')

    args = parser.parse_args()

    network_sim = NetworkSimulation()

    if args.ring_network is not None:
        G = network_sim.create_ring_network(args.ring_network)
        network_sim.plot_network(G, f"Ring Network with N={args.ring_network}")
    elif args.small_world is not None:
        G = network_sim.create_small_world_network(args.small_world, args.re_wire)
        network_sim.plot_network(G, f"Small World Network with N={args.small_world} and Rewiring Probability {args.re_wire}")

if __name__ == "__main__":
    main()
