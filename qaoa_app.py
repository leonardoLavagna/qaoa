import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from classes import Qaoa, Problems
import os
os.chdir('..')

# Function to create a graph from user input
def create_graph():
    st.header("Create Your Graph")

    num_nodes = st.number_input("Enter the number of nodes", min_value=2, max_value=20, value=5)
    st.write(f"Graph with {num_nodes} nodes will be created.")

    edges_input = st.text_area("Enter the edges (format: 'node1 node2')", 
                               placeholder="1 2\n2 3\n3 4", 
                               height=200)
    edges = []
    for line in edges_input.split('\n'):
        if line.strip():
            node1, node2 = map(int, line.split())
            edges.append((node1, node2))

    G = nx.Graph()
    G.add_edges_from(edges)
    return G, num_nodes


# Function to display the graph
def plot_graph(G):
    plt.figure(figsize=(6,6))
    nx.draw(G, with_labels=True, font_weight='bold', node_color='lightblue', edge_color='gray')
    st.pyplot(plt)


# Function to solve MaxCut using QAOA
def solve_maxcut(G, num_nodes):
    st.header("Solve MaxCut with QAOA")

    # Create the problem instance using the Problems class
    problem = Problems(num_nodes, G)

    # Initialize the QAOA class
    qaoa_solver = Qaoa(problem)

    # Solve the MaxCut
    maxcut_result = qaoa_solver.solve()

    st.write("MaxCut Solution:")
    st.write(f"MaxCut Value: {maxcut_result['cut_value']}")
    st.write(f"Partitioning of nodes: {maxcut_result['partition']}")

# Streamlit app layout
def main():
    st.title("MaxCut Problem Solver with QAOA")

    G, num_nodes = create_graph()

    if len(G.edges) > 0:  # Only proceed if graph is created
        plot_graph(G)
        if st.button("Solve MaxCut"):
            solve_maxcut(G, num_nodes)

if __name__ == "__main__":
    main()
