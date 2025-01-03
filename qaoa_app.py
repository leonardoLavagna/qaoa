import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from classes import Qaoa as Q
from classes import Problems as P
from functions import qaoa_utilities as qaoa_utils
from functions import maxcut_utilities as mcut_utils
from functions import qaoa_optimizers as optims
from qiskit.visualization import plot_histogram
from qiskit.visualization import circuit_drawer
from qiskit import transpile
from qiskit_aer import Aer

def create_instance():
    st.header("Create Your QAOA instance")
    num_nodes = st.number_input("Enter the number of nodes of the graph over which you want to compute the maxcut value:", min_value=2, max_value=20, value=4)
    edges_input = st.text_area("Enter the edges (format: 'node1 node2' with a space between nodes)", 
                               placeholder="0 1\n1 2\n2 3\n3 0", 
                               height=200)
    edges = []
    for line in edges_input.split('\n'):
        if line.strip():
            node1, node2 = map(int, line.split())
            edges.append((node1, node2))
    G = nx.Graph()
    G.add_edges_from(edges)
    p = st.number_input("Enter the number of layers of the QAOA circuit:", min_value=1, max_value=8, value=1)
    mixer = st.text_input("Enter the mixer type (answer x,xx,y,yy or xy):", "x", max_chars=2)
    problem = P.Problems(G=G)
    return p, problem, G, mixer

def plot_circuit_and_graph(p, problem, mixer):
    st.header("Problem instance and associated QAOA circuit")
    st.write(f"**{'Graph instance'}**")
    plt.figure(figsize=(6,6))
    nx.draw(problem.G, with_labels=True, font_weight='bold', node_color='lightblue', edge_color='gray')
    st.pyplot(plt)
    qaoa = Q.Qaoa(p=p, G=problem, mixer=mixer)
    circuit_fig = circuit_drawer(qaoa.get_circuit(), output='mpl', style={'dpi': 300})
    st.write(f"QAOA circuit with given mixer {mixer}")
    st.pyplot(circuit_fig)

def get_partitions_from_solution(G, solution):
    partition_1 = [node for node, bit in enumerate(solution) if bit == '0']
    partition_2 = [node for node, bit in enumerate(solution) if bit == '1']
    return partition_1, partition_2

def plot_graph_partition(problem, p, mixer, x):
    st.write("MaxCut graph partition")
    plt.figure(figsize=(6,6))
    betas = x[:p]
    gammas = x[p:]
    init_point = list(betas) + list(gammas)
    qaoa = Q.Qaoa(p=p, G=problem, betas=betas, gammas=gammas, mixer=mixer)
    qc = qaoa.get_circuit()
    qc = qc.assign_parameters(init_point)
    backend = Aer.get_backend("aer_simulator")
    t_qc = transpile(qc, backend=backend)
    job = backend.run(t_qc)
    counts = job.result().get_counts(qc)
    most_frequent_solution = max(counts, key=counts.get)
    G = nx.Graph(qaoa.G.get_graph())
    partition_1, partition_2 = get_partitions_from_solution(G, most_frequent_solution)
    plt.figure(figsize=(6, 6))
    node_colors = ['lightblue' if node in partition_1 else 'orange' for node in G.nodes()]
    nx.draw(
        G,
        with_labels=True,
        font_weight='bold',
        node_color=node_colors,
        edge_color='gray'
    )
    st.pyplot(plt)
    st.write("Spectrum of the solutions")
    st.write(plot_histogram(counts))

def solve_maxcut(p, problem, mixer):
    st.header("Solving MaxCut with QAOA...")
    betas = qaoa_utils.generate_parameters(n=p, k=1)
    gammas = qaoa_utils.generate_parameters(n=p, k=2)
    qaoa = Q.Qaoa(p=p, G=problem, betas=betas, gammas=gammas, mixer=mixer)
    x, f = optims.simple_optimization(qaoa)
    st.write(f"Inital QAOA angles: [{betas[:p]} {gammas[:p]}]")
    st.write(f"Approximate MaxCut Value: {-f}")
    st.write(f"Updated QAOA angles: [{x[:p]} {x[p:]}]")
    plot_graph_partition(problem, p, mixer, x)
    
def main():
    st.title("MaxCut Problem Solver with QAOA")
    p, problem, G, mixer = create_instance()
    if len(problem.G.edges) > 0:
        plot_circuit_and_graph(p,problem, mixer)
        if st.button("Solve MaxCut"):
            solve_maxcut(p,problem,mixer)

if __name__ == "__main__":
    main()
