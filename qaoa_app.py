import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from classes import Qaoa as Q
from classes import Problems as P
from functions import qaoa_utilities as qaoa_utils
from functions import maxcut_utilities as mcut_utils
from functions import qaoa_optimizers as optims
from qiskit.visualization import plot_histogram
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
    plt.figure(figsize=(6,6))
    nx.draw(problem.G, with_labels=True, font_weight='bold', node_color='lightblue', edge_color='gray')
    st.pyplot(plt)
    qaoa = Q.Qaoa(p=p, G=problem, mixer=mixer)
    st.write(qaoa.get_circuit())

def plot_graph_partition(problem, p, mixer, x):
    st.header("MaxCut graph partition and solutions' spectrum")
    plt.figure(figsize=(6,6))
    betas = x[:p]
    gammas = x[p:]
    init_point = list(betas) + list(gammas)
    qaoa = Q.Qaoa(p=p, G=problem, betas=betas, gammas=gammas, mixer=mixer)
    G = qaoa.G
    qc = qaoa.get_circuit()
    qc = qc.assign_parameters(init_point)
    t_qc = transpile(qc, backend=Aer.get_backend("aer_simulator"))
    job = backend.run(t_qc, shots=shots)
    counts = job.result().get_counts(qc)
    st.write(counts)

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
