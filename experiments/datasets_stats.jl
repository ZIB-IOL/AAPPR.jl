include("../src/datasets.jl")
include("../src/m_matrices.jl")
graphs = [:facebook_combined, :as_caida, :amazon0302, :ca_astroph, :ca_condmat, :ego_twitter_u, :patent_cit_us]
for graph in graphs
    g, adj_matrix, deg_matrix = loadsnap64(graph)
    num_vertices = size(adj_matrix, 1)
    num_edges = Int(sum(adj_matrix) ÷ 2)

    println("graph: $(graph)")
    println("number of vertices: $num_vertices")
    println("number of edges: $num_edges")
end