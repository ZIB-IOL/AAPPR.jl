using SNAPDatasets, Graphs, SparseArrays, LinearAlgebra

"""
    loadsnap64(f)

Load a snap file into a simple graph data structure of type SimpleGraph{Int64}, returning the graph, the adjacency matrix, and the degree matrix..
"""
function loadsnap64(f)
    g_tmp = loadsnap(f)
    g = SimpleGraph{Int64}(nv(g_tmp))
    n = nv(g_tmp)
    for e in edges(g_tmp)
        n1 = Int64(src(e))
        n2 = Int64(dst(e))
        source = min(n1, n2)
        dest = max(n1, n2)
        if has_edge(g, source, dest) == false && source != dest
            add_edge!(g, source, dest)
        end
    end
    adj_matrix = dropzeros(map(Float64, sparse(adjacency_matrix(g))))
    deg_matrix = dropzeros(map(Float64, sparse(Diagonal(sum(adj_matrix, dims=2)[:,1]))))
    return g, adj_matrix, deg_matrix
end



