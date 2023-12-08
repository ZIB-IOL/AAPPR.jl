using Graphs, LinearAlgebra, SparseArrays

"""
    create_connected_graph(n::Int64, edge_probability::Float64)

Create a random connected graph with `n` nodes and edge probability `edge_probability`.

# Arguments
- `n::Int64`: Number of nodes in the graph.
- `edge_probability::Float64`: Probability of an edge between any two nodes.

# Returns
- `adj_matrix::SparseMatrixCSC{Float64, Int64}`: Sparse adjacency matrix of the graph.
- `deg_matrix::SparseMatrixCSC{Float64, Int64}`: Sparse diagonal matrix of the degrees of each node.
"""
function create_connected_graph(n::Int64, edge_probability::Float64)
    if edge_probability > 1 || edge_probability < 0
        throw(ArgumentError("edge_probability must be between 0 and 1"))
    end
    if n < 1
        throw(ArgumentError("n must be greater than 0"))
    end

    g = SimpleGraph(n)
    for i in 1:n, j in (i+1):n
        if rand() <= edge_probability
            add_edge!(g, i, j)
        end
    end
    
    while !is_connected(g)
        # select two different connected components at random
        i = j = rand(1:length(connected_components(g)))
        while j == i
            j = rand(1:length(connected_components(g)))
        end
        # select a random node in component i
        node_1 = rand(connected_components(g)[i])
        # select a random node in component j
        node_2 = rand(connected_components(g)[j])
        # connect the nodes
        add_edge!(g, node_1, node_2)
    end
    # make sure that the graph is connected
    @assert is_connected(g)
    # create the adjacency and degree matrices
    adj_matrix = map(Float64, sparse(adjacency_matrix(g)))
    deg_matrix = map(Float64, sparse(Diagonal(sum(adj_matrix, dims=2)[:,1])))
    return adj_matrix, deg_matrix
end

"""
    get_neighborhood_indices(indices::Union{AbstractVector{Any}, AbstractVector{Int64}}, adj_mat::Union{SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Int64, Int64}})

Get the neighborhood indices of a set of nodes in a graph represented by an adjacency matrix.

# Arguments
- `indices::Union{AbstractVector{Any}, AbstractVector{Int64}}`: A vector containing the indices of nodes for which we want to find the neighborhood. If `nothing`, the function returns all indices of non-zero entries in `adj_mat`. Default is `nothing`.
- `adj_mat::SparseMatrixCSC{Float64, Int64}`: Either the adjacency matrix of the graph, D*alpha*A*D2 + beta*I, where A is adj_mat, D, D2 are diagonal matrices, and I is the identity.

# Returns
- A vector of unique indices of the nodes in the neighborhood of the input nodes.
"""
function get_neighborhood_indices(indices::Union{AbstractVector{Any}, AbstractVector{Int64}}, adj_mat::Union{SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Int64, Int64}})
    if indices != [] && maximum(indices) > size(adj_mat, 1)
        throw(ArgumentError("The number of indices must be less than the number of nodes in the graph"))
    end

    
    neighborhood_indices = copy(indices)
    for idx in indices
        neighborhood_indices = vcat(findall(!iszero, view(adj_mat, :, idx)), neighborhood_indices)
    end
    @assert issubset(indices, neighborhood_indices)
    return sort!(unique!(neighborhood_indices))
end
