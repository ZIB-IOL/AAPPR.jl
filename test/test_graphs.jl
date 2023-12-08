using Test
using Graphs, LinearAlgebra, SparseArrays
include("../src/graphs.jl")


function test_create_connected_graph(n::Int64, edge_probabilty::Float64)
    # Tests that the graph is connected
    adj_matrix, deg_matrix = create_connected_graph(n, edge_probabilty)
    cc = connected_components(Graph(adj_matrix))
    @test length(cc) == 1
    
    # Tests that the adjacency matrix is symmetric
    @test adj_matrix == adj_matrix'
    
    # Tests that the degree matrix is diagonal
    @test size(deg_matrix) == size(adj_matrix)
    @test sum(deg_matrix .* ones(size(deg_matrix))) == sum(adj_matrix)
end

@testset "Test suite for create_connected_graph" begin
    @testset "Tests for graph properties" begin
        test_create_connected_graph(10, 0.1)
        test_create_connected_graph(50, 0.9)
        test_create_connected_graph(100, 0.0000000001)
    end
end






function test_get_neighborhood_indices()
    # Create test adjacency matrix and test indices
    mat = sparse([1 1 0 0; 1 0 1 1; 0 1 0 1; 0 1 1 0])

    @test_throws ArgumentError get_neighborhood_indices([5], mat)

    # Tests a simple example
    indices = [1]
    neighborhood_indices = get_neighborhood_indices(indices, mat)
    @test sort(neighborhood_indices) == [1, 2]

    indices = [2]
    neighborhood_indices = get_neighborhood_indices(indices, mat)
    @test sort(neighborhood_indices) == [1, 2, 3, 4]

    indices = [3]
    neighborhood_indices = get_neighborhood_indices(indices, mat)
    @test sort(neighborhood_indices) == [2, 3, 4]

    indices = [4]
    neighborhood_indices = get_neighborhood_indices(indices, mat)
    @test sort(neighborhood_indices) == [2, 3, 4]


    # Tests a more complex example
    mat = sparse([1 0 0 3 0 ; 2 0 1 3 0; 0 3 0 0 0; 0 4 0 0 0; 0 0 0 0 0])
    indices = [1, 2]
    neighborhood_indices = get_neighborhood_indices(indices, mat)
    @test sort(neighborhood_indices) == [1, 2, 3, 4]


    # Tests that the function works with Float64
    mat = sparse([1. 0.; 0. 1.])
    indices = [1]
    neighborhood_indices = get_neighborhood_indices(indices, mat)
    @test sort(neighborhood_indices) == [1]
end

# Run tests
@testset "Test suite for get_neighborhood_indices" begin
    test_get_neighborhood_indices()
end