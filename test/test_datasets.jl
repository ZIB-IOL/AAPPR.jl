using Test

include("../src/datasets.jl")

@testset "Test suite for loadsnap64" begin
    (g, A, D) = loadsnap64(:facebook_combined)
    
    # Tests that the graph is connected
    @test is_connected(g) == true
    
    # Tests that the adjacency matrix is symmetric
    @test A == A'

    # Tests that the degree matrix is diagonal
    @test size(D) == size(A) 

    # Tests that the adjacency matrix is sparse
    @test typeof(A) == SparseMatrixCSC{Float64,Int64} 

    # Tests that the degree matrix is sparse 
    @test typeof(D) == SparseMatrixCSC{Float64,Int64} 
end