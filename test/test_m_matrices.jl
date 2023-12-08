using Test
using Arpack, LinearAlgebra, SparseArrays

include("../src/m_matrices.jl")


@testset "Test suite for is_m_matrix" begin
    A_square = [1 -2; -1 0; -0 -1]
    A_diagonal = [1 0; 0 0]
    A_off_diagonal = [1 0; -0.0001 1]
    A_symmetric = [1 -0.02; -0.03 1]
    A_eigenvalues = [1 -2; -2 1]
    A_true = [4 -1 -1; -1 4 -1; -1 -1 4]

    @testset "Tests for dense matrices" begin
        @test is_m_matrix(A_square) == false
        @test is_m_matrix(A_diagonal) == false
        @test is_m_matrix(A_off_diagonal) == false
        @test is_m_matrix(A_symmetric) == false
        @test is_m_matrix(A_eigenvalues) == false
        @test is_m_matrix(A_true) == true
    end

    @testset "Tests for sparse matrices" begin
        @test is_m_matrix(sparse(A_square)) == false
        @test is_m_matrix(sparse(A_diagonal)) == false
        @test is_m_matrix(sparse(A_off_diagonal)) == false
        @test is_m_matrix(sparse(A_symmetric)) == false
        @test is_m_matrix(sparse(A_eigenvalues)) == false
        @test is_m_matrix(sparse(A_true)) == true
    end
end