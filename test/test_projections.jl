using Test
using SparseArrays

include("../src/projections.jl")


@testset "Test suite for projection_onto_orthant" begin
    sparse_vector = sparse([-1.0, -2.0, 0.0, 4.0, 5.0])
    sparse_projected_vector = projection_onto_orthant(sparse_vector; indices=[1, 3, 5])
    @test sparse_projected_vector == sparse([0.0, 0.0, 0.0, 0.0, 5.0])
    sparse_vector = sparse([-0.5, -0.6, 0.0, 2.0, 3.0])
    x0 = sparse([1.0, 0.0, 3.0, -4.0, 1.0])
    sparse_projected_vector = projection_onto_orthant(sparse_vector; indices=[1, 2, 3, 4, 5], anchor=x0)
    @test sparse_projected_vector == sparse([1.0, 0.0, 3.0, 2.0, 3.0])
end


@testset "Test suite for set_nonindex_entries_to_zero and set_nonindex_entries_to_zero! for vectors" begin
    # Tests with empty indices and empty sparse vector
    x = sparse(Vector{Float64}(undef, 0))
    indices = []
    @test set_nonindex_entries_to_zero(x, indices) == x
    y = copy(x)
    set_nonindex_entries_to_zero!(y, indices)
    @test y == set_nonindex_entries_to_zero(x, indices)

    # Tests with empty indices and non-empty sparse vector
    x = sparsevec([1, 2, 3], [1., 2., 3.], 5)
    indices = []
    @test dropzeros!(set_nonindex_entries_to_zero(x, indices)) == sparse([0., 0., 0., 0., 0.])
    y = copy(x)
    set_nonindex_entries_to_zero!(y, indices)
    @test y == set_nonindex_entries_to_zero(x, indices)

    # Tests with non-empty indices and non-empty sparse vector
    x = sparsevec([1, 2, 3], [1., 2., 3.], 5)
    indices = [1, 3]
    @test set_nonindex_entries_to_zero(x, indices) == sparsevec([1, 3], [1., 3.], 5)
    y = copy(x)
    set_nonindex_entries_to_zero!(y, indices)
    @test y == set_nonindex_entries_to_zero(x, indices)

    # Tests with non-empty indices and non-empty sparse vector where some indices are already zero
    x = sparsevec([1, 2, 3, 5], [1., 2., 0., 4.], 6)
    indices = [1, 4, 6]
    @test dropzeros!(set_nonindex_entries_to_zero(x, indices)) == sparsevec([1, 4, 6], [1., 0., 0.], 6)
    y = copy(x)
    set_nonindex_entries_to_zero!(y, indices)
    @test y == set_nonindex_entries_to_zero(x, indices)
end

@testset "Test suite for set_nonindex_entries_to_zero and set_nonindex_entries_to_zero! for matrices" begin
    # Tests with empty indices and non-empty sparse matrix
    A = sparse([1. 2.; 3. 4.; 5. 6.])
    indices = []
    @test dropzeros(set_nonindex_entries_to_zero(A, indices)) == sparse([0. 0.; 0. 0.; 0. 0.])
    B = copy(A)
    set_nonindex_entries_to_zero!(B, indices)
    @test B == set_nonindex_entries_to_zero(A, indices)

    # Tests with non-empty indices and non-empty sparse matrix
    A = sparse([1. 2. 3.; 4. 5. 6.; 7. 8. 9.])
    indices = [1, 3]
    @test set_nonindex_entries_to_zero(A, indices) == sparse([1. 0. 3.; 0. 0. 0.; 7. 0. 9.])
    B = copy(A)
    set_nonindex_entries_to_zero!(B, indices)
    @test B == set_nonindex_entries_to_zero(A, indices)

    # # Tests with non-empty indices and non-empty sparse matrix where some entries are already zero
    A = sparse([1. 0. 3.; 4. 0. 6.; 0. 0. 0.])
    indices = [1, 3]
    @test set_nonindex_entries_to_zero(A, indices) == sparse([1. 0. 3.; 0. 0. 0.; 0. 0. 0.])
    B = copy(A)
    set_nonindex_entries_to_zero!(B, indices)
    @test B == set_nonindex_entries_to_zero(A, indices)
end


@testset "Test suite for reduce_and_clip and reduce_and_clip!" begin
    # Tests that the function works with a simple input
    x = sparsevec([1, 2, 3], [1.0, -2.0, 3.0])
    @test reduce_and_clip(x, 1.0) == sparsevec([1, 3], [0.0, 2.0])
    reduce_and_clip!(x, 1.0)
    @test x == sparsevec([1, 3], [0.0, 2.0])
    
    # Tests that the function works with an empty sparse vector
    x = sparsevec(Int64[], Float64[])
    @test reduce_and_clip(x, 0.5) == sparsevec(Int64[], Float64[])
    reduce_and_clip!(x, 0.5)
    @test x == sparsevec(Int64[], Float64[])
    
    # Tests that the function works with a vector with no negative entries
    x = sparsevec([1, 2, 3], [1.0, 2.0, 3.0])
    @test reduce_and_clip(x, 1.0) == sparsevec([1, 2, 3], [0.0, 1.0, 2.0])
    reduce_and_clip!(x, 1.0)
    @test x == sparsevec([1, 2, 3], [0.0, 1.0, 2.0])
end