using LinearAlgebra, Arpack, SparseArrays
using Test

include("../src/objectives.jl")

@testset "Test suite for APPRObjective" begin
    for n in [75, 100, 500]
        
        indices_50 = collect(1:50)
        indices_70 = collect(1:70)

        A = sprand(n, n,0.1)
        A = A - diagm(diag(A))
        A = sparse(A + A')

        d = round.(rand(n) * 99) .+ 1
        D = spdiagm(d)
        D_neg = sparse(D^(-1/2))
        D_pos = sparse(D^(1/2))

        alpha = 0.15
        rho = 0.15

        Q = sparse(D_neg*(D .- (1-alpha)/2 * (D .+ A))*D_neg)

        s = spzeros(n)
        s[1] = 1.0
        grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)

        x = sprand(Float64, n, 0.3)
        L = maximum(eigs(Q, nev=1, which=:LM)[1])
        mu = minimum(eigs(Q, nev=1, which=:SM)[1])
        grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2)
        obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)

        Q = Symmetric(Q)
        Q = set_nonindex_entries_to_zero(obj.Q, indices_50)
        s = set_nonindex_entries_to_zero(obj.s, indices_50)
        grad_cst_component = set_nonindex_entries_to_zero(obj.grad_cst_component, indices_50)
        
        obj_copy_and_restrict_actual = APPRObjective(Q, s, obj.L, obj.mu, grad_cst_component, grad_norm_0)
        
        # check copy() and restrict!()
        obj_copy_and_restrict = copy(obj)
        restrict!(obj_copy_and_restrict, collect(1:n), indices_50)        
        @test obj_copy_and_restrict isa APPRObjective
        @test obj_copy_and_restrict.Q == set_nonindex_entries_to_zero(obj.Q, indices_50)
        @test obj_copy_and_restrict.s == set_nonindex_entries_to_zero(obj.s, indices_50)
        @test obj_copy_and_restrict.L == obj.L
        @test obj_copy_and_restrict.mu == obj.mu
        @test obj_copy_and_restrict.grad_cst_component == set_nonindex_entries_to_zero(obj.grad_cst_component, indices_50)
        @test abs(appr_objective(obj_copy_and_restrict, x) - appr_objective(obj_copy_and_restrict_actual, x)) < 1e-6
        @test norm(∇appr_objective(obj_copy_and_restrict, x) - ∇appr_objective(obj_copy_and_restrict_actual, x)) < 1e-6
        @test norm(∇appr_objective_cst(obj_copy_and_restrict) - ∇appr_objective_cst(obj_copy_and_restrict_actual)) < 1e-6
        @test norm(∇appr_objective_non_cst(obj_copy_and_restrict, x) - ∇appr_objective_non_cst(obj_copy_and_restrict_actual, x)) < 1e-6

        # check extend!()
        obj_50 = copy(obj_copy_and_restrict)
        extend!(obj_copy_and_restrict, obj, indices_50, indices_70)
        @test obj_copy_and_restrict isa APPRObjective
        @test obj_copy_and_restrict.Q == set_nonindex_entries_to_zero(obj.Q, indices_70)
        @test obj_copy_and_restrict.s == set_nonindex_entries_to_zero(obj.s, indices_70)
        @test obj_copy_and_restrict.L == obj.L
        @test obj_copy_and_restrict.mu == obj.mu
        @test obj_copy_and_restrict.grad_cst_component == set_nonindex_entries_to_zero(obj.grad_cst_component, indices_70)

        # check restrict!()
        obj_restricted = copy(obj)
        restrict!(obj_restricted, collect(1:n), indices_50)
        @test obj_restricted isa APPRObjective
        @test obj_restricted.Q == obj_50.Q
        @test obj_restricted.s == obj_50.s
        @test obj_restricted.L == obj_50.L
        @test obj_restricted.mu == obj_50.mu
        @test obj_restricted.grad_cst_component == obj_50.grad_cst_component
        @test abs(appr_objective(obj_restricted, x) - appr_objective(obj_50, x)) < 1e-6
        @test norm(∇appr_objective(obj_restricted, x) - ∇appr_objective(obj_50, x)) < 1e-6
        @test norm(∇appr_objective_cst(obj_restricted) - ∇appr_objective_cst(obj_50)) < 1e-6
        @test norm(∇appr_objective_non_cst(obj_restricted, x) - ∇appr_objective_non_cst(obj_50, x)) < 1e-6
    end
end
