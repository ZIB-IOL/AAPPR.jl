using Test

println("running tests...")
println("-----------------------------------------------------------")
start_time = time()

tests = filter(x -> startswith(x, "test_"), readdir("test"))
@testset verbose = true "Test suite for AAPPR" begin
    for test in tests
        include(test)
    end
end
elapsed_time = time() - start_time

println("-----------------------------------------------------------")
println("tests done")
println("elapsed_time = $(round(elapsed_time, digits=3)) seconds")