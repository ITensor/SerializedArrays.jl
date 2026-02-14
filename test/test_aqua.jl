using Aqua: Aqua
using SerializedArrays: SerializedArrays
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(SerializedArrays)
end
