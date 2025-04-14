using Adapt: adapt
using JLArrays: JLArray
using SerializedArrays: SerializedArray
using StableRNGs: StableRNG
using Test: @test, @testset
using TestExtras: @constinferred

elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
arrayts = (Array, JLArray)
@testset "SerializedArraysAdaptExt (eltype=$elt, arraytype=$arrayt)" for elt in elts,
  arrayt in arrayts

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  y = PermutedDimsArray(x, (2, 1))
  a = adapt(SerializedArray, x)
  @test a isa SerializedArray{elt,2,arrayt{elt,2}}
  b = adapt(SerializedArray, y)
  @test b isa
    PermutedDimsArray{elt,2,(2, 1),(2, 1),<:SerializedArray{elt,2,<:arrayt{elt,2}}}
  @test parent(b) == a
end
