using GPUArraysCore: @allowscalar
using JLArrays: JLArray
using SerializedArrays: SerializedArray
using StableRNGs: StableRNG
using Test: @test, @testset
using TestExtras: @constinferred

elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
arrayts = (Array, JLArray)
@testset "SerializedArrays (eltype=$elt, arraytype=$arrayt)" for elt in elts,
  arrayt in arrayts

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  @test @constinferred(copy(a)) == x
  @test typeof(copy(a)) == typeof(x)

  x = arrayt(zeros(elt, 4, 4))
  a = SerializedArray(x)
  @allowscalar begin
    a[1, 1] = 2
    @test @constinferred(a[1, 1]) == 2
  end

  x = arrayt(zeros(elt, 4, 4))
  a = SerializedArray(x)
  b = 2a
  @test @constinferred(copy(b)) == 2x

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  y = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  b = SerializedArray(y)
  c = @constinferred(a * b)
  @test c == x * y
  @test c isa arrayt{elt,2}
end
