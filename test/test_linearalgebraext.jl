using JLArrays: JLArray
using LinearAlgebra: eigen, qr, svd
using SerializedArrays: SerializedArray
using StableRNGs: StableRNG
using Test: @test, @testset
using TestExtras: @constinferred

elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
arrayts = (Array, JLArray)
@testset "SerializedArraysLinearAlgebraExt (eltype=$elt, arraytype=$arrayt)" for elt in
                                                                                 elts,
  arrayt in arrayts

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  y = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  b = SerializedArray(y)
  c = @constinferred(a * b)
  @test c == x * y
  @test c isa arrayt{elt,2}

  a = permutedims(SerializedArray(x), (2, 1))
  b = permutedims(SerializedArray(y), (2, 1))
  c = @constinferred(a * b)
  @test c == permutedims(x, (2, 1)) * permutedims(y, (2, 1))
  @test c isa arrayt{elt,2}

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  # `LinearAlgebra.eigen(::JLArray)` is broken with
  # a scalar indexing issue.
  if arrayt ≠ JLArray
    @test eigen(a) == eigen(x)
  end
  Q, R = qr(a)
  Qₓ, Rₓ = qr(x)
  @test Q == Qₓ
  @test R == Rₓ
  @test svd(a) == svd(x)
end
