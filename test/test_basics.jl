using GPUArraysCore: @allowscalar
using JLArrays: JLArray
using SerializedArrays: PermutedSerializedArray, ReshapedSerializedArray, SerializedArray
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

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  b = similar(a)
  @test b isa arrayt{elt,2}
  @test size(b) == size(a) == size(x)

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = permutedims(SerializedArray(x), (2, 1))
  @test a isa PermutedSerializedArray{elt,2}
  @test similar(a) isa arrayt{elt,2}
  @test copy(a) == permutedims(x, (2, 1))

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = reshape(SerializedArray(x), 16)
  @test a isa ReshapedSerializedArray{elt,1}
  @test similar(a) isa arrayt{elt,1}
  @test copy(a) == reshape(x, 16)

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = reshape(permutedims(SerializedArray(x), (2, 1)), 16)
  @test a isa ReshapedSerializedArray{elt,1,<:PermutedSerializedArray{elt,2}}
  @test similar(a) isa arrayt{elt,1}
  @test copy(a) == reshape(permutedims(x, (2, 1)), 16)

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  @test a == a
  @test x == a
  @test a == x

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  y = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  b = SerializedArray(y)
  copyto!(b, a)
  @test b == a
  @test b == x

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  y = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  b = SerializedArray(y)
  copyto!(b, x)
  @test b == a

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  y = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  copyto!(y, a)
  b = SerializedArray(y)
  @test b == a
end
