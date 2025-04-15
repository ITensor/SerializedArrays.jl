using GPUArraysCore: @allowscalar
using JLArrays: JLArray
using SerializedArrays:
  AdjointSerializedArray,
  PermutedSerializedArray,
  ReshapedSerializedArray,
  SerializedArray,
  SubSerializedArray,
  TransposeSerializedArray,
  disk,
  memory
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
  @test memory(a) == x
  @test memory(a) isa arrayt{elt,2}
  @test memory(x) === x
  @test disk(a) === a
  @test disk(x) == a
  @test disk(x) isa SerializedArray{elt,2,<:arrayt{elt,2}}

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
  @test copy(2a) == 2permutedims(x, (2, 1))

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = transpose(SerializedArray(x))
  @test a isa TransposeSerializedArray{elt}
  @test similar(a) isa arrayt{elt,2}
  @test copy(a) == transpose(x)
  @test copy(2a) == 2transpose(x)

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = adjoint(SerializedArray(x))
  @test a isa AdjointSerializedArray{elt}
  @test similar(a) isa arrayt{elt,2}
  @test copy(a) == adjoint(x)
  @test copy(2a) == 2adjoint(x)

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  a = SerializedArray(x)
  @test transpose(transpose(a)) === a
  @test adjoint(adjoint(a)) === a
  if isreal(a)
    @test adjoint(transpose(a)) === a
    @test transpose(adjoint(a)) === a
  end

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

  rng = StableRNG(123)
  x = arrayt(randn(rng, elt, 4, 4))
  y = @view x[2:3, 2:3]
  a = SerializedArray(a)
  b = @view a[2:3, 2:3]
  @test b isa SubSerializedArray{elt,2}
  c = 2b
  @test 2y == copy(c)
  @allowscalar begin
    b[1, 1] = 2
    @test @constinferred(b[1, 1]) == 2
  end
end
