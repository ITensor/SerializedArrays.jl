module SerializedArraysLinearAlgebraExt

using LinearAlgebra: LinearAlgebra, mul!
using SerializedArrays: AbstractSerializedMatrix, memory

function mul_serialized!(
  a_dest::AbstractMatrix, a1::AbstractMatrix, a2::AbstractMatrix, α::Number, β::Number
)
  mul!(a_dest, memory(a1), memory(a2), α, β)
  return a_dest
end

function LinearAlgebra.mul!(
  a_dest::AbstractMatrix,
  a1::AbstractSerializedMatrix,
  a2::AbstractSerializedMatrix,
  α::Number,
  β::Number,
)
  return mul_serialized!(a_dest, a1, a2, α, β)
end

function LinearAlgebra.mul!(
  a_dest::AbstractMatrix,
  a1::AbstractMatrix,
  a2::AbstractSerializedMatrix,
  α::Number,
  β::Number,
)
  return mul_serialized!(a_dest, a1, a2, α, β)
end

function LinearAlgebra.mul!(
  a_dest::AbstractMatrix,
  a1::AbstractSerializedMatrix,
  a2::AbstractMatrix,
  α::Number,
  β::Number,
)
  return mul_serialized!(a_dest, a1, a2, α, β)
end

for f in [:eigen, :qr, :svd]
  @eval begin
    function LinearAlgebra.$f(a::AbstractSerializedMatrix; kwargs...)
      return LinearAlgebra.$f(copy(a))
    end
  end
end

end
