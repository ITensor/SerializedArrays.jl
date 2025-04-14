module SerializedArraysLinearAlgebraExt

using LinearAlgebra: LinearAlgebra, mul!
using SerializedArrays: AbstractSerializedMatrix

function LinearAlgebra.mul!(
  a_dest::AbstractMatrix,
  a1::AbstractSerializedMatrix,
  a2::AbstractSerializedMatrix,
  α::Number,
  β::Number,
)
  mul!(a_dest, copy(a1), copy(a2), α, β)
  return a_dest
end

for f in [:eigen, :qr, :svd]
  @eval begin
    function LinearAlgebra.$f(a::AbstractSerializedMatrix; kwargs...)
      return LinearAlgebra.$f(copy(a))
    end
  end
end

end
