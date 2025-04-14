module SerializedArraysLinearAlgebraExt

using LinearAlgebra: LinearAlgebra, mul!
using SerializedArrays: SerializedArray

function LinearAlgebra.mul!(
  a_dest::AbstractMatrix, a1::SerializedArray, a2::SerializedArray, α::Number, β::Number
)
  mul!(a_dest, copy(a1), copy(a2), α, β)
  return a_dest
end

end
