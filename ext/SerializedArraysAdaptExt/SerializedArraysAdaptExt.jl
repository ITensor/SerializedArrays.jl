module SerializedArraysAdaptExt

using Adapt: Adapt
using SerializedArrays: SerializedArray

function Adapt.adapt_storage(arrayt::Type{<:SerializedArray}, a::AbstractArray)
  return convert(arrayt, a)
end

end
