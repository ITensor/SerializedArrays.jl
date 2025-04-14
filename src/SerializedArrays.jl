module SerializedArrays

using ConstructionBase: constructorof
using DiskArrays: DiskArrays, AbstractDiskArray, Unchunked
using LinearAlgebra: LinearAlgebra, mul!
using Serialization: deserialize, serialize

struct SerializedArray{T,N,A<:AbstractArray{T,N},Axes} <: AbstractDiskArray{T,N}
  file::String
  axes::Axes
end
file(a::SerializedArray) = getfield(a, :file)
Base.axes(a::SerializedArray) = getfield(a, :axes)
arraytype(a::SerializedArray{<:Any,<:Any,A}) where {A} = A

function SerializedArray(file::String, a::AbstractArray)
  serialize(file, a)
  ax = axes(a)
  return SerializedArray{eltype(a),ndims(a),typeof(a),typeof(ax)}(file, ax)
end
function SerializedArray(a::AbstractArray)
  return SerializedArray(tempname(), a)
end

function Base.similar(a::SerializedArray, elt::Type, dims::Tuple{Vararg{Int}})
  return constructorof(arraytype(a)){elt}(undef, dims...)
end

function Base.copy(a::SerializedArray)
  arrayt = arraytype(a)
  return convert(arrayt, deserialize(file(a)))::arrayt
end

Base.size(a::SerializedArray) = length.(axes(a))

#
# DiskArrays
#

DiskArrays.haschunks(::SerializedArray) = Unchunked()
function DiskArrays.readblock!(
  a::SerializedArray{<:Any,N}, aout, i::Vararg{AbstractUnitRange,N}
) where {N}
  if i == axes(a)
    aout .= copy(a)
    return a
  end
  aout .= @view copy(a)[i...]
  return a
end
function DiskArrays.writeblock!(
  a::SerializedArray{<:Any,N}, ain, i::Vararg{AbstractUnitRange,N}
) where {N}
  if i == axes(a)
    serialize(file(a), ain)
    return a
  end
  a′ = copy(a)
  a′[i...] = ain
  serialize(file(a), a′)
  return a
end
function DiskArrays.create_outputarray(::Nothing, a::SerializedArray, output_size::Tuple)
  return similar(a, output_size)
end

#
# Broadcast
#

using Base.Broadcast:
  BroadcastStyle, Broadcasted, DefaultArrayStyle, combine_styles, flatten

struct BroadcastSerializedArray{T,N,BC<:Broadcasted{<:SerializedArrayStyle{N}}} <:
       AbstractDiskArray{T,N}
  broadcasted::BC
end
function BroadcastSerializedArray(
  broadcasted::B
) where {B<:Broadcasted{<:SerializedArrayStyle{N}}} where {N}
  ElType = Base.Broadcast.combine_eltypes(broadcasted.f, broadcasted.args)
  return BroadcastSerializedArray{ElType,N,B}(broadcasted)
end
Base.size(a::BroadcastSerializedArray) = size(a.broadcasted)
Base.broadcastable(a::BroadcastSerializedArray) = a.broadcasted
function Base.copy(a::BroadcastSerializedArray)
  # Broadcast over the materialized arrays.
  return copy(Base.Broadcast.broadcasted(a.broadcasted.f, copy.(a.broadcasted.args)...))
end

struct SerializedArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
Base.BroadcastStyle(arrayt::Type{<:SerializedArray}) = SerializedArrayStyle{ndims(arrayt)}()
function Base.BroadcastStyle(
  ::SerializedArrayStyle{N}, ::SerializedArrayStyle{M}
) where {N,M}
  SerializedArrayStyle{max(N, M)}()
end
function Base.BroadcastStyle(::SerializedArrayStyle{N}, ::DefaultArrayStyle{M}) where {N,M}
  return SerializedArrayStyle{max(N, M)}()
end
function Base.BroadcastStyle(::DefaultArrayStyle{M}, ::SerializedArrayStyle{N}) where {N,M}
  return SerializedArrayStyle{max(N, M)}()
end
function Base.copy(broadcasted::Broadcasted{SerializedArrayStyle{N}}) where {N}
  return BroadcastSerializedArray(flatten(broadcasted))
end

#
# LinearAlgebra
#

function LinearAlgebra.mul!(
  a_dest::AbstractMatrix, a1::SerializedArray, a2::SerializedArray, α::Number, β::Number
)
  mul!(a_dest, copy(a1), copy(a2), α, β)
  return a_dest
end

end
