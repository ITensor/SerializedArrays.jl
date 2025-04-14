module SerializedArrays

using Base.PermutedDimsArrays: genperm
using ConstructionBase: constructorof
using DiskArrays: DiskArrays, AbstractDiskArray, Unchunked, readblock!, writeblock!
using Serialization: deserialize, serialize

abstract type AbstractSerializedArray{T,N} <: AbstractDiskArray{T,N} end
const AbstractSerializedMatrix{T} = AbstractSerializedArray{T,2}
const AbstractSerializedVector{T} = AbstractSerializedArray{T,1}

function _copyto_write!(dst, src)
  writeblock!(dst, src, axes(src)...)
  return dst
end
function _copyto_read!(dst, src)
  readblock!(src, dst, axes(src)...)
  return dst
end

function Base.copyto!(dst::AbstractSerializedArray, src::AbstractArray)
  return _copyto_write!(dst, src)
end
function Base.copyto!(dst::AbstractArray, src::AbstractSerializedArray)
  return _copyto_read!(dst, src)
end
# Fix ambiguity error.
function Base.copyto!(dst::AbstractSerializedArray, src::AbstractSerializedArray)
  return copyto!(dst, copy(src))
end
# Fix ambiguity error.
function Base.copyto!(dst::AbstractDiskArray, src::AbstractSerializedArray)
  return copyto!(dst, copy(src))
end
# Fix ambiguity error.
function Base.copyto!(dst::AbstractSerializedArray, src::AbstractDiskArray)
  return _copyto_write!(dst, src)
end
# Fix ambiguity error.
function Base.copyto!(dst::PermutedDimsArray, src::AbstractSerializedArray)
  return _copyto_read!(dst, src)
end

function Base.:(==)(a1::AbstractSerializedArray, a2::AbstractSerializedArray)
  return copy(a1) == copy(a2)
end
function Base.:(==)(a1::AbstractArray, a2::AbstractSerializedArray)
  return a1 == copy(a2)
end
function Base.:(==)(a1::AbstractSerializedArray, a2::AbstractArray)
  return copy(a1) == a2
end

# # These cause too many ambiguity errors, try bringing them back.
# function Base.convert(arrayt::Type{<:AbstractSerializedArray}, a::AbstractArray)
#   return arrayt(a)
# end
# function Base.convert(arrayt::Type{<:AbstractArray}, a::AbstractSerializedArray)
#   return convert(arrayt, copy(a))
# end
# # Fixes ambiguity error.
# function Base.convert(arrayt::Type{<:Array}, a::AbstractSerializedArray)
#   return convert(arrayt, copy(a))
# end

struct SerializedArray{T,N,A<:AbstractArray{T,N},Axes} <: AbstractSerializedArray{T,N}
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

function Base.convert(arrayt::Type{<:SerializedArray}, a::AbstractArray)
  return arrayt(a)
end

function Base.similar(a::SerializedArray, elt::Type, dims::Tuple{Vararg{Int}})
  return constructorof(arraytype(a)){elt}(undef, dims...)
end

function materialize(a::SerializedArray)
  return deserialize(file(a))::arraytype(a)
end
function Base.copy(a::SerializedArray)
  return materialize(a)
end

Base.size(a::SerializedArray) = length.(axes(a))

to_axis(r::AbstractUnitRange) = r
to_axis(d::Integer) = Base.OneTo(d)

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

struct PermutedSerializedArray{T,N,P<:PermutedDimsArray{T,N}} <:
       AbstractSerializedArray{T,N}
  permuted_parent::P
end
Base.parent(a::PermutedSerializedArray) = parent(getfield(a, :permuted_parent))

perm(a::PermutedSerializedArray) = perm(a.permuted_parent)
perm(::PermutedDimsArray{<:Any,<:Any,p}) where {p} = p

iperm(a::PermutedSerializedArray) = iperm(a.permuted_parent)
iperm(::PermutedDimsArray{<:Any,<:Any,<:Any,ip}) where {ip} = ip

Base.axes(a::PermutedSerializedArray) = genperm(axes(parent(a)), perm(a))
Base.size(a::PermutedSerializedArray) = length.(axes(a))

function PermutedSerializedArray(a::AbstractArray, perm)
  a′ = PermutedDimsArray(a, perm)
  return PermutedSerializedArray{eltype(a),ndims(a),typeof(a′)}(a′)
end

function Base.permutedims(a::AbstractSerializedArray, perm)
  return PermutedSerializedArray(a, perm)
end

function Base.similar(a::PermutedSerializedArray, elt::Type, dims::Tuple{Vararg{Int}})
  return similar(parent(a), elt, dims)
end

function materialize(a::PermutedSerializedArray)
  return PermutedDimsArray(copy(parent(a)), perm(a))
end
function Base.copy(a::PermutedSerializedArray)
  return copy(materialize(a))
end

haschunks(a::PermutedSerializedArray) = Unchunked()
function DiskArrays.readblock!(a::PermutedSerializedArray, aout, i::OrdinalRange...)
  ip = iperm(a)
  # Permute the indices
  inew = genperm(i, ip)
  # Permute the dest block and read from the true parent
  DiskArrays.readblock!(parent(a), PermutedDimsArray(aout, ip), inew...)
  return nothing
end
function DiskArrays.writeblock!(a::PermutedSerializedArray, v, i::OrdinalRange...)
  ip = iperm(a)
  inew = genperm(i, ip)
  # Permute the dest block and write from the true parent
  DiskArrays.writeblock!(parent(a), PermutedDimsArray(v, ip), inew...)
  return nothing
end

struct ReshapedSerializedArray{T,N,P<:AbstractArray{T},Axes} <: AbstractSerializedArray{T,N}
  parent::P
  axes::Axes
end
Base.parent(a::ReshapedSerializedArray) = getfield(a, :parent)
Base.axes(a::ReshapedSerializedArray) = getfield(a, :axes)

function ReshapedSerializedArray(
  a::AbstractSerializedArray,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
)
  return ReshapedSerializedArray{eltype(a),length(ax),typeof(a),typeof(ax)}(a, ax)
end
function ReshapedSerializedArray(
  a::AbstractSerializedArray,
  shape::Tuple{
    Union{Integer,AbstractUnitRange{<:Integer}},
    Vararg{Union{Integer,AbstractUnitRange{<:Integer}}},
  },
)
  return ReshapedSerializedArray(a, to_axis.(shape))
end

Base.size(a::ReshapedSerializedArray) = length.(axes(a))

function Base.similar(a::ReshapedSerializedArray, elt::Type, dims::Tuple{Vararg{Int}})
  return similar(parent(a), elt, dims)
end

function materialize(a::ReshapedSerializedArray)
  return reshape(materialize(parent(a)), axes(a))
end
function Base.copy(a::ReshapedSerializedArray)
  a′ = materialize(a)
  return a′ isa Base.ReshapedArray ? copy(a′) : a′
end

# Special case for handling nested wrappers that aren't
# friendly on GPU. Consider special cases of strded arrays
# and handle with stride manipulations.
function Base.copy(a::ReshapedSerializedArray{<:Any,<:Any,<:PermutedSerializedArray})
  a′ = reshape(copy(parent(a)), axes(a))
  return a′ isa Base.ReshapedArray ? copy(a′) : a′
end

function Base.reshape(a::AbstractSerializedArray, dims::Tuple{Int,Vararg{Int}})
  return ReshapedSerializedArray(a, dims)
end

DiskArrays.haschunks(a::ReshapedSerializedArray) = Unchunked()
function DiskArrays.readblock!(
  a::ReshapedSerializedArray{<:Any,N}, aout, i::Vararg{AbstractUnitRange,N}
) where {N}
  if i == axes(a)
    aout .= copy(a)
    return a
  end
  aout .= @view copy(a)[i...]
  return nothing
end
function DiskArrays.writeblock!(
  a::ReshapedSerializedArray{<:Any,N}, ain, i::Vararg{AbstractUnitRange,N}
) where {N}
  if i == axes(a)
    serialize(file(a), ain)
    return a
  end
  a′ = copy(a)
  a′[i...] = ain
  serialize(file(a), a′)
  return nothing
end

#
# Broadcast
#

using Base.Broadcast:
  BroadcastStyle, Broadcasted, DefaultArrayStyle, combine_styles, flatten

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

struct BroadcastSerializedArray{T,N,BC<:Broadcasted{<:SerializedArrayStyle{N}}} <:
       AbstractSerializedArray{T,N}
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

function Base.copy(broadcasted::Broadcasted{SerializedArrayStyle{N}}) where {N}
  return BroadcastSerializedArray(flatten(broadcasted))
end

end
