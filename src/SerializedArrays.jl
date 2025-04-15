module SerializedArrays

using Base.PermutedDimsArrays: genperm
using ConstructionBase: constructorof
using DiskArrays: DiskArrays, AbstractDiskArray, Unchunked, readblock!, writeblock!
using Serialization: deserialize, serialize

memory(a) = a

#
# AbstractSerializedArray
#

abstract type AbstractSerializedArray{T,N} <: AbstractDiskArray{T,N} end
const AbstractSerializedMatrix{T} = AbstractSerializedArray{T,2}
const AbstractSerializedVector{T} = AbstractSerializedArray{T,1}

memory(a::AbstractSerializedArray) = copy(a)
disk(a::AbstractSerializedArray) = a

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
  return copyto!(dst, memory(src))
end
# Fix ambiguity error.
function Base.copyto!(dst::AbstractDiskArray, src::AbstractSerializedArray)
  return copyto!(dst, memory(src))
end
# Fix ambiguity error.
function Base.copyto!(dst::AbstractSerializedArray, src::AbstractDiskArray)
  return _copyto_write!(dst, src)
end
# Fix ambiguity error.
function Base.copyto!(dst::PermutedDimsArray, src::AbstractSerializedArray)
  return _copyto_read!(dst, src)
end

equals_serialized(a1, a2) = memory(a1) == memory(a2)

function Base.:(==)(a1::AbstractSerializedArray, a2::AbstractSerializedArray)
  return equals_serialized(a1, a2)
end
function Base.:(==)(a1::AbstractArray, a2::AbstractSerializedArray)
  return equals_serialized(a1, a2)
end
function Base.:(==)(a1::AbstractSerializedArray, a2::AbstractArray)
  return equals_serialized(a1, a2)
end

# # These cause too many ambiguity errors, try bringing them back.
# function Base.convert(arrayt::Type{<:AbstractSerializedArray}, a::AbstractArray)
#   return arrayt(a)
# end
# function Base.convert(arrayt::Type{<:AbstractArray}, a::AbstractSerializedArray)
#   return convert(arrayt, memory(a))
# end
# # Fixes ambiguity error.
# function Base.convert(arrayt::Type{<:Array}, a::AbstractSerializedArray)
#   return convert(arrayt, memory(a))
# end

#
# SerializedArray
#

struct SerializedArray{T,N,A<:AbstractArray{T,N},Axes} <: AbstractSerializedArray{T,N}
  file::String
  axes::Axes
end
file(a::SerializedArray) = getfield(a, :file)
Base.axes(a::SerializedArray) = getfield(a, :axes)
arraytype(a::SerializedArray{<:Any,<:Any,A}) where {A} = A

disk(a::AbstractArray) = SerializedArray(a)

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

# DiskArrays interface
DiskArrays.haschunks(::SerializedArray) = Unchunked()
function DiskArrays.readblock!(
  a::SerializedArray{<:Any,N}, aout, i::Vararg{AbstractUnitRange,N}
) where {N}
  if i == axes(a)
    aout .= memory(a)
    return a
  end
  aout .= @view memory(a)[i...]
  return a
end
function DiskArrays.writeblock!(
  a::SerializedArray{<:Any,N}, ain, i::Vararg{AbstractUnitRange,N}
) where {N}
  if i == axes(a)
    serialize(file(a), ain)
    return a
  end
  a′ = memory(a)
  a′[i...] = ain
  serialize(file(a), a′)
  return a
end
function DiskArrays.create_outputarray(::Nothing, a::SerializedArray, output_size::Tuple)
  return similar(a, output_size)
end

#
# PermutedSerializedArray
#

struct PermutedSerializedArray{T,N,P<:PermutedDimsArray{T,N}} <:
       AbstractSerializedArray{T,N}
  permuted_parent::P
end
Base.parent(a::PermutedSerializedArray) = parent(getfield(a, :permuted_parent))

file(a::PermutedSerializedArray) = file(parent(a))

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
  return PermutedDimsArray(memory(parent(a)), perm(a))
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
  readblock!(parent(a), PermutedDimsArray(aout, ip), inew...)
  return nothing
end
function DiskArrays.writeblock!(a::PermutedSerializedArray, v, i::OrdinalRange...)
  ip = iperm(a)
  inew = genperm(i, ip)
  # Permute the dest block and write from the true parent
  writeblock!(parent(a), PermutedDimsArray(v, ip), inew...)
  return nothing
end

#
# ReshapedSerializedArray
#

struct ReshapedSerializedArray{T,N,P<:AbstractArray{T},Axes} <: AbstractSerializedArray{T,N}
  parent::P
  axes::Axes
end
Base.parent(a::ReshapedSerializedArray) = getfield(a, :parent)
Base.axes(a::ReshapedSerializedArray) = getfield(a, :axes)

file(a::ReshapedSerializedArray) = file(parent(a))

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
  a′ = reshape(memory(parent(a)), axes(a))
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
    aout .= memory(a)
    return a
  end
  aout .= @view memory(a)[i...]
  return nothing
end
function DiskArrays.writeblock!(
  a::ReshapedSerializedArray{<:Any,N}, ain, i::Vararg{AbstractUnitRange,N}
) where {N}
  if i == axes(a)
    serialize(file(a), ain)
    return a
  end
  a′ = memory(a)
  a′[i...] = ain
  serialize(file(a), a′)
  return nothing
end

#
# SubSerializedArray
#

struct SubSerializedArray{T,N,P,I,L} <: AbstractSerializedArray{T,N}
  sub_parent::SubArray{T,N,P,I,L}
end

file(a::SubSerializedArray) = file(parent(a))

# Base methods
function Base.view(a::SerializedArray, i...)
  return SubSerializedArray(SubArray(a, Base.to_indices(a, i)))
end
function Base.view(a::SerializedArray, i::CartesianIndices)
  return SubSerializedArray(SubArray(a, Base.to_indices(a, i)))
end
Base.view(a::SubSerializedArray, i...) = SubSerializedArray(view(a.sub_parent, i...))
Base.view(a::SubSerializedArray, i::CartesianIndices) = view(a, i.indices...)
Base.size(a::SubSerializedArray) = size(a.sub_parent)
Base.axes(a::SubSerializedArray) = axes(a.sub_parent)
Base.parent(a::SubSerializedArray) = parent(a.sub_parent)
Base.parentindices(a::SubSerializedArray) = parentindices(a.sub_parent)

function materialize(a::SubSerializedArray)
  return view(copy(parent(a)), parentindices(a)...)
end
function Base.copy(a::SubSerializedArray)
  return copy(materialize(a))
end

DiskArrays.haschunks(a::SubSerializedArray) = Unchunked()
function DiskArrays.readblock!(a::SubSerializedArray, aout, i::OrdinalRange...)
  if i == axes(a)
    aout .= memory(a)
  end
  aout[i...] = memory(view(a, i...))
  return nothing
end
function DiskArrays.writeblock!(a::SubSerializedArray, ain, i::OrdinalRange...)
  if i == axes(a)
    serialize(file(a), ain)
    return a
  end
  a_parent = memory(parent(a))
  pinds = parentindices(view(a.sub_parent, i...))
  a_parent[pinds...] = ain
  serialize(file(a), a_parent)
  return nothing
end

#
# TransposeSerializedArray
#

struct TransposeSerializedArray{T,P<:AbstractSerializedArray{T}} <:
       AbstractSerializedMatrix{T}
  parent::P
end
Base.parent(a::TransposeSerializedArray) = getfield(a, :parent)

file(a::TransposeSerializedArray) = file(parent(a))

Base.axes(a::TransposeSerializedArray) = reverse(axes(parent(a)))
Base.size(a::TransposeSerializedArray) = length.(axes(a))

function Base.transpose(a::AbstractSerializedArray)
  return TransposeSerializedArray(a)
end
Base.transpose(a::TransposeSerializedArray) = parent(a)

function Base.similar(a::TransposeSerializedArray, elt::Type, dims::Tuple{Vararg{Int}})
  return similar(parent(a), elt, dims)
end

function materialize(a::TransposeSerializedArray)
  return transpose(memory(parent(a)))
end
function Base.copy(a::TransposeSerializedArray)
  return copy(materialize(a))
end

haschunks(a::TransposeSerializedArray) = Unchunked()
function DiskArrays.readblock!(a::TransposeSerializedArray, aout, i::OrdinalRange...)
  readblock!(parent(a), transpose(aout), reverse(i)...)
  return nothing
end
function DiskArrays.writeblock!(a::TransposeSerializedArray, ain, i::OrdinalRange...)
  writeblock!(parent(a), transpose(aout), reverse(i)...)
  return nothing
end

#
# AdjointSerializedArray
#

struct AdjointSerializedArray{T,P<:AbstractSerializedArray{T}} <:
       AbstractSerializedMatrix{T}
  parent::P
end
Base.parent(a::AdjointSerializedArray) = getfield(a, :parent)

file(a::AdjointSerializedArray) = file(parent(a))

Base.axes(a::AdjointSerializedArray) = reverse(axes(parent(a)))
Base.size(a::AdjointSerializedArray) = length.(axes(a))

function Base.adjoint(a::AbstractSerializedArray)
  return AdjointSerializedArray(a)
end
Base.adjoint(a::AdjointSerializedArray) = parent(a)
Base.adjoint(a::TransposeSerializedArray{<:Real}) = parent(a)
Base.transpose(a::AdjointSerializedArray{<:Real}) = parent(a)

function Base.similar(a::AdjointSerializedArray, elt::Type, dims::Tuple{Vararg{Int}})
  return similar(parent(a), elt, dims)
end

function materialize(a::AdjointSerializedArray)
  return adjoint(memory(parent(a)))
end
function Base.copy(a::AdjointSerializedArray)
  return copy(materialize(a))
end

haschunks(a::AdjointSerializedArray) = Unchunked()
function DiskArrays.readblock!(a::AdjointSerializedArray, aout, i::OrdinalRange...)
  readblock!(parent(a), adjoint(aout), reverse(i)...)
  return nothing
end
function DiskArrays.writeblock!(a::AdjointSerializedArray, ain, i::OrdinalRange...)
  writeblock!(parent(a), adjoint(aout), reverse(i)...)
  return nothing
end

#
# Broadcast
#

using Base.Broadcast:
  BroadcastStyle, Broadcasted, DefaultArrayStyle, combine_styles, flatten

struct SerializedArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
function Base.BroadcastStyle(arrayt::Type{<:AbstractSerializedArray})
  SerializedArrayStyle{ndims(arrayt)}()
end
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
  return copy(Base.Broadcast.broadcasted(a.broadcasted.f, memory.(a.broadcasted.args)...))
end

function Base.copy(broadcasted::Broadcasted{SerializedArrayStyle{N}}) where {N}
  return BroadcastSerializedArray(flatten(broadcasted))
end

end
