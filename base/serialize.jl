# This file is a part of Julia. License is MIT: http://julialang.org/license

module Serializer

import Base: GMP, Bottom, svec, unsafe_convert, uncompressed_ast

export serialize, deserialize

## serializing values ##

# type SerializationState  # defined in dict.jl

const TAGS = Any[
    Symbol, Int8, UInt8, Int16, UInt16, Int32, UInt32,
    Int64, UInt64, Int128, UInt128, Float32, Float64, Char, Ptr,
    DataType, UnionType, Function,
    Tuple, Array, Expr,
    #LongSymbol, LongTuple, LongExpr,
    Symbol, Tuple, Expr,  # dummy entries, intentionally shadowed by earlier ones
    LineNumberNode, SymbolNode, LabelNode, GotoNode,
    QuoteNode, TopNode, TypeVar, Box, LambdaStaticData,
    Module, #=UndefRefTag=#Symbol, Task, ASCIIString, UTF8String,
    UTF16String, UTF32String, Float16,
    SimpleVector, #=BackrefTag=#Symbol, :reserved11, :reserved12,

    (), Bool, Any, :Any, Bottom, :reserved21, :reserved22, Type,
    :Array, :TypeVar, :Box,
    :lambda, :body, :return, :call, symbol("::"),
    :(=), :null, :gotoifnot, :A, :B, :C, :M, :N, :T, :S, :X, :Y,
    :a, :b, :c, :d, :e, :f, :g, :h, :i, :j, :k, :l, :m, :n, :o,
    :p, :q, :r, :s, :t, :u, :v, :w, :x, :y, :z,
    :add_int, :sub_int, :mul_int, :add_float, :sub_float,
    :mul_float, :unbox, :box,
    :eq_int, :slt_int, :sle_int, :ne_int,
    :arrayset, :arrayref,
    :Core, :Base, svec(), Tuple{},
    :reserved17, :reserved18, :reserved19, :reserved20,
    false, true, nothing, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32
]

const ser_version = 2 # do not make changes without bumping the version #!

const NTAGS = length(TAGS)

function sertag(v::ANY)
    ptr = pointer_from_objref(v)
    ptags = convert(Ptr{Ptr{Void}}, pointer(TAGS))
    @inbounds for i = 1:NTAGS
        ptr == unsafe_load(ptags,i) && return (i+1)%Int32
    end
    return Int32(-1)
end
desertag(i::Int32) = TAGS[i-1]

# tags >= this just represent themselves, their whole representation is 1 byte
const VALUE_TAGS = sertag(())
const ZERO_TAG = sertag(0)
const TRUE_TAG = sertag(true)
const FALSE_TAG = sertag(false)
const EMPTYTUPLE_TAG = sertag(())
const TUPLE_TAG = sertag(Tuple)
const LONGTUPLE_TAG = Int32(sertag(Expr)+2)
const SIMPLEVECTOR_TAG = sertag(SimpleVector)
const SYMBOL_TAG = sertag(Symbol)
const LONGSYMBOL_TAG = Int32(sertag(Expr)+1)
const ARRAY_TAG = sertag(Array)
const UNDEFREF_TAG = Int32(sertag(Module)+1)
const BACKREF_TAG = Int32(sertag(SimpleVector)+1)
const EXPR_TAG = sertag(Expr)
const LONGEXPR_TAG = Int32(sertag(Expr)+3)
const MODULE_TAG = sertag(Module)
const FUNCTION_TAG = sertag(Function)
const LAMBDASTATICDATA_TAG = sertag(LambdaStaticData)
const TASK_TAG = sertag(Task)
const DATATYPE_TAG = sertag(DataType)
const INT_TAG = sertag(Int)

writetag(s::IO, tag) = write(s, UInt8(tag))

function write_as_tag(s::IO, tag)
    tag < VALUE_TAGS && write(s, UInt8(0))
    write(s, UInt8(tag))
end

# cycle handling
function serialize_cycle(s::SerializationState, x)
    if !isimmutable(x) && !typeof(x).pointerfree
        offs = get(s.table, x, -1)
        if offs != -1
            writetag(s.io, BACKREF_TAG)
            write(s.io, Int(offs))
            return true
        end
        s.table[x] = s.counter
        s.counter += 1
    end
    return false
end

serialize(s::SerializationState, x::Bool) = x ? writetag(s.io, TRUE_TAG) :
                                                writetag(s.io, FALSE_TAG)

serialize(s::SerializationState, ::Ptr) = error("cannot serialize a pointer")

serialize(s::SerializationState, ::Tuple{}) = writetag(s.io, EMPTYTUPLE_TAG)

function serialize(s::SerializationState, t::Tuple)
    l = length(t)
    if l <= 255
        writetag(s.io, TUPLE_TAG)
        write(s.io, UInt8(l))
    else
        writetag(s.io, LONGTUPLE_TAG)
        write(s.io, Int32(l))
    end
    for i = 1:l
        serialize(s, t[i])
    end
end

function serialize(s::SerializationState, v::SimpleVector)
    writetag(s.io, SIMPLEVECTOR_TAG)
    write(s.io, Int32(length(v)))
    for i = 1:length(v)
        serialize(s.io, v[i])
    end
end

function serialize(s::SerializationState, x::Symbol)
    tag = sertag(x)
    if tag > 0
        return write_as_tag(s.io, tag)
    end
    pname = unsafe_convert(Ptr{UInt8}, x)
    ln = Int(ccall(:strlen, Csize_t, (Ptr{UInt8},), pname))
    if ln <= 255
        writetag(s.io, SYMBOL_TAG)
        write(s.io, UInt8(ln))
    else
        writetag(s.io, LONGSYMBOL_TAG)
        write(s.io, Int32(ln))
    end
    write(s.io, pname, ln)
end

function serialize_array_data(s::IO, a)
    elty = eltype(a)
    if elty === Bool && length(a)>0
        last = a[1]
        count = 1
        for i = 2:length(a)
            if a[i] != last || count == 127
                write(s, UInt8((UInt8(last)<<7) | count))
                last = a[i]
                count = 1
            else
                count += 1
            end
        end
        write(s, UInt8((UInt8(last)<<7) | count))
    else
        write(s, a)
    end
end

function serialize(s::SerializationState, a::Array)
    elty = eltype(a)
    !isbits(elty) && serialize_cycle(s, a) && return
    writetag(s.io, ARRAY_TAG)
    if elty !== UInt8
        serialize(s, elty)
    end
    if ndims(a) != 1
        serialize(s, size(a))
    else
        serialize(s, length(a))
    end
    if isbits(elty)
        serialize_array_data(s.io, a)
    else
        for i = 1:length(a)
            if isdefined(a, i)
                serialize(s, a[i])
            else
                writetag(s.io, UNDEFREF_TAG)
            end
        end
    end
end

function serialize{T,N,A<:Array}(s::SerializationState, a::SubArray{T,N,A})
    if !isbits(T) || stride(a,1)!=1
        return serialize(s, copy(a))
    end
    writetag(s.io, ARRAY_TAG)
    serialize(s, T)
    serialize(s, size(a))
    serialize_array_data(s.io, a)
end

function serialize{T<:AbstractString}(s::SerializationState, ss::SubString{T})
    # avoid saving a copy of the parent string, keeping the type of ss
    invoke(serialize, Tuple{SerializationState,Any}, s, convert(SubString{T}, convert(T,ss)))
end

# Don't serialize the pointers
function serialize(s::SerializationState, r::Regex)
    serialize_type(s, typeof(r))
    serialize(s, r.pattern)
    serialize(s, r.options)
end

function serialize(s::SerializationState, n::BigInt)
    serialize_type(s, BigInt)
    serialize(s, base(62,n))
end

function serialize(s::SerializationState, n::BigFloat)
    serialize_type(s, BigFloat)
    serialize(s, string(n))
end

function serialize(s::SerializationState, ex::Expr)
    serialize_cycle(s, e) && return
    l = length(ex.args)
    if l <= 255
        writetag(s.io, EXPR_TAG)
        write(s.io, UInt8(l))
    else
        writetag(s.io, LONGEXPR_TAG)
        write(s.io, Int32(l))
    end
    serialize(s, ex.head)
    serialize(s, ex.typ)
    for a = ex.args
        serialize(s, a)
    end
end

function serialize(s::SerializationState, t::Dict)
    serialize_cycle(s, t) && return
    serialize_type(s, typeof(t))
    write(s.io, Int32(length(t)))
    for (k,v) in t
        serialize(s, k)
        serialize(s, v)
    end
end

function serialize_mod_names(s::SerializationState, m::Module)
    p = module_parent(m)
    if m !== p
        serialize_mod_names(s, p)
        serialize(s, module_name(m))
    end
end

function serialize(s::SerializationState, m::Module)
    writetag(s.io, MODULE_TAG)
    serialize_mod_names(s, m)
    writetag(s.io, EMPTYTUPLE_TAG)
end

function serialize(s::SerializationState, f::Function)
    serialize_cycle(s, f) && return
    writetag(s.io, FUNCTION_TAG)
    name = false
    if isgeneric(f)
        name = f.env.name
    elseif isa(f.env,Symbol)
        name = f.env
    end
    if isa(name,Symbol)
        if isdefined(Base,name) && is(f,getfield(Base,name))
            write(s.io, UInt8(0))
            serialize(s, name)
            return
        end
        mod = ()
        if isa(f.env,Symbol)
            mod = Core
        elseif !is(f.env.defs, ())
            mod = f.env.defs.func.code.module
        end
        if mod !== ()
            if isdefined(mod,name) && is(f,getfield(mod,name))
                # toplevel named func
                write(s.io, UInt8(2))
                serialize(s, mod)
                serialize(s, name)
                return
            end
        end
        write(s.io, UInt8(3))
        serialize(s, f.env)
    else
        linfo = f.code
        @assert isa(linfo,LambdaStaticData)
        write(s.io, UInt8(1))
        serialize(s, linfo)
        serialize(s, f.env)
    end
end

const lambda_numbers = WeakKeyDict()
lnumber_salt = 0
function lambda_number(l::LambdaStaticData)
    global lnumber_salt, lambda_numbers
    if haskey(lambda_numbers, l)
        return lambda_numbers[l]
    end
    # a hash function that always gives the same number to the same
    # object on the same machine, and is unique over all machines.
    ln = lnumber_salt+(UInt64(myid())<<44)
    lnumber_salt += 1
    lambda_numbers[l] = ln
    return ln
end

function serialize(s::SerializationState, linfo::LambdaStaticData)
    serialize_cycle(s, linfo) && return
    writetag(s.io, LAMBDASTATICDATA_TAG)
    serialize(s, lambda_number(linfo))
    serialize(s, uncompressed_ast(linfo))
    if isdefined(linfo.def, :roots)
        serialize(s, linfo.def.roots)
    else
        serialize(s, [])
    end
    serialize(s, linfo.sparams)
    serialize(s, linfo.inferred)
    serialize(s, linfo.module)
    serialize(s, linfo.target)
    if isdefined(linfo, :capt)
        serialize(s, linfo.capt)
    else
        serialize(s, nothing)
    end
end

function serialize(s::SerializationState, t::Task)
    serialize_cycle(s, t) && return
    if istaskstarted(t) && !istaskdone(t)
        error("cannot serialize a running Task")
    end
    writetag(s.io, TASK_TAG)
    serialize(s, t.code)
    serialize(s, t.storage)
    serialize(s, t.state == :queued || t.state == :waiting ? (:runnable) : t.state)
    serialize(s, t.result)
    serialize(s, t.exception)
end

function serialize_type_data(s, t)
    tname = t.name.name
    serialize(s, tname)
    mod = t.name.module
    serialize(s, mod)
    if length(t.parameters) > 0
        if isdefined(mod,tname) && is(t,getfield(mod,tname))
            serialize(s, svec())
        else
            serialize(s, t.parameters)
        end
    end
end

function serialize(s::SerializationState, t::DataType)
    tag = sertag(t)
    if tag > 0
        return write_as_tag(s.io, tag)
    end
    writetag(s.io, DATATYPE_TAG)
    write(s.io, UInt8(0))
    serialize_type_data(s, t)
end

function serialize_type(s::SerializationState, t::DataType)
    tag = sertag(t)
    if tag > 0
        return writetag(s.io, tag)
    end
    writetag(s.io, DATATYPE_TAG)
    write(s.io, UInt8(1))
    serialize_type_data(s, t)
end

function serialize(s::SerializationState, n::Int)
    if 0 <= n <= 32
        write(s.io, UInt8(ZERO_TAG+n))
        return
    end
    writetag(s.io, INT_TAG)
    write(s.io, n)
end

function serialize(s::SerializationState, x)
    tag = sertag(x)
    if tag > 0
        return write_as_tag(s.io, tag)
    end
    t = typeof(x)::DataType
    nf = nfields(t)
    if nf == 0 && t.size > 0
        serialize_type(s, t)
        write(s.io, x)
    else
        t.mutable && serialize_cycle(s, x) && return
        serialize_type(s, t)
        for i in 1:nf
            if isdefined(x, i)
                serialize(s, getfield(x, i))
            else
                writetag(s.io, UNDEFREF_TAG)
            end
        end
    end
end

serialize(s::IO, x) = serialize(SerializationState(s), x)

## deserializing values ##

deserialize(s::IO) = deserialize(SerializationState(s))

function deserialize(s::SerializationState)
    handle_deserialize(s, Int32(read(s.io, UInt8)::UInt8))
end

function deserialize_cycle(s::SerializationState, x)
    if !isimmutable(x) && !typeof(x).pointerfree
        s.table[s.counter] = x
        s.counter += 1
    end
    nothing
end

# deserialize_ is an internal function to dispatch on the tag
# describing the serialized representation. the number of
# representations is fixed, so deserialize_ does not get extended.
function handle_deserialize(s::SerializationState, b::Int32)
    if b == 0
        return desertag(Int32(read(s.io, UInt8)::UInt8))
    end
    if b >= VALUE_TAGS
        return desertag(b)
    elseif b == TUPLE_TAG
        return deserialize_tuple(s, Int(read(s.io, UInt8)::UInt8))
    elseif b == LONGTUPLE_TAG
        return deserialize_tuple(s, Int(read(s.io, Int32)::Int32))
    elseif b == BACKREF_TAG
        id = read(s.io, Int)::Int
        return s.table[id]
    elseif b == ARRAY_TAG
        return deserialize_array(s)
    elseif b == DATATYPE_TAG
        return deserialize_datatype(s)
    elseif b == SYMBOL_TAG
        return symbol(read(s.io, UInt8, Int(read(s.io, UInt8)::UInt8)))
    elseif b == LONGSYMBOL_TAG
        return symbol(read(s.io, UInt8, Int(read(s.io, Int32)::Int32)))
    elseif b == EXPR_TAG
        return deserialize_expr(s, Int(read(s.io, UInt8)::UInt8))
    elseif b == LONGEXPR_TAG
        return deserialize_expr(s, Int(read(s.io, Int32)::Int32))
    end
    return deserialize(s, desertag(b))
end

deserialize_tuple(s::SerializationState, len) = ntuple(i->deserialize(s), len)

function deserialize(s::SerializationState, ::Type{SimpleVector})
    n = read(s.io, Int32)
    svec([ deserialize(s) for i=1:n ]...)
end

function deserialize(s::SerializationState, ::Type{Module})
    path = deserialize(s)
    m = Main
    if isa(path,Tuple) && path !== ()
        # old version
        for mname in path
            if !isdefined(m,mname)
                warn("Module $mname not defined on process $(myid())")  # an error seemingly fails
            end
            m = getfield(m,mname)::Module
        end
    else
        mname = path
        while mname !== ()
            if !isdefined(m,mname)
                warn("Module $mname not defined on process $(myid())")  # an error seemingly fails
            end
            m = getfield(m,mname)::Module
            mname = deserialize(s)
        end
    end
    m
end

const known_lambda_data = Dict()

function deserialize(s::SerializationState, ::Type{Function})
    b = read(s.io, UInt8)::UInt8
    if b==0
        name = deserialize(s)::Symbol
        if !isdefined(Base,name)
            return (args...)->error("function $name not defined on process $(myid())")
        end
        return getfield(Base,name)::Function
    elseif b==2
        mod = deserialize(s)::Module
        name = deserialize(s)::Symbol
        if !isdefined(mod,name)
            return (args...)->error("function $name not defined on process $(myid())")
        end
        return getfield(mod,name)::Function
    elseif b==3
        env = deserialize(s)
        return ccall(:jl_new_gf_internal, Any, (Any,), env)::Function
    end
    linfo = deserialize(s)
    f = ccall(:jl_new_closure, Any, (Ptr{Void}, Ptr{Void}, Any), C_NULL, C_NULL, linfo)::Function
    deserialize_cycle(s, f)
    f.env = deserialize(s)
    return f
end

function deserialize(s::SerializationState, ::Type{LambdaStaticData})
    lnumber = deserialize(s)
    if haskey(known_lambda_data, lnumber)
        linfo = known_lambda_data[lnumber]::LambdaStaticData
        makenew = false
    else
        linfo = ccall(:jl_new_lambda_info, Any, (Ptr{Void}, Ptr{Void}), C_NULL, C_NULL)::LambdaStaticData
        makenew = true
    end
    deserialize_cycle(s, linfo)
    ast = deserialize(s)
    roots = deserialize(s)
    sparams = deserialize(s)
    infr = deserialize(s)
    mod = deserialize(s)
    target = deserialize(s)
    capt = deserialize(s)
    if makenew
        linfo.ast = ast
        linfo.sparams = sparams
        linfo.inferred = infr
        linfo.module = mod
        linfo.target = target
        linfo.roots = roots
        if !is(capt,nothing)
            linfo.capt = capt
        end
        known_lambda_data[lnumber] = linfo
    end
    return linfo
end

function deserialize_array(s::SerializationState)
    d1 = deserialize(s)
    if isa(d1,Type)
        elty = d1
        d1 = deserialize(s)
    else
        elty = UInt8
    end
    if isa(d1,Integer)
        if elty !== Bool && isbits(elty)
            return read!(s.io, Array(elty, d1))
        end
        dims = (Int(d1),)
    else
        dims = convert(Dims, d1)::Dims
    end
    if isbits(elty)
        n = prod(dims)::Int
        if elty === Bool && n>0
            A = Array(Bool, dims)
            i = 1
            while i <= n
                b = read(s.io, UInt8)::UInt8
                v = (b>>7) != 0
                count = b&0x7f
                nxt = i+count
                while i < nxt
                    A[i] = v; i+=1
                end
            end
        else
            A = read(s.io, elty, dims)
        end
        return A
    end
    A = Array(elty, dims)
    deserialize_cycle(s, A)
    for i = 1:length(A)
        tag = Int32(read(s.io, UInt8)::UInt8)
        if tag != UNDEFREF_TAG
            A[i] = handle_deserialize(s, tag)
        end
    end
    return A
end

function deserialize_expr(s::SerializationState, len)
    hd = deserialize(s)::Symbol
    ty = deserialize(s)
    e = Expr(hd)
    deserialize_cycle(s, e)
    e.args = Any[ deserialize(s) for i=1:len ]
    e.typ = ty
    e
end

function deserialize(s::SerializationState, ::Type{UnionType})
    types = deserialize(s)
    Union(types...)
end

function deserialize_datatype(s::SerializationState)
    form = read(s.io, UInt8)::UInt8
    name = deserialize(s)::Symbol
    mod = deserialize(s)::Module
    ty = getfield(mod,name)
    if length(ty.parameters) == 0
        t = ty
    else
        params = deserialize(s)
        t = ty{params...}
    end
    if form == 0
        return t
    end
    deserialize(s, t)
end

deserialize{T}(s::SerializationState, ::Type{Ptr{T}}) = convert(Ptr{T}, 0)

function deserialize(s::SerializationState, ::Type{Task})
    t = Task(()->nothing)
    deserialize_cycle(s, t)
    t.code = deserialize(s)
    t.storage = deserialize(s)
    t.state = deserialize(s)
    t.result = deserialize(s)
    t.exception = deserialize(s)
    t
end

# default DataType deserializer
function deserialize(s::SerializationState, t::DataType)
    nf = nfields(t)
    if nf == 0 && t.size > 0
        # bits type
        return read(s.io, t)
    end
    if nf == 0
        return ccall(:jl_new_struct, Any, (Any,Any...), t)
    elseif isbits(t)
        if nf == 1
            return ccall(:jl_new_struct, Any, (Any,Any...), t, deserialize(s))
        elseif nf == 2
            f1 = deserialize(s)
            f2 = deserialize(s)
            return ccall(:jl_new_struct, Any, (Any,Any...), t, f1, f2)
        elseif nf == 3
            f1 = deserialize(s)
            f2 = deserialize(s)
            f3 = deserialize(s)
            return ccall(:jl_new_struct, Any, (Any,Any...), t, f1, f2, f3)
        else
            flds = Any[ deserialize(s) for i = 1:nf ]
            return ccall(:jl_new_structv, Any, (Any,Ptr{Void},UInt32), t, flds, nf)
        end
    else
        x = ccall(:jl_new_struct_uninit, Any, (Any,), t)
        t.mutable && deserialize_cycle(s, x)
        for i in 1:nf
            tag = Int32(read(s.io, UInt8)::UInt8)
            if tag != UNDEFREF_TAG
                ccall(:jl_set_nth_field, Void, (Any, Csize_t, Any), x, i-1, handle_deserialize(s, tag))
            end
        end
        return x
    end
end

function deserialize{K,V}(s::SerializationState, T::Type{Dict{K,V}})
    n = read(s.io, Int32)
    t = T(); sizehint!(t, n)
    deserialize_cycle(s, t)
    for i = 1:n
        k = deserialize(s)
        v = deserialize(s)
        t[k] = v
    end
    return t
end

deserialize(s::SerializationState, ::Type{BigFloat}) = BigFloat(deserialize(s))

deserialize(s::SerializationState, ::Type{BigInt}) = get(GMP.tryparse_internal(BigInt, deserialize(s), 62, true))

deserialize(s::SerializationState, ::Type{BigInt}) = get(GMP.tryparse_internal(BigInt, deserialize(s), 62, true))

function deserialize(s::SerializationState, t::Type{Regex})
    pattern = deserialize(s)
    options = deserialize(s)
    Regex(pattern, options)
end

end
