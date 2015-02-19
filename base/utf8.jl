## from base/boot.jl:
#
# immutable UTF8String <: AbstractString
#     data::Array{UInt8,1}
# end
#

## basic UTF-8 decoding & iteration ##

const utf8_offset = [
    0x00000000, 0x00003080,
    0x000e2080, 0x03c82080,
    0xfa082080, 0x82082080,
]

const utf8_trailing = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,4,4,4,4,5,5,5,5,
]

is_utf8_start(byte::UInt8) = ((byte&0xc0)!=0x80)

## required core functionality ##

function endof(s::UTF8String)
    d = s.data
    i = length(d)
    i == 0 && return StringIndex(i)
    while !is_utf8_start(d[i])
        i -= 1
    end
    StringIndex(i)
end
length(s::UTF8String) = int(ccall(:u8_strlen, Csize_t, (Ptr{UInt8},), s.data))

function next(s::UTF8String, i::StringIndex)
    # potentially faster version
    # d = s.data
    # a::UInt32 = d[i]
    # if a < 0x80; return char(a); end
    # #if a&0xc0==0x80; return '\ufffd'; end
    # b::UInt32 = a<<6 + d[i+1]
    # if a < 0xe0; return char(b - 0x00003080); end
    # c::UInt32 = b<<6 + d[i+2]
    # if a < 0xf0; return char(c - 0x000e2080); end
    # return char(c<<6 + d[i+3] - 0x03c82080)

    j = i.i
    d = s.data
    b = d[j]
    if !is_utf8_start(b)
        k = j-1
        while 0 < k && !is_utf8_start(d[k])
            k -= 1
        end
        if 0 < k && j <= k+utf8_trailing[d[k]+1] <= length(d)
            # b is a continuation byte of a valid UTF-8 character
            throw(ArgumentError("invalid UTF-8 character index"))
        end
        # move past 1 byte in case the data is actually Latin-1
        return '\ufffd', StringIndex(j+1)
    end
    trailing = utf8_trailing[b+1]
    if length(d) < j + trailing
        return '\ufffd', StringIndex(j+1)
    end
    c::UInt32 = 0
    for k = 1:trailing+1
        c <<= 6
        c += d[j]
        j += 1
    end
    c -= utf8_offset[trailing+1]
    char(c), StringIndex(j)
end

# TODO: deprecate
function next(s::UTF8String, i::Int)
    # potentially faster version
    # d = s.data
    # a::UInt32 = d[i]
    # if a < 0x80; return char(a); end
    # #if a&0xc0==0x80; return '\ufffd'; end
    # b::UInt32 = a<<6 + d[i+1]
    # if a < 0xe0; return char(b - 0x00003080); end
    # c::UInt32 = b<<6 + d[i+2]
    # if a < 0xf0; return char(c - 0x000e2080); end
    # return char(c<<6 + d[i+3] - 0x03c82080)

    d = s.data
    b = d[i]
    if !is_utf8_start(b)
        j = i-1
        while 0 < j && !is_utf8_start(d[j])
            j -= 1
        end
        if 0 < j && i <= j+utf8_trailing[d[j]+1] <= length(d)
            # b is a continuation byte of a valid UTF-8 character
            throw(ArgumentError("invalid UTF-8 character index"))
        end
        # move past 1 byte in case the data is actually Latin-1
        return '\ufffd', i+1
    end
    trailing = utf8_trailing[b+1]
    if length(d) < i + trailing
        return '\ufffd', i+1
    end
    c::UInt32 = 0
    for j = 1:trailing+1
        c <<= 6
        c += d[i]
        i += 1
    end
    c -= utf8_offset[trailing+1]
    char(c), i
end

function first_utf8_byte(ch::Char)
    c = reinterpret(UInt32, ch)
    c < 0x80    ? uint8(c) :
    c < 0x800   ? uint8((c>>6)  | 0xc0) :
    c < 0x10000 ? uint8((c>>12) | 0xe0) :
                  uint8((c>>18) | 0xf0)
end

function reverseind(s::UTF8String, i::StringIndex)
    j = lastidx(s).i + 1 - i.i
    d = s.data
    while !is_utf8_start(d[j])
        j -= 1
    end
    return StringIndex(j)
end

# TODO: deprecate
function reverseind(s::UTF8String, i::Integer)
    j = lastidx(s) + 1 - i
    d = s.data
    while !is_utf8_start(d[j])
        j -= 1
    end
    return j
end

## overload methods for efficiency ##

sizeof(s::UTF8String) = sizeof(s.data)

# FIXME: should probably return an Int instead of a StringIndex,
# as this is used for internally indexing an array of bytes
lastidx(s::UTF8String) = StringIndex(length(s.data))

isvalid(s::UTF8String, i::StringIndex) =
    (start(s) <= i <= endof(s)) && is_utf8_start(s.data[i.i])

# TODO: deprecate
isvalid(s::UTF8String, i::Integer) =
    (1 <= i <= endof(s.data)) && is_utf8_start(s.data[i])

const empty_utf8 = UTF8String(UInt8[])

function getindex(s::UTF8String, r::StringRange)
    isempty(r) && return empty_utf8
    i, j = first(r), last(r)
    d = s.data
    if !is_utf8_start(d[i.i])
        i = nextind(s,i)
    end
    # FIXME: is this correct?
    if j.i > length(d)
        throw(BoundsError())
    end
    jj = nextind(s,j).i - 1
    UTF8String(d[i.i:jj])
end

function search(s::UTF8String, c::Char, i::StringIndex)
    c < char(0x80) && return StringIndex(search(s.data, uint8(c), i.i))
    while true
        i = StringIndex(search(s.data, first_utf8_byte(c), i.i))
        (i == StringIndex(0) || s[i] == c) && return i
        i = next(s,i)[2]
    end
end

function rsearch(s::UTF8String, c::Char, i::StringIndex)
    c < char(0x80) && return StringIndex(rsearch(s.data, uint8(c), i.i))
    b = first_utf8_byte(c)
    while true
        i = StringIndex(rsearch(s.data, b, i.i))
        (i == StringIndex(0) || s[i] == c) && return i
        i = prevind(s,i)
    end
end

# TODO: deprecate
function getindex(s::UTF8String, r::UnitRange{Int})
    isempty(r) && return empty_utf8
    i, j = first(r), last(r)
    d = s.data
    if !is_utf8_start(d[i])
        i = nextind(s,i)
    end
    if j > length(d)
        throw(BoundsError())
    end
    j = nextind(s,j)-1
    UTF8String(d[i:j])
end

function search(s::UTF8String, c::Char, i::Integer)
    c < char(0x80) && return search(s.data, uint8(c), i)
    while true
        i = search(s.data, first_utf8_byte(c), i)
        (i == StringIndex(0) || s[i] == c) && return i
        i = next(s,i)[2]
    end
end

function rsearch(s::UTF8String, c::Char, i::Integer)
    c < char(0x80) && return rsearch(s.data, uint8(c), i)
    b = first_utf8_byte(c)
    while true
        i = rsearch(s.data, b, i)
        (i == StringIndex(0) || s[i] == c) && return i
        i = prevind(s,i)
    end
end

function string(a::ByteString...)
    if length(a) == 1
        return a[1]::UTF8String
    end
    # FIXME: why not preallocate the array as for ASCIIString?
    # ^^ at least one must be UTF-8 or the ASCII-only method would get called
    data = Array(UInt8,0)
    for d in a
        append!(data,d.data)
    end
    UTF8String(data)
end

function reverse(s::UTF8String)
    out = similar(s.data)
    if ccall(:u8_reverse, Cint, (Ptr{UInt8}, Ptr{UInt8}, Csize_t),
             out, s.data, length(out)) == 1
        throw(ArgumentError("invalid UTF-8 data"))
    end
    UTF8String(out)
end

## outputing UTF-8 strings ##

write(io::IO, s::UTF8String) = write(io, s.data)

## transcoding to UTF-8 ##

utf8(x) = convert(UTF8String, x)
convert(::Type{UTF8String}, s::UTF8String) = s
convert(::Type{UTF8String}, s::ASCIIString) = UTF8String(s.data)
convert(::Type{UTF8String}, a::Array{UInt8,1}) = is_valid_utf8(a) ? UTF8String(a) : throw(ArgumentError("invalid UTF-8 sequence"))
function convert(::Type{UTF8String}, a::Array{UInt8,1}, invalids_as::AbstractString)
    l = length(a)
    idx = 1
    iscopy = false
    while idx <= l
        if is_utf8_start(a[idx])
            nextidx = idx+1+utf8_trailing[a[idx]+1]
            (nextidx <= (l+1)) && (idx = nextidx; continue)
        end
        !iscopy && (a = copy(a); iscopy = true)
        endn = idx
        while endn <= l
            is_utf8_start(a[endn]) && break
            endn += 1
        end
        (endn > idx) && (endn -= 1)
        splice!(a, idx:endn, invalids_as.data)
        l = length(a)
    end
    UTF8String(a)
end
convert(::Type{UTF8String}, s::AbstractString) = utf8(bytestring(s))

# The last case is the replacement character 0xfffd (3 bytes)
utf8sizeof(c::Char) = c < char(0x80) ? 1 : c < char(0x800) ? 2 : c < char(0x10000) ? 3 : c < char(0x110000) ? 4 : 3
