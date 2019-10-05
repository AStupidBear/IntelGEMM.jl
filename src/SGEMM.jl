module SGEMM

using LinearAlgebra
using .BLAS: has_offset_axes, chkstride1, BlasInt

function BLAS.gemm!(transA::AbstractChar, transB::AbstractChar, alpha::Float32, A::AbstractVecOrMat{Float32}, B::AbstractVecOrMat{Float32}, beta::Float32, C::AbstractVecOrMat{Float32})
    @assert !has_offset_axes(A, B, C)
    m = size(A, transA == 'N' ? 1 : 2)
    ka = size(A, transA == 'N' ? 2 : 1)
    kb = size(B, transB == 'N' ? 1 : 2)
    n = size(B, transB == 'N' ? 2 : 1)
    if ka != kb || m != size(C,1) || n != size(C,2)
        throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
    end
    chkstride1(A)
    chkstride1(B)
    chkstride1(C)
    ccall((:sgemm_, "libmkl_rt"), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
        Ref{BlasInt}, Ref{Float32}, Ptr{Float32}, Ref{BlasInt},
        Ptr{Float32}, Ref{BlasInt}, Ref{Float32}, Ptr{Float32},
        Ref{BlasInt}),
        transA, transB, m, n,
        ka, alpha, A, max(1,stride(A,2)),
        B, max(1,stride(B,2)), beta, C,
        max(1,stride(C,2)))
    C
end

end # module
