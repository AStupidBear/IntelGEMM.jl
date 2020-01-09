module SGEMM

using LinearAlgebra, Libdl
using LinearAlgebra: chkstride1, require_one_based_indexing

include(joinpath("..", "deps", "deps.jl"))

@enum Threading begin
    THREADING_INTEL
    THREADING_SEQUENTIAL
    THREADING_PGI
    THREADING_GNU
    THREADING_TBB
end

@enum Interface begin
    INTERFACE_LP64
    INTERFACE_ILP64
    INTERFACE_GNU
end

function set_threading_layer(layer::Threading = THREADING_INTEL)
    err = ccall((:MKL_Set_Threading_Layer, libmkl_rt), Cint, (Cint,), layer)
    if err == -1
        throw(ErrorException("return value was -1"))
    end
    return nothing
end

function set_interface_layer(interface = INTERFACE_LP64)
    err = ccall((:MKL_Set_Interface_Layer, libmkl_rt), Cint, (Cint,), interface)
    if err == -1
        throw(ErrorException("return value was -1"))
    end
    return nothing
end

function __init__()

    if Sys.iswindows()
        # On Windows, we have to open the threading library before calling libmkl_rt
        dlopen(libmkl_intel_thread)
    end

    set_threading_layer()
    set_interface_layer()
end

macro blasfunc(x)
    return Expr(:quote, x)
end

for (gemm, elty) in
    ((:dgemm_,:Float64),
     (:sgemm_,:Float32),
     (:zgemm_,:ComplexF64),
     (:cgemm_,:ComplexF32))
@eval begin
         # SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
         # *     .. Scalar Arguments ..
         #       DOUBLE PRECISION ALPHA,BETA
         #       INTEGER K,LDA,LDB,LDC,M,N
         #       CHARACTER TRANSA,TRANSB
         # *     .. Array Arguments ..
         #       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
    function BLAS.gemm!(transA::AbstractChar, transB::AbstractChar,
                   alpha::Union{($elty), Bool},
                   A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty},
                   beta::Union{($elty), Bool},
                   C::AbstractVecOrMat{$elty})
#           if any([stride(A,1), stride(B,1), stride(C,1)] .!= 1)
#               error("gemm!: BLAS module requires contiguous matrix columns")
#           end  # should this be checked on every call?
        require_one_based_indexing(A, B, C)
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
        ccall((@blasfunc($gemm), libmkl_rt), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{Int32}, Ref{Int32},
             Ref{Int32}, Ref{$elty}, Ptr{$elty}, Ref{Int32},
             Ptr{$elty}, Ref{Int32}, Ref{$elty}, Ptr{$elty},
             Ref{Int32}),
             transA, transB, m, n,
             ka, alpha, A, max(1,stride(A,2)),
             B, max(1,stride(B,2)), beta, C,
             max(1,stride(C,2)))
        C
    end
end
end

for (fname, elty) in ((:dgemv_,:Float64),
                      (:sgemv_,:Float32),
                      (:zgemv_,:ComplexF64),
                      (:cgemv_,:ComplexF32))
    @eval begin
             #SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
             #*     .. Scalar Arguments ..
             #      DOUBLE PRECISION ALPHA,BETA
             #      INTEGER INCX,INCY,LDA,M,N
             #      CHARACTER TRANS
             #*     .. Array Arguments ..
             #      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
        function BLAS.gemv!(trans::AbstractChar, alpha::Union{($elty), Bool},
                       A::AbstractVecOrMat{$elty}, X::AbstractVector{$elty},
                       beta::Union{($elty), Bool}, Y::AbstractVector{$elty})
            require_one_based_indexing(A, X, Y)
            m,n = size(A,1),size(A,2)
            if trans == 'N' && (length(X) != n || length(Y) != m)
                throw(DimensionMismatch("A has dimensions $(size(A)), X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'C' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("the adjoint of A has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            elseif trans == 'T' && (length(X) != m || length(Y) != n)
                throw(DimensionMismatch("the transpose of A has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
            end
            chkstride1(A)
            ccall((@blasfunc($fname), libmkl_rt), Cvoid,
                (Ref{UInt8}, Ref{Int32}, Ref{Int32}, Ref{$elty},
                 Ptr{$elty}, Ref{Int32}, Ptr{$elty}, Ref{Int32},
                 Ref{$elty}, Ptr{$elty}, Ref{Int32}),
                 trans, size(A,1), size(A,2), alpha,
                 A, max(1,stride(A,2)), X, stride(X,1),
                 beta, Y, stride(Y,1))
            Y
        end
    end
end

end # module
