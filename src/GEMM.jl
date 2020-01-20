module GEMM

using LinearAlgebra, Libdl, Requires
using LinearAlgebra: chkstride1, checksquare, require_one_based_indexing
import LinearAlgebra.BLAS: gemm!, gemv!
import LinearAlgebra.LAPACK: chkargsok, chklapackerror, chknonsingular, chkposdef, chkfinite
import LinearAlgebra.LAPACK: geevx!, ggev!, syev!, syevr!, sygvd!

include(joinpath("..", "deps", "deps.jl"))

const BlasInt = Int32
const libblas = libmkl_rt
const liblapack = libmkl_rt

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
    err = ccall((:MKL_Set_Threading_Layer, libblas), Cint, (Cint,), layer)
    if err == -1
        throw(ErrorException("return value was -1"))
    end
    return nothing
end

function set_interface_layer(interface = INTERFACE_LP64)
    err = ccall((:MKL_Set_Interface_Layer, libblas), Cint, (Cint,), interface)
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
    @require NNlib="872c559c-99b0-510c-b3b7-b6c96a88d5cd" include("nnlib.jl")
end

macro blasfunc(x)
    return Expr(:quote, x)
end

for chk in (:chkargsok, :chklapackerror, :chknonsingular, :chkposdef)
    @eval $chk(ret) = $chk(BLAS.BlasInt(ret))
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
    function gemm!(transA::AbstractChar, transB::AbstractChar,
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
            ccall((@blasfunc($gemm), libblas), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
                Ref{BlasInt}),
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
        function gemv!(trans::AbstractChar, alpha::Union{($elty), Bool},
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
            ccall((@blasfunc($fname), libblas), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ref{$elty}, Ptr{$elty}, Ref{BlasInt}),
                 trans, size(A,1), size(A,2), alpha,
                 A, max(1,stride(A,2)), X, stride(X,1),
                 beta, Y, stride(Y,1))
            Y
        end
    end
end

for (geevx, ggev, elty) in
    ((:dgeevx_,:dggev_,:Float64),
     (:sgeevx_,:sggev_,:Float32))
    @eval begin
        #     SUBROUTINE DGEEVX( BALANC, JOBVL, JOBVR, SENSE, N, A, LDA, WR, WI,
        #                          VL, LDVL, VR, LDVR, ILO, IHI, SCALE, ABNRM,
        #                          RCONDE, RCONDV, WORK, LWORK, IWORK, INFO )
        #
        #       .. Scalar Arguments ..
        #       CHARACTER          BALANC, JOBVL, JOBVR, SENSE
        #       INTEGER            IHI, ILO, INFO, LDA, LDVL, LDVR, LWORK, N
        #       DOUBLE PRECISION   ABNRM
        #       ..
        #       .. Array Arguments ..
        #       INTEGER            IWORK( * )
        #       DOUBLE PRECISION   A( LDA, * ), RCONDE( * ), RCONDV( * ),
        #      $                   SCALE( * ), VL( LDVL, * ), VR( LDVR, * ),
        #      $                   WI( * ), WORK( * ), WR( * )
        function geevx!(balanc::AbstractChar, jobvl::AbstractChar, jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix{$elty})
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            lda = max(1,stride(A,2))
            wr = similar(A, $elty, n)
            wi = similar(A, $elty, n)
            if balanc ∉ ['N', 'P', 'S', 'B']
                throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
            end
            ldvl = 0
            if jobvl == 'V'
                ldvl = n
            elseif jobvl == 'N'
                ldvl = 0
            else
                throw(ArgumentError("jobvl must be 'V' or 'N', but $jobvl was passed"))
            end
            VL = similar(A, $elty, ldvl, n)
            ldvr = 0
            if jobvr == 'V'
                ldvr = n
            elseif jobvr == 'N'
                ldvr = 0
            else
                throw(ArgumentError("jobvr must be 'V' or 'N', but $jobvr was passed"))
            end
            VR = similar(A, $elty, ldvr, n)
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            scale = similar(A, $elty, n)
            abnrm = Ref{$elty}()
            rconde = similar(A, $elty, n)
            rcondv = similar(A, $elty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iworksize = 0
            if sense == 'N' || sense == 'E'
                iworksize = 0
            elseif sense == 'V' || sense == 'B'
                iworksize = 2*n - 2
            else
                throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
            end
            iwork = Vector{BlasInt}(undef, iworksize)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                       balanc, jobvl, jobvr, sense,
                       n, A, lda, wr,
                       wi, VL, max(1,ldvl), VR,
                       max(1,ldvr), ilo, ihi, scale,
                       abnrm, rconde, rcondv, work,
                       lwork, iwork, info)
                # chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(work[1])
                    resize!(work, lwork)
                end
            end
            A, wr, wi, VL, VR, ilo[], ihi[], scale, abnrm[], rconde, rcondv
        end

        #       SUBROUTINE DGGEV( JOBVL, JOBVR, N, A, LDA, B, LDB, ALPHAR, ALPHAI,
        #      $                  BETA, VL, LDVL, VR, LDVR, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBVL, JOBVR
        #       INTEGER            INFO, LDA, LDB, LDVL, LDVR, LWORK, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ),
        #      $                   B( LDB, * ), BETA( * ), VL( LDVL, * ),
        #      $                   VR( LDVR, * ), WORK( * )
        function ggev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            require_one_based_indexing(A, B)
            chkstride1(A,B)
            n, m = checksquare(A,B)
            if n != m
                throw(DimensionMismatch("A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            alphar = similar(A, $elty, n)
            alphai = similar(A, $elty, n)
            beta = similar(A, $elty, n)
            ldvl = 0
            if jobvl == 'V'
                ldvl = n
            elseif jobvl == 'N'
                ldvl = 1
            else
                throw(ArgumentError("jobvl must be 'V' or 'N', but $jobvl was passed"))
            end
            vl = similar(A, $elty, ldvl, n)
            ldvr = 0
            if jobvr == 'V'
                ldvr = n
            elseif jobvr == 'N'
                ldvr = 1
            else
                throw(ArgumentError("jobvr must be 'V' or 'N', but $jobvr was passed"))
            end
            vr = similar(A, $elty, ldvr, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($ggev), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{BlasInt}),
                    jobvl, jobvr, n, A,
                    lda, B, ldb, alphar,
                    alphai, beta, vl, ldvl,
                    vr, ldvr, work, lwork,
                    info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(work[1])
                    resize!(work, lwork)
                end
            end
            alphar, alphai, beta, vl, vr
        end
    end
end

for (geevx, ggev, elty, relty) in
    ((:zgeevx_,:zggev_,:ComplexF64,:Float64),
     (:cgeevx_,:cggev_,:ComplexF32,:Float32))
    @eval begin
        #     SUBROUTINE ZGEEVX( BALANC, JOBVL, JOBVR, SENSE, N, A, LDA, W, VL,
        #                          LDVL, VR, LDVR, ILO, IHI, SCALE, ABNRM, RCONDE,
        #                          RCONDV, WORK, LWORK, RWORK, INFO )
        #
        #       .. Scalar Arguments ..
        #       CHARACTER          BALANC, JOBVL, JOBVR, SENSE
        #       INTEGER            IHI, ILO, INFO, LDA, LDVL, LDVR, LWORK, N
        #       DOUBLE PRECISION   ABNRM
        #       ..
        #       .. Array Arguments ..
        #       DOUBLE PRECISION   RCONDE( * ), RCONDV( * ), RWORK( * ),
        #      $                   SCALE( * )
        #       COMPLEX*16         A( LDA, * ), VL( LDVL, * ), VR( LDVR, * ),
        #      $                   W( * ), WORK( * )
        function geevx!(balanc::AbstractChar, jobvl::AbstractChar, jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix{$elty})
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            lda = max(1,stride(A,2))
            w = similar(A, $elty, n)
            if balanc ∉ ['N', 'P', 'S', 'B']
                throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
            end
            ldvl = 0
            if jobvl == 'V'
                ldvl = n
            elseif jobvl == 'N'
                ldvl = 0
            else
                throw(ArgumentError("jobvl must be 'V' or 'N', but $jobvl was passed"))
            end
            VL = similar(A, $elty, ldvl, n)
            ldvr = 0
            if jobvr == 'V'
                ldvr = n
            elseif jobvr == 'N'
                ldvr = 0
            else
                throw(ArgumentError("jobvr must be 'V' or 'N', but $jobvr was passed"))
            end
            if sense ∉ ['N','E','V','B']
                throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
            end
            VR = similar(A, $elty, ldvr, n)
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            scale = similar(A, $relty, n)
            abnrm = Ref{$relty}()
            rconde = similar(A, $relty, n)
            rcondv = similar(A, $relty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            rwork = Vector{$relty}(undef, 2n)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                       Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}),
                       balanc, jobvl, jobvr, sense,
                       n, A, lda, w,
                       VL, max(1,ldvl), VR, max(1,ldvr),
                       ilo, ihi, scale, abnrm,
                       rconde, rcondv, work, lwork,
                       rwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(work[1])
                    resize!(work, lwork)
                end
            end
            A, w, VL, VR, ilo[], ihi[], scale, abnrm[], rconde, rcondv
        end

        # SUBROUTINE ZGGEV( JOBVL, JOBVR, N, A, LDA, B, LDB, ALPHA, BETA,
        #      $                  VL, LDVL, VR, LDVR, WORK, LWORK, RWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBVL, JOBVR
        #       INTEGER            INFO, LDA, LDB, LDVL, LDVR, LWORK, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   RWORK( * )
        #       COMPLEX*16         A( LDA, * ), ALPHA( * ), B( LDB, * ),
        #      $                   BETA( * ), VL( LDVL, * ), VR( LDVR, * ),
        #      $                   WORK( * )
        function ggev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            require_one_based_indexing(A, B)
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            alpha = similar(A, $elty, n)
            beta = similar(A, $elty, n)
            ldvl = 0
            if jobvl == 'V'
                ldvl = n
            elseif jobvl == 'N'
                ldvl = 1
            else
                throw(ArgumentError("jobvl must be 'V' or 'N', but $jobvl was passed"))
            end
            vl = similar(A, $elty, ldvl, n)
            ldvr = 0
            if jobvr == 'V'
                ldvr = n
            elseif jobvr == 'N'
                ldvr = 1
            else
                throw(ArgumentError("jobvr must be 'V' or 'N', but $jobvr was passed"))
            end
            vr = similar(A, $elty, ldvr, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            rwork = Vector{$relty}(undef, 8n)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($ggev), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                     Ptr{BlasInt}),
                    jobvl, jobvr, n, A,
                    lda, B, ldb, alpha,
                    beta, vl, ldvl, vr,
                    ldvr, work, lwork, rwork,
                    info)
                # chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(work[1])
                    resize!(work, lwork)
                end
            end
            alpha, beta, vl, vr
        end
    end
end

# Symmetric (real) eigensolvers
for (syev, syevr, sygvd, elty) in
    ((:dsyev_,:dsyevr_,:dsygvd_,:Float64),
     (:ssyev_,:ssyevr_,:ssygvd_,:Float32))
    @eval begin
        #       SUBROUTINE DSYEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LWORK, N
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * )
        function syev!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)
            W     = similar(A, $elty, n)
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($syev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                      Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      jobz, uplo, n, A, max(1,stride(A,2)), W, work, lwork, info)
                # chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            jobz == 'V' ? (W, A) : W
        end

        #       SUBROUTINE DSYEVR( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU,
        #      $                   ABSTOL, M, W, Z, LDZ, ISUPPZ, WORK, LWORK,
        #      $                   IWORK, LIWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, RANGE, UPLO
        #       INTEGER            IL, INFO, IU, LDA, LDZ, LIWORK, LWORK, M, N
        #       DOUBLE PRECISION   ABSTOL, VL, VU
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            ISUPPZ( * ), IWORK( * )
        #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * ), Z( LDZ, * )
        function syevr!(jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty},
                        vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer, abstol::AbstractFloat)
            chkstride1(A)
            n = checksquare(A)
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            lda = stride(A,2)
            m = Ref{BlasInt}()
            w = similar(A, $elty, n)
            ldz = n
            if jobz == 'N'
                Z = similar(A, $elty, ldz, 0)
            elseif jobz == 'V'
                Z = similar(A, $elty, ldz, n)
            end
            isuppz = similar(A, BlasInt, 2*n)
            work   = Vector{$elty}(undef, 1)
            lwork  = BlasInt(-1)
            iwork  = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info   = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($syevr), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                        Ptr{BlasInt}),
                    jobz, range, uplo, n,
                    A, max(1,lda), vl, vu,
                    il, iu, abstol, m,
                    w, Z, max(1,ldz), isuppz,
                    work, lwork, iwork, liwork,
                    info)
                # chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = iwork[1]
                    resize!(iwork, liwork)
                end
            end
            w[1:m[]], Z[:,1:(jobz == 'V' ? m[] : 0)]
        end
        syevr!(jobz::AbstractChar, A::AbstractMatrix{$elty}) =
            syevr!(jobz, 'A', 'U', A, 0.0, 0.0, 0, 0, -1.0)

        # Generalized eigenproblem
        #           SUBROUTINE DSYGVD( ITYPE, JOBZ, UPLO, N, A, LDA, B, LDB, W, WORK,
        #      $                   LWORK, IWORK, LIWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, ITYPE, LDA, LDB, LIWORK, LWORK, N
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            IWORK( * )
        #       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), W( * ), WORK( * )
        function sygvd!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            w = similar(A, $elty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($sygvd), liblapack), Cvoid,
                    (Ref{BlasInt}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                     Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                     Ref{BlasInt}, Ptr{BlasInt}),
                    itype, jobz, uplo, n,
                    A, lda, B, ldb,
                    w, work, lwork, iwork,
                    liwork, info)
                chkargsok(info[])
                if i == 1
                    lwork = BlasInt(work[1])
                    resize!(work, lwork)
                    liwork = iwork[1]
                    resize!(iwork, liwork)
                end
            end
            chkposdef(info[])
            w, A, B
        end
    end
end

end # module
