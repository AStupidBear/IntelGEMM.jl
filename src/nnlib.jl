for (gemm, elt) in NNlib.gemm_datatype_mappings
    @eval begin
        @inline function NNlib.gemm!(transA::Val, transB::Val,
                               M::Int, N::Int, K::Int,
                               alpha::$(elt), A::Ptr{$elt}, B::Ptr{$elt},
                               beta::$(elt), C::Ptr{$elt})
            # Convert our compile-time transpose marker to a char for BLAS
            convtrans(V::Val{false}) = 'N'
            convtrans(V::Val{true})  = 'T'

            if transA == Val(false)
                lda = M
            else
                lda = K
            end
            if transB == Val(false)
                ldb = K
            else
                ldb = N
            end
            ldc = M
            ccall((@blasfunc($(gemm)), libblas), Nothing,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Ref{$elt}, Ptr{$elt}, Ref{BlasInt},
                   Ptr{$elt}, Ref{BlasInt}, Ref{$elt}, Ptr{$elt},
                   Ref{BlasInt}),
                  convtrans(transA), convtrans(transB), M, N, K,
                  alpha, A, lda, B, ldb, beta, C, ldc)
        end
    end
end