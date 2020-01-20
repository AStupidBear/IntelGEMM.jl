using GEMM
using LinearAlgebra
using Test

for T in (Float32, Float64)
    A = T[2 1; 4 3]
    Bv = T[1; 2]
    B = reshape(Bv, :, 1)
    Cv = T[4; 10]
    C = reshape(Cv, :, 1)
    @test A * Bv  == Cv
    @test A * B == C
    eigen(Symmetric(A))
    eigen(A)
end