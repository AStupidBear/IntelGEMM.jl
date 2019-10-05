using SGEMM
using Test

@test Float32[2 1; 4 3] * Float32[1; 2] == Float32[4; 10]