@testset "C API basics" begin
    @test sig_length(1, 0) == 0
    @test sig_length(1, 0; scalar_term=true) == 1
    @test sig_length(1, 1) == 1
    @test sig_length(1, 2) == 2
    @test sig_length(2, 2) == 6
    @test sig_length(2, 2; scalar_term=true) == 7
    @test sig_length(3, 4; time_aug=true, lead_lag=true, scalar_term=true) == 2801
    @test_throws ArgumentError sig_length(0, 2)
    @test_throws ArgumentError sig_length(2, -1)
end
