@testset "log signature workspace" begin
    for T in (Float32, Float64), method in 0:3
        for shape in ((2, 5), (2, 5, 4), (2, 5, 3, 2))
            path = reshape(T.(sin.(1:prod(shape))), shape)
            ws = LogSigWorkspace(path, 3; method)
            expected = log_sig(path, 3; method)
            first_result = log_sig!(ws, path)
            @test first_result === ws.result
            @test first_result ≈ expected

            changed = path .* T(0.7)
            saved = copy(first_result)
            @test log_sig!(ws, changed) === first_result
            @test first_result ≈ log_sig(changed, 3; method)
            @test saved ≈ expected

            clear_log_sig_cache()
            @test log_sig!(ws, path) ≈ expected
            @test log_sig!(ws, path; n_jobs=2) ≈ expected
        end

        storage = reshape(T.(cos.(1:120)), 2, 5, 3, 4)
        path = @view storage[:, 1:2:5, 3:-1:1, 1:2:4]
        ws = LogSigWorkspace(path, 3; method)
        @test log_sig!(ws, path) ≈ log_sig(path, 3; method)
        changed = @view storage[:, 1:2:5, 3:-1:1, 2:2:4]
        @test log_sig!(ws, changed) ≈ log_sig(changed, 3; method)

        for path in (zeros(T, 2, 3), zeros(T, 2, 3, 0))
            ws = LogSigWorkspace(path, 0; method)
            @test log_sig!(ws, path) == log_sig(path, 0; method)
        end
        empty_path = zeros(T, 2, 3, 0)
        @test log_sig!(LogSigWorkspace(empty_path, 3; method), empty_path) == log_sig(empty_path, 3; method)
        if method != 3
            point = reshape(T[0.2, -0.3], 2, 1)
            @test log_sig!(LogSigWorkspace(point, 3; method), point) ≈ log_sig(point, 3; method)
        end
    end

    path = zeros(3, 32, 4)
    ws = LogSigWorkspace(path, 5)
    @test_throws DimensionMismatch log_sig!(ws, zeros(3, 31, 4))
    @test_throws DimensionMismatch log_sig!(ws, zeros(3, 128))
    @test_throws ArgumentError log_sig!(ws, Float32.(path))
    @test_throws ArgumentError log_sig!(ws, path; n_jobs=0)
    @test_throws ArgumentError LogSigWorkspace(path, -1)
    @test_throws ArgumentError LogSigWorkspace(path, 3; method=4)
    @test_throws ArgumentError LogSigWorkspace(zeros(2, 0), 3)
    @test_throws ArgumentError LogSigWorkspace(zeros(0, 3), 3)
    @test_throws ArgumentError LogSigWorkspace(zeros(3), 3)
    @test_throws TypeError LogSigWorkspace(zeros(Int, 2, 3), 3)
end

@testset "workspace allocations" begin
    function allocated_call(ws, path)
        log_sig!(ws, path)
        return @allocated log_sig!(ws, path)
    end
    for T in (Float32, Float64), method in 0:3
        path = reshape(T.(sin.(1:384)), 3, 32, 4)
        ws = LogSigWorkspace(path, 5; method)
        allocated_call(ws, path)
        @test allocated_call(ws, path) == 0
    end
end

@testset "workspace cache concurrency" begin
    path = reshape(sin.(1:120), 6, 5, 4)
    expected = log_sig(path, 3; method=2)
    tasks = map(1:4) do _
        Threads.@spawn begin
            ws = LogSigWorkspace(path, 3; method=2)
            for _ in 1:8
                clear_log_sig_cache()
                yield()
                log_sig!(ws, path) ≈ expected || error("Concurrent workspace result mismatch")
            end
            true
        end
    end
    @test all(fetch, tasks)
end
