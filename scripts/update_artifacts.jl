# Import a release's checksummed native assets; runtime loading stays offline once installed.
using Downloads, TOML

length(ARGS) == 1 || error("Usage: julia scripts/update_artifacts.jl vVERSION")
tag = only(ARGS)
occursin(r"^v?\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$", tag) || error("Invalid release tag: $tag")
url = "https://github.com/daniil-shmelev/pySigLib/releases/download/$tag/Artifacts.toml"
destination = joinpath(@__DIR__, "..", "Artifacts.toml")
mktemp(dirname(destination)) do path, io
    close(io)
    Downloads.download(url, path)
    manifest = TOML.parsefile(path)
    isempty(get(manifest, "pysiglib_cpu", [])) && error("Release manifest has no pysiglib_cpu artifact")
    for entries in values(manifest), entry in entries
        occursin(r"^[0-9a-f]{40}$", entry["git-tree-sha1"]) || error("Invalid artifact tree hash")
        downloads = entry["download"]
        isempty(downloads) && error("Artifact has no downloads")
        for download in downloads
            startswith(download["url"], "https://github.com/daniil-shmelev/pySigLib/releases/download/$tag/") || error("Unexpected artifact URL")
            occursin(r"^[0-9a-f]{64}$", download["sha256"]) || error("Invalid artifact checksum")
        end
    end
    cp(path, destination; force=true)
end
println("Pinned native artifacts from $tag. Run Pkg.test() before committing the manifest.")
