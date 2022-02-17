using Documenter, SpinGlassEngine
# makedocs(
#     modules=[SpinGlassEngine],
#     sitename="SpinGlassEngine.jl",
#     format=Documenter.LaTeX()
# )
makedocs(
    modules=[SpinGlassEngine],
    sitename="SpinGlassEngine.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    )
)
deploydocs(
    repo="github.com/euro-hpc-pl/SpinGlassEngine.jl.git",
    devbranch="chimera_to_publish-doc"
)
