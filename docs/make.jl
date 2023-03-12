using Documenter, SpinGlassEngine

_pages = [
    "Introduction" => "index.md",
    "User Guide" => "guide.md",    
    "API Reference" => "api.md"
]
# ============================

format = Documenter.HTML(edit_link = "master",
                         prettyurls = get(ENV, "CI", nothing) == "true",
)

# format = Documenter.LaTeX(platform="none")

makedocs(
    sitename="SpinGlassEngine.jl",
    modules = [SpinGlassEngine],
    pages = _pages,
    format = format
    )