using Documenter, SpinGlassEngine

_pages = [
    "Introduction" => "index.md",
    "User Guide" => "guide.md",    
    "API Reference" => "api.md"
]
# ============================

makedocs(
    sitename="SpinGlassEngine",
    modules = [SpinGlassEngine],
    pages = _pages
    )