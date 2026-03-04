using Documenter: Documenter, DocMeta, deploydocs, makedocs
using ITensorFormatter: ITensorFormatter
using SerializedArrays: SerializedArrays

DocMeta.setdocmeta!(
    SerializedArrays, :DocTestSetup, :(using SerializedArrays); recursive = true
)

ITensorFormatter.make_index!(pkgdir(SerializedArrays))

makedocs(;
    modules = [SerializedArrays],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "SerializedArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/SerializedArrays.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"]
)

deploydocs(;
    repo = "github.com/ITensor/SerializedArrays.jl", devbranch = "main", push_preview = true
)
