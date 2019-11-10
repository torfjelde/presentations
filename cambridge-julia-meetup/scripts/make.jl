using DrWatson
separator = repeat("-", 80)

output_dir = joinpath(projectdir(), "assets")
mkpath(output_dir)

println() # separate a bit from the command

cd(output_dir)
mathjax_path = joinpath(output_dir, "MathJax-2.7.5")
if !ispath(mathjax_path)
    println("> Downloading MathJax v2.7.5")
    println(separator)

    run(`wget https://github.com/mathjax/MathJax/archive/2.7.5.tar.gz -O $(mathjax_path).tar.gz`)
    run(`tar -xzf $(mathjax_path).tar.gz`)

    println(separator)
end

revealjs_path = joinpath(output_dir, "reveal.js-3.8.0")
if !ispath(joinpath(output_dir, revealjs_path))
    println("> Downloading Reveal.js v3.8.0")
    println(separator)

    run(`wget https://github.com/hakimel/reveal.js/archive/3.8.0.tar.gz -O $(revealjs_path).tar.gz`)
    run(`tar -xzf $(revealjs_path).tar.gz`)

    println(separator)
end

cd(projectdir())

println("> Fixing references to \$HOME in presentation.html")
html_fname = joinpath(projectdir(), "presentation.html")
@info html_fname
txt = read(html_fname, String)
open(html_fname, "w") do f
    write(f, replace(txt, "/home/tor/Projects/mine/presentations/cambridge-julia-meetup" => projectdir()))
end

println()
