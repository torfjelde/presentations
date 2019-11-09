using DrWatson
output_dir = joinpath(projectdir(), "assets")
mkpath(output_dir)

cd(output_dir)
mathjax_path = joinpath(output_dir, "MathJax-2.7.5")
if !ispath(mathjax_path)
    run(`wget https://github.com/mathjax/MathJax/archive/2.7.5.tar.gz -O $(mathjax_path).tar.gz`)
    run(`tar -xzf $(mathjax_path).tar.gz`)
end
revealjs_path = joinpath(output_dir, "reveal.js-3.8.0")
if !ispath(joinpath(output_dir, revealjs_path))
    run(`wget https://github.com/hakimel/reveal.js/archive/3.8.0.tar.gz -O $(revealjs_path).tar.gz`)
    run(`tar -xzf $(revealjs_path).tar.gz`)
end

cd(projectdir())


html_fname = joinpath(projectdir(), "presentation.html")
txt = read(html_fname, String)
open(html_fname, "w") do f
    write(f, replace(txt, "/home/tor/Projects/mine/cambridge-julia-meetup/presentations" => projectdir()))
end
