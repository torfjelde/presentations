using DrWatson
using ArgParse

# Handling the arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--use-cdn"
    help = "if active, will use CDNs for MathJax and Reveal.js"
    action = :store_true
end

parsed_args = parse_args(ARGS, s)

# Actual script
separator = repeat("-", 80)

output_dir = projectdir("assets")
mkpath(output_dir)

html_fname = projectdir("index.html")

revealjs_path = joinpath(output_dir, "reveal.js-3.8.0")
mathjax_path = joinpath(output_dir, "MathJax-2.7.5")

orig_path = "/home/tor/Projects/mine/presentations/cambridge-julia-meetup"
revealjs_orig_path = joinpath("/home/tor/Projects/mine/presentations/cambridge-julia-meetup", "reveal.js-3.8.0")
mathjax_orig_path = joinpath("/home/tor/Projects/mine/presentations/cambridge-julia-meetup", "MathJax-2.7.5")

revealjs_cdn = "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.8.0/"
mathjax_cdn = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/"

println() # separate a bit from the command

println("> Fixing references to \$HOME in index.html")
txt = read(html_fname, String)
open(html_fname, "w") do f
    write(f, replace(
        txt,
        orig_path => projectdir()
    ))
end

# replace MathJax.js and Reveal.js by local paths
txt = read(html_fname, String)
open(html_fname, "w") do f
    write(f, replace(
        replace(txt, Regex("file://($(mathjax_cdn)|$(mathjax_orig_path))") => mathjax_path),
        Regex("file://($(revealjs_cdn)|$(revealjs_orig_path))") => revealjs_path
    ))
end

if parsed_args["use-cdn"]
    println("> Using CDNs for MathJax and Reveal.js")
    println(separator)

    # replace local MathJax.js and Reveal.js with CDNs
    txt = read(html_fname, String)
    open(html_fname, "w") do f
        write(f, replace(
            replace(txt, Regex("file://($(revealjs_path)|$(revealjs_orig_path))") => revealjs_cdn),
            Regex("file://($(mathjax_path)|$(mathjax_orig_path))") => mathjax_cdn
        ))
    end
else
    # download MathJax.js and Reveal.js for local use
    if !ispath(mathjax_path)
        println("> Downloading MathJax v2.7.5")
        println(separator)

        run(`wget https://github.com/mathjax/MathJax/archive/2.7.5.tar.gz -O $(mathjax_path).tar.gz`)
        run(`tar -C $(output_dir) -xzf $(mathjax_path).tar.gz`)
    end

    if !ispath(joinpath(output_dir, revealjs_path))
        println("> Downloading Reveal.js v3.8.0")
        println(separator)

        run(`wget https://github.com/hakimel/reveal.js/archive/3.8.0.tar.gz -O $(revealjs_path).tar.gz`)
        run(`tar -C $(output_dir) -xzf $(revealjs_path).tar.gz`)
    end
    cd(projectdir())
end

println()
