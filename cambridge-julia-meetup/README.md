# You have data and I have distributions: a talk on `Turing.jl` and `Bijectors.jl`
This contains the files used in my (informal) presentation at the [2nd Julia meetup in Cambridge](https://www.meetup.com/London-Julia-User-Group/events/265586612/).

You can view the Reveal.js presentation [here](https://torfjelde.github.io/presentations/cambridge-julia-meetup/).

Unfortunately, I haven't been able to export this to a PDF due to running out of memory, though my intention was to do so. Sorry!

## Setup
The source of the presentation can be found in `presentation.org`, and the actual presentation is `presentation.html` which I've generated using the magnificent [ox-reveal.el](https://github.com/yjwen/org-reveal) package in Emacs, resulting in a presentation viewable in the browser (using the Javascript [Reveal.js](https://github.com/hakimel/reveal.js/).

For those *without* Emacs; **get it!** 

Nah, I'm kidding:) Do the following
```sh
julia --project -e "using Pkg; Pkg.instantiate(); include(\"scripts/make.jl\")"
```
This will download `MathJax` and `Reveal.js`, and then fix the local references in `presentation.html`. Then just open `presentation.html` in your browser and you're good to go!

To generate from source, you need the `ox-reveal.el` package and Emacs (I'm using Emacs 26.2). Then do
```emacs-lisp
(load-package 'ox-reveal)
```
open `presentation.org`, then do `C-c C-e` followed by `R B` to export and browse.

## Disclaimer
Under no circumstance do I take any responsibility for monetary loss in the process of running an ice-cream parlour. The strategy outlined is by no means a guaranteed success despite the wording used in the presentation.

A (slightly) more serious disclaimer: this is my first "proper" presentation and so I have not yet settled on a proper structure/approach for presentations. The project you're seeing here is the glorious first attempt in all its glory. Enjoy.
