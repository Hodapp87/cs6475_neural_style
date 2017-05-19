# cs6475\_neural\_style

This was a final project for
[CS6475](http://www.omscs.gatech.edu/cs-6475-computational-photography)
at Georgia Tech.  Most of the material here is duplicated in
`final_report.tex` and in the Python notebook.

## Goal

The goal of this project was to create a literate implementation (as
in
[literate programming](https://en.wikipedia.org/wiki/Literate_programming))
of the algorithm described in the recent paper, [A Neural Algorithm of
Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys,
Alexander S. Ecker, and Matthias Bethge.

The paper is sufficiently well-known by now that it has many
open-source and commercial implementations that are quite good:

- https://github.com/jcjohnson/neural-style
- https://github.com/kaishengtai/neuralart
- https://github.com/andersbll/neural_artistic_style
- https://github.com/fzliu/style-transfer
- https://github.com/woodrush/neural-art-tf
- https://deepart.io

However, I found that many of these lacked clear explanations on why
they were implemented how they were.  The hope was that students in
CS6475 (and maybe
[CS4495/CS6476](http://www.cc.gatech.edu/~hays/compvision/)) could
understand and use this implementation, starting from their existing
familiarity with Python, NumPy, SciPy, and OpenCV in the algorithms of
computational photography.

## Notebook

The eventual result was an IPython notebook (via
[Jupyter](https://jupyter.org/)) which gives a simplified (but still
functional) example of how to actually implement this algorithm with
[Caffe](http://caffe.berkeleyvision.org/).

That notebook is available in
["Neural Algorithm of Style" Notebook.ipynb](./%22Neural%20Algorithm%20of%20Style%22%20Notebook.ipynb)
in this repository.  The conversion of this to
a PDF is available in
[neural-style-notebook.pdf](./neural-style-notebook.pdf).
