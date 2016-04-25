#!/bin/sh

pdflatex \\nonstopmode\\input final_report.tex
jupyter nbconvert \"Neural\ Algorithm\ of\ Style\"\ Notebook.ipynb --to pdf --output=neural-style-notebook
pdftk final_report.pdf neural-style-notebook.pdf cat output chodapp3-final-report.pdf

zip -9r chodapp3-cs6475-neural-style.zip \
    neural-algorithm-of-style-notebook-export.py \
    \"Neural\ Algorithm\ of\ Style\"\ Notebook.ipynb \
    french_park.jpg \
    1280px-Great_Wave_off_Kanagawa2b.jpg \
    caffenet/deploy.prototxt \
    caffenet/ilsvrc_2012_mean.npy
