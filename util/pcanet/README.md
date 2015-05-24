

paper: http://arxiv.org/pdf/1404.3606v2.pdf

original code from: https://github.com/ldpe2g/pcanet


* changed everything to float Mat's (lower memory footprint)
* removed the indexing
* enabled variable stage count
* removed various memleaks
* removed some functions (heaviside is actually just a threshold operation in cv)
* curious, if replacing the pca learning with e.g. predetermined wavelets will bring some gain ?
