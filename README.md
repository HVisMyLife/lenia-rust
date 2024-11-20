# lenia-rust

### It's a dev branch for new version, it's uncomparably more "powerful", but still needs polishing, and is generally a work in progress.

Lenia universe written in rust.

It's just implementation from this doc: https://arxiv.org/pdf/2005.03742.pdf

Controls:
 - p - pause simulation
 - up/down - select parameter
 - left/right - change parameter
 - pageUp/pageDown - change func shape
 - s - save configurations to file
 - l - load configurations from file
 - insert - duplicate config or layer
 - delete - remove config or layer
 - Esc - exit

Layer data, like function parameters, are saved to .toml and matrix values to .bin.
Additionally there is one correlation .toml, that have references to above files.

Features ideas are greatly appreciated.

![example](https://github.com/HVisMyLife/lenia-rust/blob/master/recording2.gif)
![example](https://github.com/HVisMyLife/lenia-rust/blob/master/recording.gif)

There are 5 possible function shapes, for use as kernels or growth maps (centering vertically around 0 possible via parameters):
![example](https://github.com/HVisMyLife/lenia-rust/blob/master/functions.png)
