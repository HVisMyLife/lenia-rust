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

Growth map function parameters and kernel radiuses are beeing saved to .toml and matrix values to .bin. 

Features ideas are greatly appreciated.

![example](https://github.com/HVisMyLife/lenia-rust/blob/master/recording.gif)

For now, there are two kernels:
![example](https://github.com/HVisMyLife/lenia-rust/blob/master/kernels.png)

Each kernel have it's own growth mapping function, which is defined by 3 parameters:
![example](https://github.com/HVisMyLife/lenia-rust/blob/master/growth_map.png)
