# lenia-rust
Lenia written in rust.

Work in progress...

It's just implementation from this doc: https://arxiv.org/pdf/2005.03742.pdf

Controls:
 - a - pause simulation
 - up/down - select parameter
 - left/right - change parameter
 - s - save map
 - l - load saved map

It's learning itself to "not die": there are starting parameters and simulation runs. When it dies ( or is "exploding") it's survival time is saved. If new generation (random changes are introduced) outlives previous, it's parameters are being saved.

![example](https://github.com/HVisMyLife/lenia-rust/blob/master/recording.gif)
