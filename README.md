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
 - g - makes current gen best
 - f - ditches current gen

It's learning itself to "not die": there are starting parameters and simulation runs. When it dies ( or is "exploding") it's survival time is saved. If new generation (random changes are introduced) outlives previous, it's parameters are being saved and passed on.

![example](https://github.com/HVisMyLife/lenia-rust/blob/master/recording.gif)

For now, there are two kernels:
![example](https://github.com/HVisMyLife/lenia-rust/blob/master/kernels.png)
