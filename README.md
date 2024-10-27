# lenia-rust

### It's a dev branch for new version, it's uncomparably more "powerful", but still needs polishing, and is generally a work in progress.

Lenia universe written in rust.

It's just implementation from this doc: https://arxiv.org/pdf/2005.03742.pdf

Controls:
 - a - pause simulation
 - up/down - select parameter
 - left/right - change parameter
 - s - save map
 - l - load saved map
 - g - makes current gen best
 - f - ditches current gen
 - t - toggle autotune

It's learning itself to "not die": there are starting parameters and simulation runs. When it dies ( or is "exploding") it's survival time is saved. If new generation (random changes are introduced) outlives previous, it's parameters are being saved and passed on.

Growth map function parameters and kernel radiuses are beeing saved to .toml and matrix values to .bin. 
The implementation is awful, but it works and I couldn't care less xD.
Changing kernel sizes is not supported on the fly ( lookup table need to be regenerated each time ), so at the time there are not present in UI.

Features ideas are greatly appreciated.

![example](https://github.com/HVisMyLife/lenia-rust/blob/master/recording.gif)

For now, there are two kernels:
![example](https://github.com/HVisMyLife/lenia-rust/blob/master/kernels.png)

Each kernel have it's own growth mapping function, which is defined by 3 parameters:
![example](https://github.com/HVisMyLife/lenia-rust/blob/master/growth_map.png)
