# HLX FSR 4 Full: INT8 kernels for RDNA 2 / RDNA 3

The source of the faster, exact forms of AMD's FSR 4.1.2 shaders that HLX FSR 4 Full (part of HelixSR) runs on GPUs with
int8 dot products: AMD RDNA 2 and RDNA 3 (desktop and integrated), the Steam Deck, and any GPU when `Int8 = 1` is set.

Every kernel here is **bit-exact** with the AMD shader it replaces: same numbers in every output, in every frame. What
changes is how the work is laid out on the GPU.

| part | AMD's shaders | these kernels |
|---|---|---|
| network, 12 passes | 1024 µs | **939 µs** (every pass faster) |
| postpass | 322-329 µs | **276 µs** |
| prepass | 318 µs | **267 µs** |

Timed at 1080p in the speed test on a BC-250 (RDNA 1.5). The BC-250 has no int8 dot product, so the network and postpass
times use a stand-in instruction of the same cost (wrong numbers, right timing; `PROXY=1` builds those forms). Timings on
real RDNA 2 / RDNA 3 GPUs are welcome.

## What is in here

| file | builds |
|---|---|
| `vk/gen_cnxi8.py` | ConvNeXt blocks (passes 1, 2, 12): 64 pixels of a row per group, one pixel per lane, the 16 channels per lane, weights as constants in the code |
| `vk/gen_fpi8.py` | FasterNet blocks with 32 channels (passes 4, 5, 10, 11): one pixel per lane, constant weights |
| `vk/gen_fni8.py` | FasterNet blocks with 64 channels (passes 7, 8, 9): four waves per 32 pixels |
| `vk/gen_dsi8.py` | the downscale passes (3, 6) |
| `vk/gen_posti8.py` | the postpass: last ConvNeXt block + 2x2 transposed convolution on int8 dot products, then the reconstruction; the tone-mapped render-resolution colour of the group's area is kept in workgroup memory |
| `pre/genpre_px.py` | the prepass: 128-thread groups of 16 x 8 output pixels; each pixel's 7 features go to workgroup memory as packed halves, then a wave per 4 output channels with wave-uniform weights |
| `vk/glsl2hlsl.py` | the network kernels are written as GLSL-style templates and turned into HLSL for DXC |
| `ref/` | a CPU interpreter for AMD's shaders (`dxil_interp.py`) and the model of each layer type, used to read the weights, scales and tensor layout out of AMD's shaders |
| `prepare/` | extracts AMD's shaders from AMD's DLL and prepares the pass offsets and weight tables |
| `data/` | the four layout families (1080p / 4K layout, standard / Ultra Performance model) and which postpass / prepass variant belongs to which runtime set |

No AMD shader or weight is included: `prepare/` reads them from AMD's own DLL.

## Build

Needs Python 3 with numpy, DXC 1.9 (the DirectX Shader Compiler), and `amd_fidelityfx_upscaler_dx12.dll` from AMD's
FidelityFX SDK 2.4.0 (MIT license; `Kits/FidelityFX/signedbin`, FSR 4.1.2.4555).

```
export AMD_FSR_DLL=/path/to/amd_fidelityfx_upscaler_dx12.dll DXC=/path/to/dxc
prepare/prepare.sh        # AMD's shaders -> work/ (a few minutes)
./build.sh out            # the kernels -> out/
```

`out/` then holds:

- `i8_<family><pass>.dxil` + `i8_<family><pass>w.bin`: the network kernels; `i8repl.txt` lists, per kernel, the sha256 of
  the AMD shader it replaces, its dispatch size and its push constants (sizes and the scratch offsets of its tensors).
- `post/<sha256>.dxil`: the postpass for each of AMD's postpass variants, named after the AMD shader it replaces, with
  AMD's bindings.
- `pre/<sha256>.dxil`: the prepass the same way. It runs 16 x 8 pixel groups, so it is dispatched with twice AMD's number
  of group rows. Variants with non-linear colour keep AMD's shader.

These are the same files, byte for byte, that HLX FSR 4 Full ships.

## Checking exactness

HLX FSR 4 Full's speed test compares each picture with AMD's FSR 4 (0 dB distance when identical); these kernels give AMD's
picture in every test scene and over multi-frame runs. `ref/dxil_interp.py` runs one thread of any AMD shader on the CPU,
which is how the layouts and weights were read and how single outputs can be checked against a kernel.

One thing that matters for exactness: GPU compilers fuse a multiply and an add into one instruction, and which product gets
fused follows the order of the statements. The kernels therefore write sums in AMD's order (colour taps, luma sums, history
blend), and keep AMD's `(x / size) * size` as a divide and a multiply.

## License

GNU General Public License, version 2 or (at your option) any later version; see `LICENSE`. If you use these kernels or
this code in your own project, that project's source has to be available under the GPL too.

The weights and constants the build writes into the kernels come from AMD's FSR 4.1.2 (MIT license, Copyright Advanced
Micro Devices, Inc.); see `NOTICE`. Not affiliated with or endorsed by AMD.
