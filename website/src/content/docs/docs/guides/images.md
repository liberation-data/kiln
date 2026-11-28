---
title: Images
description: Generate and edit images from Java with stable-diffusion.cpp - FLUX.2 [klein], Qwen-Image, Qwen-Image-Edit, Z-Image and ERNIE-Image - and pass any engine option through.
---

`fornax-sd` runs image (and [video](/docs/guides/video/)) diffusion models through
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp): FLUX.1 and FLUX.2, Qwen-Image and
Qwen-Image-Edit, Z-Image, ERNIE-Image, SD 1.x, 2.x, XL and 3, and more. The class is `DiffusionModel`.

A diffusion model is usually several files: the **diffusion model** itself, a **VAE** that turns latents into pixels,
and a **text encoder** (a language model, for the models on this page). You name each one in a `Config`, and the
model loads all of them in its constructor and keeps them loaded across calls.

For the models Fornax's catalogue knows (`flux2-klein`, `qwen-image`, `qwen-image-edit`, `z-image-turbo`, `ernie-image-turbo`), `Config.builder().model(name)`
names every file at once and downloads what's missing. Each file option also takes a
[model reference](/docs/guides/models/#by-reference) in place of a path, such as a GGUF quant from another repo.

## FLUX.2 [klein]

FLUX.2 [klein] base 4B is the model Fornax's demo paints its covers with, and a good first one to try. By name:

<!-- snippet: Images#flux-by-name -->
```java
try (var images = new DiffusionModel(DiffusionModel.Config.builder().model("flux2-klein").build())) {
    BufferedImage img = images.generate(new DiffusionModel.Request("a neon sign that says FORNAX", 768, 768, 12, 4.0, 7)).get(0);
}
```

Or file by file, after fetching them:

```sh
hf download black-forest-labs/FLUX.2-klein-base-4b-fp8 flux-2-klein-base-4b-fp8.safetensors
hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors
hf download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors
```

<!-- snippet: Images#flux -->
```java
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors").toString())
        .vae(HfCache.require("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors").toString())
        .llm(HfCache.require("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors").toString())
        .build();

try (var images = new DiffusionModel(config)) {
    //                                              prompt, width, height, steps, cfg, seed
    var request = new DiffusionModel.Request("a neon sign that says FORNAX", 768, 768, 12, 4.0, 7);
    BufferedImage img = images.generate(request).get(0);
    ImageIO.write(img, "png", new File("fornax.png"));
}
```

`generate` returns a `List<BufferedImage>`, one image unless you ask for more (below). Fornax's examples use 20 steps
at CFG 4.0 for klein; at 768×768 and 12 steps it took about 130 s on the Radeon 8060S with Vulkan.

The text encoder here is Qwen3 4B, from Z-Image Turbo's repository: FLUX.2 [klein] 4B and Z-Image use the same one.

## Qwen-Image

Qwen-Image 2512 is a much larger model, and slower. Its text encoder is Qwen2.5-VL 7B.

```sh
hf download Comfy-Org/Qwen-Image_ComfyUI split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors \
    split_files/vae/qwen_image_vae.safetensors split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
```

<!-- snippet: Images#qwen-image -->
```java
String repo = "Comfy-Org/Qwen-Image_ComfyUI";
var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require(repo, "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors").toString())
        .vae(HfCache.require(repo, "split_files/vae/qwen_image_vae.safetensors").toString())
        .llm(HfCache.require(repo, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors").toString())
        .build();

try (var images = new DiffusionModel(config)) {
    BufferedImage img = images.generate(new DiffusionModel.Request(
            "a hand-painted shop sign reading 'Fornax & Co. Pottery'", 1024, 1024, 20, 2.5, 7)).get(0);
}
```

Fornax's examples use 20 steps at CFG 2.5. At 1024×1024 that took about 13 minutes on the 8060S with Vulkan.

### Editing an image: Qwen-Image-Edit

Qwen-Image-Edit 2511 takes a reference image and an instruction. It uses Qwen-Image's VAE and text encoder:

```sh
hf download Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors
```

<!-- snippet: Images#edit -->
```java
import java.util.List;
import java.util.Map;

var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require("Comfy-Org/Qwen-Image-Edit_ComfyUI",
                "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors").toString())
        .vae(HfCache.require(repo, "split_files/vae/qwen_image_vae.safetensors").toString())
        .llm(HfCache.require(repo, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors").toString())
        .build();

BufferedImage photo = ImageIO.read(new File("vase.png"));
int w = photo.getWidth() / 16 * 16, h = photo.getHeight() / 16 * 16;   // a multiple of 16, near the original

try (var editor = new DiffusionModel(config)) {
    BufferedImage edited = editor.generate(new DiffusionModel.Request(
            "make the glaze deep cobalt blue", w, h, 20, 2.5, 7, List.of(photo), Map.of())).get(0);
}
```

`withRefImages` adds the reference images to a request (the eight-argument constructor takes them too, with a map
of extra options). FLUX.2 [klein] takes reference images the same way.

Qwen-Image-Edit 2511 needs its "zero conditioning timestep" mode. stable-diffusion.cpp turns it on when the
weights carry the `__index_timestep_zero__` marker, which Comfy-Org's files do. For other files, such as some GGUF
conversions, set it yourself with `.set("model_args", "qwen_image_zero_cond_t=true")` on the `Config`, or edits
come out much worse.

## Editing an image: img2img, inpainting and ControlNet

Any of these models can start from an image instead of from noise. All three inputs are fields on `Request`.

### img2img

`withInitImage(image, strength)` starts from your image. `strength` says how much of it to redraw: 0 keeps it as
it is, 1 ignores it, and 0.5 to 0.75 keeps the layout and changes the look. Leave out the strength to get the
engine's default of 0.75. The image is fitted to the request's width and height.

<!-- snippet: Images#img2img -->
```java
BufferedImage shop = ImageIO.read(new File("shop.png"));
var request = new DiffusionModel.Request("the same shop at night in heavy snow", 512, 512, 20, 4.0, 7)
        .withInitImage(shop, 0.6);
BufferedImage snowy = images.generate(request).get(0);
```

### Inpainting

Add a mask, the same size as the init image, and only its white parts are repainted. The rest of the image stays
as it was. With strength 1, the masked area is drawn from scratch.

<!-- snippet: Images#inpaint -->
```java
BufferedImage mask = ImageIO.read(new File("mask.png"));     // white where the bicycle was, black elsewhere
var request = new DiffusionModel.Request("a large terracotta pot", 512, 512, 20, 4.0, 7)
        .withInitImage(shop, 1.0)
        .withMaskImage(mask);
```

This works with any model, including FLUX.2 [klein] and Qwen-Image-Edit: the mask limits which part of the image
is denoised. Dedicated inpainting models (SD 1.5 inpaint, FLUX.1 Fill) also see the masked image, and blend the
edges better.

### ControlNet

A ControlNet steers the layout with a hint image, such as edges, a depth map or a pose. Name the ControlNet model
in the `Config`, and give each request a control image:

<!-- snippet: Images#controlnet -->
```java
var config = DiffusionModel.Config.builder()
        .diffusionModel(sdxl).vae(vae)
        .controlNet("controlnet-canny-sdxl.safetensors")
        .build();

var request = new DiffusionModel.Request("a glass teapot", 1024, 1024, 25, 6.0, 7)
        .withControlImage(cannyEdges, 0.8);     // strength defaults to 0.9
```

The control image is fitted to the request's width and height. The ControlNet has to match the base model family.
FLUX.1 Controls and Flex.2 take the control image directly and need no ControlNet file; they also ignore the control
strength.

Fornax has run SD 1.5 with the canny ControlNet (`comfyanonymous/ControlNet-v1-1_fp16_safetensors`); the example
below does the same:

```sh
./gradlew edit --args='shop.png out.png "a pottery shop, watercolour painting"' -Pmodel=sd15 -Pstrength=1.0 \
    -Pcontrol=canny.png -Pcontrolnet=control_v11p_sd15_canny_fp16.safetensors
```

### Settings that would be ignored are refused

stable-diffusion.cpp silently ignores some settings that don't fit together. Fornax refuses them instead, with an
`IllegalArgumentException` when you call `generate`:

- a strength without an init image (there is nothing to keep),
- a mask without an init image, or a mask that isn't the init image's size,
- a control strength without a control image,
- `strength`, `control_strength`, the images or `loras` in the `extra` map, when `Request` has a field for them.

## LoRA

A LoRA is a small adapter that teaches a model a style, a subject or a faster schedule. Give a request a list of
`Lora(file, weight)`; weight 1 is full strength.

```sh
hf download adirik/pixel-art-lora-flux.2-klein-4B pixelart_lora.safetensors     # Apache 2.0
```

<!-- snippet: Images#lora -->
```java
var pixelArt = new DiffusionModel.Lora(
        HfCache.require("adirik/pixel-art-lora-flux.2-klein-4B", "pixelart_lora.safetensors"), 1.0f);

var request = new DiffusionModel.Request("PXART4. A pixel art of a small pottery shop", 512, 512, 20, 4.0, 7)
        .withLoras(pixelArt);
```

LoRAs are per request, so one loaded model can render with and without them. A LoRA only works with the model
family it was trained for: SD 1.x/2.x/XL/3, FLUX.1, FLUX.2 (klein too), Qwen-Image and Qwen-Image-Edit, Z-Image,
Wan and others. stable-diffusion.cpp reads the common LoRA formats (kohya, diffusers and PEFT, ai-toolkit) and
applies a LoRA to the diffusion model, the text encoder or both, whichever it has weights for. Many LoRAs also
have a trigger word, like `PXART4` above, which goes in the prompt.

By default the engine merges a LoRA into the weights before sampling when it can, and applies it at run time
for quantized weights and for fp8 weights the backend can't merge into (E5M2 everywhere, E4M3 on Vulkan and
Metal). `.set("lora_apply_mode", "at_runtime")` (or `"immediately"`) on the `Config` chooses for you.

### Qwen-Image-Edit with the Lightning LoRA

The Lightning LoRAs cut Qwen-Image and Qwen-Image-Edit from 20 or more steps to 4, at CFG 1. Fornax compiles the same
program as sd-cli for this recipe, but hasn't rendered it on the Radeon 8060S: the bf16 Qwen-Image-Edit 2511 and
its text encoder take about 51 GB of the 64 GB the Vulkan driver lets the GPU use, which leaves too little to
compute with (see [engine notes](https://github.com/liberation-data/fornax/blob/main/docs/engine-notes.md)). A
smaller (fp8 or GGUF) file should fit. These are the
settings ComfyUI's Qwen-Image-Edit 2511 workflow uses:

```sh
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

<!-- snippet: Images#edit-lightning -->
```java
var lightning = new DiffusionModel.Lora(HfCache.require("lightx2v/Qwen-Image-Edit-2511-Lightning",
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"), 1.0f);

var request = new DiffusionModel.Request("make the sign say FORNAX", w, h, 4, 1.0, 1)
        .withRefImages(List.of(photo))
        .withLoras(lightning)
        .withExtra(Map.of("sample.flow_shift", 3.1, "sample.method", "euler", "sample.scheduler", "simple"));
```

## Upscaling

`Upscaler` runs an ESRGAN-family model, such as RealESRGAN_x4plus (BSD-3-Clause), over an image. It loads the
model once and keeps it loaded:

```sh
hf download Comfy-Org/Real-ESRGAN RealESRGAN_x4plus.safetensors
```

<!-- snippet: Images#upscale -->
```java
import org.liberationdata.fornax.sd.Upscaler;

try (var up = new Upscaler(HfCache.require("Comfy-Org/Real-ESRGAN", "RealESRGAN_x4plus.safetensors"))) {
    BufferedImage big = up.upscale(img, 4);      // 512×512 -> 2048×2048
}
```

The model's own scale (4 for x4plus, read from the weights) is what one pass gives; `upscale(img, 16)` runs it
twice. Factors that aren't a power of the scale are refused: resize the result for those. The model works over
tiles of 128 input pixels by default. `new Upscaler(path, tileSize)` changes that: smaller tiles use less memory
and keep each GPU dispatch short.

`upscaleAsync` runs on the upscaler's own thread, as `generateAsync` does (see [Threading](/docs/guides/threading/)).
stable-diffusion.cpp can't stop an upscale part way, so `close()` waits for a pass that has started.

## Z-Image

Z-Image Turbo is a distilled model: it needs only a few steps, at CFG 1. It shares Qwen3 4B with FLUX.2 [klein] and
uses the FLUX.1 VAE.

```sh
hf download Comfy-Org/z_image_turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors \
    split_files/vae/ae.safetensors split_files/text_encoders/qwen_3_4b.safetensors
```

<!-- snippet: Images#z-image -->
```java
String z = "Comfy-Org/z_image_turbo";
var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require(z, "split_files/diffusion_models/z_image_turbo_bf16.safetensors").toString())
        .vae(HfCache.require(z, "split_files/vae/ae.safetensors").toString())
        .llm(HfCache.require(z, "split_files/text_encoders/qwen_3_4b.safetensors").toString())
        .build();

try (var images = new DiffusionModel(config)) {
    BufferedImage img = images.generate(new DiffusionModel.Request(
            "a rain-slicked street at night, neon reflections, a sign that says FORNAX", 1024, 1024, 8, 1.0, 7)).get(0);
}
```

The catalogue knows it as `z-image-turbo`, so `fornax image z-image-turbo "..."`, the OpenAI-compatible server and
Fornax Studio use these settings (8 steps, CFG 1.0) unless told otherwise.

## ERNIE-Image-Turbo

Baidu's ERNIE-Image-Turbo (Apache-2.0) is distilled like Z-Image Turbo: 8 steps at CFG 1. It uses the FLUX.2 VAE
that klein uses and a Ministral 3 3B text encoder, here both as GGUFs:

```sh
fornax pull ernie-image-turbo     # or: hf download unsloth/ERNIE-Image-Turbo-GGUF ernie-image-turbo-Q8_0.gguf, and
                                  #     unsloth/Ministral-3-3B-Instruct-2512-GGUF Ministral-3-3B-Instruct-2512-Q8_0.gguf
fornax image ernie-image-turbo "a brick kiln at dusk in the Australian outback, warm light, photograph"
```

On the Radeon 8060S with Vulkan it draws 1024x1024 in about 60 s, against Z-Image Turbo's 100 s (bf16), and both
spelled a neon sign's word right in a test prompt.

## Options

`Request` sets what most models need: the prompt, the size, the number of steps, the CFG scale and the seed. Anything
else that stable-diffusion.cpp accepts goes in the request's `extra` map, under the engine's own key names (from its
[C API guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/c_api.md)). Values can be
`Boolean`, `Integer` or `Long` (set as an integer), any other `Number` (set as a float), `String` or `float[]`.
The engine doesn't convert between them, so a float option needs a floating-point value: `"strength", 1.0`, not
`"strength", 1`.

<!-- snippet: Images#extra -->
```java
var request = new DiffusionModel.Request("a neon sign", 1024, 1024, 20, 4.0, 7, List.of(), Map.of(
        "negative_prompt", "blurry, low quality",
        "sample.method", "euler",
        "batch_count", 2,                  // two images from one call
        "vae_tiling.enabled", true));      // decode in tiles, to use less memory
List<BufferedImage> two = images.generate(request);
```

Options for the model as a whole, such as the paths of extra files or how weights are stored, go in the `Config`,
with `set(key, value)`:

<!-- snippet: Images#config-set -->
```java
var config = DiffusionModel.Config.builder()
        .diffusionModel(flux).vae(vae).llm(textEncoder)
        .set("flash_attn", true)
        .build();
```

A key or value the engine rejects fails with an `Sd.SdException` that names the key. A Java type Fornax can't pass
(anything but `Boolean`, `Integer`, `Long`, another `Number`, `String` or `float[]`) fails with an
`IllegalArgumentException`.

## Progress and logs

The engine reports each sampling step. `onProgress` and `onLog` are process-wide.

<!-- snippet: Images#progress -->
```java
DiffusionModel.onProgress((step, steps, seconds) -> System.out.printf("step %d/%d (%.2f s/it)%n", step, steps, seconds));
DiffusionModel.onLog((level, text) -> System.err.print(text));    // levels: 0 debug, 1 info, 2 warn, 3 error
```

`modelVersion()` names the model stable-diffusion.cpp detected, which is a quick check that the files are the ones
you meant.

The engine is quiet by default, apart from errors. `FORNAX_VERBOSE=1` shows its info messages, and
`FORNAX_VERBOSE=debug` everything, including the program it compiles for each generation.

## Try it

```sh
./gradlew image --args='klein out.png "a neon sign that says FORNAX"' -Psteps=12 -Pwidth=768 -Pheight=768
./gradlew image --args='qwen-image out.png "a hand-painted shop sign"'
./gradlew image --args='qwen-edit out.png "make the glaze cobalt blue" vase.png'
./gradlew image --args='klein out.png "a fornax at night"' -Prepeat=2 -Pseed=3     # the second image reuses the loaded weights

# img2img, inpainting and LoRAs (klein by default; the size follows in.png)
./gradlew edit --args='in.png out.png "the same street in heavy snow"' -Pstrength=0.6
./gradlew edit --args='in.png out.png "a terracotta pot"' -Pstrength=1.0 -Pmask=mask.png
./gradlew edit --args='in.png out.png "PXART4. A pixel art shop"' -Pstrength=1.0 -Plora=pixelart_lora.safetensors:1.0
# an instruction edit with Qwen-Image-Edit 2511 and the Lightning LoRA
./gradlew edit --args='in.png out.png "make the sign say FORNAX"' -Pmodel=qwen-edit \
    -Plora=Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors:1 -Psteps=4 -Pcfg=1 -Pshift=3.1 \
    -Psampler=euler -Pscheduler=simple

./gradlew upscale --args='in.png out.png'          # RealESRGAN_x4plus; -Pfactor=16 -Ptile=64 -Pmodel=other.safetensors
```
