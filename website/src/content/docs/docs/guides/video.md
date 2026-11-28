---
title: Video
description: Generate video with sound from Java with LTX-2.5 through stable-diffusion.cpp - a one-stage draft, the two-stage recipe, first and last frames, and long clips.
---

`DiffusionModel` makes video as well as [images](/docs/guides/images/), through stable-diffusion.cpp's video
pipeline. This page covers **LTX-2.5**, which generates picture and sound together. The same API runs Wan and the
other video models stable-diffusion.cpp supports.

Video is the heaviest thing Fornax does. LTX-2.5 is a 22B model with a 12B text encoder, and a render takes minutes
even at small sizes. Start small (see [Render small first](#render-small-first)).

## The files

Fornax's example uses ComfyUI's int8 build of LTX-2.5 from `Lightricks/LTX-2.5`:

```sh
hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
    vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

The upscaler is only for the [two-stage recipe](#the-two-stage-recipe). For the larger, slower `dev` model, swap
`distilled` for `dev` in the first file name.

## A clip with sound

`Config.builder().model("ltx-2.5-distilled")` (or `"ltx-2.5-dev"`) names the four model files, downloading what's
missing (the upscaler too, for a request's `hires.model_path`). The repo is
gated, so accept its terms and set `HF_TOKEN` first. File by file, as Fornax's example does:

<!-- snippet: Video#ltx -->
```java
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.sd.DiffusionModel;

import java.util.Map;

String ltx = "Lightricks/LTX-2.5";
var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require(ltx, "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors").toString())
        .llm(HfCache.require(ltx, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors").toString())
        .vae(HfCache.require(ltx, "vae/ltx-2.5-video-vae-conv-bf16.safetensors").toString())
        .set("audio_vae_path", HfCache.require(ltx, "vae/ltx-2.5-audio-vae-bf16.safetensors").toString())
        .build();

try (var model = new DiffusionModel(config)) {
    //                                                    prompt, width, height, frames, fps, steps, cfg, seed
    var request = new DiffusionModel.VideoRequest("a potter at a wheel, warm light, the hum of the wheel",
            768, 448, 49, 24, 8, 1.0, 7)
            .withExtra(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"));
    DiffusionModel.Video video = model.generateVideo(request);
    System.out.printf("%d frames at %d fps (%.1f s)%s%n", video.frames().size(), video.fps(), video.seconds(),
            video.audio() != null ? ", with sound" : "");
}
```

`generateVideo` returns a `Video`: its `frames` as `BufferedImage`s, its `fps`, and its `audio` (an `Audio`, the same
type the song and speech models return, or `null` for a model without sound). Fornax doesn't encode video files; write
the frames and the audio out and join them with ffmpeg, as the example does.

The distilled model takes 8 steps at CFG 1; `dev` takes about 20 at CFG 3.

**Keep the audio VAE.** LTX-2.5 was trained to make sound and picture together. Leaving `audio_vae_path` out removes
the soundtrack, but that isn't how the model was trained to run.

**bf16 on drivers without it.** ComfyUI's int8 checkpoint keeps a few tensors in bf16. Up to 0.2.0 they had to be
widened to f32 with `tensor_type_rules` on GPUs whose Vulkan driver has no bf16 (the Radeon 8060S's RADV). From 0.3.0,
stable-diffusion.cpp loads the one that broke the run, the keyframe position embedding, as f32 itself
(leejet/stable-diffusion.cpp#2432), so no rules are needed.

## First and last frames

Pin the first frame, the last frame, or both, with an image:

<!-- snippet: Video#keyframes -->
```java
import javax.imageio.ImageIO;
import java.io.File;

var request = new DiffusionModel.VideoRequest("the vase turns slowly on the wheel", 768, 448, 49, 24, 8, 1.0, 7)
        .withExtra(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"))
        .withInitImage(ImageIO.read(new File("first.png")))
        .withEndImage(ImageIO.read(new File("last.png")));
```

The `VideoRequest` methods return a new request each time (it is a record), so chain them. `withRefAudio` passes a
reference clip to models that are driven by speech (such as Wan2.2 S2V).

## Sing to a song (audio to video)

Give LTX a soundtrack and it generates the picture to it: a singer's mouth follows the words, a hand strums on the
beat. `withAudio` cuts the clip-length section from a start time (in seconds) and holds it as the soundtrack:

<!-- snippet: Video#audio -->
```java
import java.nio.file.Path;

var request = new DiffusionModel.VideoRequest(
        "close-up of a singer at a vintage microphone, his jaw opening and closing in time with the words. "
                + "Audio: a man singing a country song.", 768, 448, 97, 24, 8, 1.0, 7)
        .withExtra(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"))
        .withInitImage(ImageIO.read(new File("singer.png")))
        .withAudio(Path.of("song.wav"), 25.9);   // 97 frames at 24 fps: 25.9 s to 29.9 s
var video = model.generateVideo(request);     // video.audio() is the song section
```

The song is encoded with LTX's audio VAE and held fixed in the joint audio and video latent while the picture is
denoised, as ComfyUI's LTX audio-to-video workflow does. So the clip's soundtrack is the section you gave, through the
audio VAE (close to the original, not bit for bit: lay the original over the final cut if that matters). It needs the
audio VAE (`audio_vae_path`) and works with first and last frames and with the two-stage recipe. Endless Country drove
its lip-sync shots with a vocals-only stem (split from the mix) and laid the full mix over the cut.

LTX was trained on compressed frames. ComfyUI's LTX workflow passes each keyframe through H.264 at CRF 18 first, and
Fornax's example can do the same (`-Ppreprocess=18`), though in tests it made no visible difference.

## The two-stage recipe

For a sharper result, render a draft at half size, upscale its latent with LTX's learned ×2 upscaler, then refine it
at full size. This is ComfyUI's "Image to Video (LTX-2.5)" workflow, step for step:

| ComfyUI | Fornax |
|---|---|
| Stage 1: half size, `ManualSigmas` 1.0 … 0.421875, 0, euler_ancestral, CFG 1 | `sample.sigmas`, `sample.method=euler_a` |
| `LTXVImgToVideoInplace` strength 0.7 | `strength=0.7` |
| `LTXVPreprocess` 18 | an ffmpeg H.264 round trip at CRF 18 |
| `LTXVLatentUpsampler` with the ×2 model | `hires.upscaler=model` plus `hires.model_path` |
| Stage 2: sigmas 0.85, 0.725, 0.4219, 0, first frame at 1.0 | `hires.sigmas`; stable-diffusion.cpp holds the first frame fully in the refine |
| Tiled VAE decode | `vae_tiling.enabled` |

In Java, that is a request with more options, at the draft size:

<!-- snippet: Video#hires -->
```java
import java.util.HashMap;

var extra = new HashMap<String, Object>();
extra.put("sample.method", "euler_a");
extra.put("sample.scheduler", "ltx2");
extra.put("sample.sigmas", new float[] {1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f});
extra.put("strength", 0.7);
extra.put("hires.enabled", true);
extra.put("hires.upscaler", "model");
extra.put("hires.model_path", HfCache.require(ltx, "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors").toString());
extra.put("hires.sigmas", new float[] {0.85f, 0.725f, 0.4219f, 0.0f});
extra.put("vae_tiling.enabled", true);    // short GPU dispatches stay under the driver's compute timeout

var request = new DiffusionModel.VideoRequest("a potter at a wheel, warm light", 672, 448, 121, 24, 8, 1.0, 7)
        .withInitImage(ImageIO.read(new File("first.png")))
        .withExtra(extra);
```

The output is twice the draft's size: 1344×896 here.

:::caution[`hires.model_path` needs `hires.upscaler = "model"`]
stable-diffusion.cpp only runs a learned upscaler when `hires.upscaler` is `"model"`. Otherwise it interpolates the
latent and silently ignores `hires.model_path`, and refining an interpolated video latent ghosts badly. Fornax selects
the model for you when you give a model path without an upscaler, and refuses a model path with any other upscaler.
:::

To tell the stages' artifacts apart, render the draft alone with `"hires.enabled", false` (`-Phires=false` in the
example).

## Long clips

On Vulkan, a long clip can lose the GPU. A 13 s LTX clip (313 frames) refined at 1344×896 lost the device on its first
refine step: its attention and feed-forward layers over 47,040 tokens each ran as one GPU dispatch, longer than the
amdgpu driver's 2 s compute timeout. Fornax's stable-diffusion.cpp now splits that work into short chunks with the
same output, by default only on Vulkan (`SD_FA_CHUNK_FLOPS`, `SD_FF_CHUNK_FLOPS`); for a long clip on Vulkan also pass
`-Dfornax.vulkan.maxNodesPerSubmit=1`. See [Troubleshooting](/docs/troubleshooting/#device-lost-or-ring-timeout-on-long-video).

Long clips used to come out wrong on Vulkan: from 257 frames at 1344×896 the two-stage recipe gave a hazy, mottled
clip and the sound was off ([#112](https://github.com/liberation-data/fornax/issues/112)). A flash-attention sum
overflowed 16-bit floats in the cross-attention between sound and picture. Fornax's engines now keep that sum in 32-bit
floats, and a 257-frame Vulkan render matches HIP, at no measurable cost in speed.

HIP (ROCm) is still the better backend for long clips. It doesn't have that timeout: its work goes through user
queues that the timeout doesn't cover. It was also about 2.7× faster per LTX step on the Radeon 8060S (44 s against
119 s at 672×448×313, after a slower first step). Build it with `native/build.sh hip`; see
[Choosing a GPU backend](/docs/guides/gpu-backends/#hip-rocm). Fornax turns off ggml's HIP graphs by default, which
would otherwise overflow a JVM thread's stack on LTX.

For shorter clips on Vulkan, tiled VAE decoding and `-Dfornax.vulkan.maxNodesPerSubmit=8` (`-Pfornax.vulkan.maxNodesPerSubmit=8`
on `./gradlew video`) keep each GPU submission short enough. See [Troubleshooting](/docs/troubleshooting/#long-gpu-work-loses-the-device).

## Render small first

336×224 and 49 frames took about 3 minutes, and showed every artifact that the 1344×896 renders did (those took 10 to
30 minutes). Get the prompt, the keyframes and the settings right small, then render big.

## Try it

```sh
./gradlew video --args='clip.mp4 "a potter at a wheel, warm light"'
./gradlew video --args='clip.mp4 "the vase turns slowly" first.png last.png' -Pwidth=672 -Pheight=448 -Pframes=121
./gradlew video --args='clip.mp4 "a potter at a wheel" first.png' -Precipe=two-stage
./gradlew video --args='clip.mp4 "a potter at a wheel"' -Pmodel=dev -Paudio=false
./gradlew video --args='clip.mp4 "a singer at a microphone, singing" singer.png' -Psong=song.wav -Psong.start=25.9 -Pframes=97
```

The example writes the frames and the soundtrack beside the `.mp4` and joins them with ffmpeg (which must be on the
`PATH` for the `.mp4`). `-Pltx.dir=<folder>` reads the files from a folder laid out like the repository
(`diffusion_models/`, `text_encoders/`, `vae/`) instead of the Hugging Face cache.
