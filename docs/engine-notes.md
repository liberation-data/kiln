# Engine notes

What we learned getting the engines to behave, so the next person doesn't relearn it. Add to it when an engine
surprises you.

## How to debug a model that looks wrong

1. **Compare against the engine's own tool first.** Before you guess at causes, run the same settings through the
   engine's CLI (`sd-cli` for stable-diffusion.cpp). If the CLI is right and Fornax is wrong, it's Fornax.
   `scripts/compare-with-sdcli.sh` diffs the program stable-diffusion.cpp compiles for each side, in seconds and
   without rendering.
2. **Render small first.** 336×224 and 49 frames took about 3 minutes and showed every artifact the
   1344×896 renders did (those took 10 to 30 minutes).
3. **Split a pipeline into its stages.** For LTX, a draft-only render (`-Dhires=false`) showed that the draft was
   clean and the refine wasn't.

## stable-diffusion.cpp

- **The hires upscaler needs `hires.upscaler = "model"`.** Without it, the latent is interpolated (bilinear by
  default) and `hires.model_path` is silently ignored. Refining an interpolated video latent produces ghosting and
  ghost structures. `DiffusionModel` now selects the model when a model path is given, and refuses contradictory
  settings.
- **bf16 on RADV.** The Radeon 8060S (Strix Halo) Vulkan driver reports `bf16: 0`, so bf16 ops fall back to the
  CPU, which has no f32 + bf16 add, and an LTX-2.5 run aborted. The culprit was one tensor,
  `keyframes_abs_pos_embedding`, which took its bf16 type from the file; biases were already loaded as f32, and
  `tensor_type_rules` skip names containing `embedding`. Fornax carried a patch (`0001`, bf16→f32 rules for any
  tensor) and widened ~0.9 GB of LTX tensors until leejet/stable-diffusion.cpp#2432 (from Fornax's report) created that
  tensor as F32 upstream. Since the pin includes it, LTX renders on RADV with no patch and no rules (checked at
  336×224, 49 frames, with audio).
- **Long GPU submissions hit the amdgpu compute timeout.** ggml-vulkan puts up to 100 graph nodes in each
  submission. One big decode (LTX's audio VAE at full size) can exceed the kernel's timeout and lose the device.
  `GGML_VK_MAX_NODES_PER_SUBMIT=8` (`-Dfornax.vulkan.maxNodesPerSubmit=8`) fixed it. ComfyUI never hit this, because PyTorch on ROCm launches many small
  kernels.
- **Long video on Vulkan hits the timeout even at 8 nodes per submission.** A 13 s LTX clip (313 frames) refined at
  1344×896 lost the device on the first refine step. At 47,040 video tokens single dispatches outlast amdgpu's 2 s job
  timeout, and no submission size can split one node: the video self-attention (one `FLASH_ATTN_EXT` node, 3.6e13 FLOP,
  4.5 to 7.6 s) and each int8 feed-forward matmul (6.3e12 ops: about 2.7 s up, 1.8 s down). The perf logger doesn't
  time the int8 matmuls, which are sd.cpp extension ops; a measurement-only build timed each extension dispatch on its
  own (flush, wait idle, dispatch, wait idle). At 4,480 tokens the up-projection (4096→16384) took 258 ms (2.3 TOPS,
  454 ms at worst), the down 174 ms, q/k/v/out 25 ms; time is linear in tokens.
  `native/patches/stable-diffusion.cpp/0003-chunk-long-dispatches.patch` splits both:
  - attention, in `ggml_ext_attention_ext`: past `SD_FA_CHUNK_FLOPS` (default 1e12) Q is cut into row chunks, each its
    own `ggml_flash_attn_ext` node over the same K and V, and the outputs are concatenated (a mask shared by all heads
    is sliced with the rows; any other mask keeps one node). LTX's self-attention ran at 5.9 TFLOPS at 1,120 tokens and
    8.1 at 4,480, so a 1e12 chunk is 0.12 to 0.21 s;
  - linear layers, in `Linear::forward`, and whole feed-forwards (up, activation, down), in `FeedForward::forward`:
    past `SD_FF_CHUNK_FLOPS` (default 4e11, per matmul) the token rows are cut into chunks of whole 128-row tiles.
    At 47,040 tokens that is 16 feed-forward chunks of 2,944 rows, measured at that shape at 188 ms up (271 ms at
    worst) and 117 ms down, and 4 chunks for each attention projection (~65 ms). Layers reading one input (q, k, v)
    share its row views, so its int8 quantization still runs once per chunk.
  A final chunk under 64 rows is merged into the one before, so no chunk is small enough to take another kernel
  variant (split_k, mul_mat_vec, scalar paths). Both budgets are on by default only when the runner's backend is
  Vulkan (`sd_backend_is(backend, "Vulkan")`): HIP, CUDA and Metal have no 2 s job limit, and there the split would
  only change kernel choices (not bit-identical), add up to ~1.5 GB of peak memory on big layers and break Vulkan's
  int8 fusions for nothing. Setting either variable to a positive value turns that split on for any backend; `0`, a
  negative or an unparsable value turns it off everywhere. The attention chunk count is capped at its query rows.
  Rows are independent (the int8 activations are quantized per row), so the output is the same: 512×320×49 and
  1024×640×49 clips came out bit-identical, frames and audio, with every split forced small (down to 128-row
  feed-forward chunks) and with the defaults. Only those budgets were checked: much smaller ones also chunk small
  layers and the flash-attention auto probe, which can then take other kernels or the other attention path, so they
  aren't guaranteed bit-identical. At 1024×640×49 the defaults (two feed-forward chunks) left steps at
  39.4 s against 40.2 s whole; forcing 6 feed-forward, 2 projection and 4 attention chunks made them 43.8 s. An
  `expand_graph` per attention chunk, as MiniMax-H3's head chunks use, cost 2 to 3 s a step, so the patch has none.
  The patch is to be proposed upstream. HIP has no such timeout (ROCm work goes through KFD user queues, which is why
  ComfyUI never hit it) and was about 2.7× faster per LTX step than Vulkan on the 8060S (44 s against 119 s at
  672×448×313, after a slower first step), so it stays the better backend for long clips.
- **Vulkan flash attention overflowed f16 on long video (#112).** With the chunking above, the two-stage LTX recipe
  survived at 1344×896×313, but from 257 frames (38,808 refine tokens) the Vulkan output was haze and the audio
  decorrelated, while 241 frames looked clean. HIP was right. ggml-vulkan's flash-attention shaders kept the
  unnormalised sum of P·V in `O_TYPE`, which was f16 in the fp16 build even when F32 precision was asked for (that only
  widened the scores and row sums). LTX's video-to-audio cross-attention has 268 audio queries over 38,808 video keys
  and often spreads evenly over them, so the sum reaches about n_kv × |mean V| (up to 7e5 here) and overflows 65504 to
  inf, which the shader clamps. From block 10 the audio stream blew up (std 15 → 2,773) and the audio-to-video attention
  carried it into the picture. It depends on the data, not on a fixed size: the same node fails on its dumped inputs at
  36,456 keys too. `native/patches/*/0004-vulkan-fa-f32-output-accumulator.patch` makes `O_TYPE` follow `ACC_TYPE` (f32
  unless f16 accumulation is allowed) on the scalar, coopmat1 and coopmat2 paths, in every engine's ggml (all six
  carried the f16 accumulator; in llama.cpp's newer coopmat1 the patch also stops packing PV tiles into f16 pairs).
  After it a 257-frame Vulkan render matches HIP (PSNR 25.3 against 21.8 before, audio correlation 0.58 against 0.04).
  On the 8060S the wider accumulator costs nothing measurable: a 97-frame two-stage render (672×448 → 1344×896) took
  1,120.9 s against 1,118.3 s (refine steps 150.9 s against 150.5 s), and llama.cpp prompt processing on Qwen3-0.6B
  (`llama-bench -fa 1`) ran at 8,065 against 8,082 tokens/s at 4k and 5,999 against 6,016 at 8k. Upstream
  ggml-org/llama.cpp master still has the f16 accumulator (2026-11-28).

  How it was found, for next time: the per-block CPU recompute from the GPU's own inputs was too noisy to tell (HIP
  against the CPU differs as much), and the first checks only looked at the video stream. Feeding one backend's dumped
  graph inputs to the other (HIP) and comparing the block outputs found block 10's audio at once, and a single-node CPU
  check then named the `FLASH_ATTN_EXT`.
- **HIP graphs overflow a JVM thread's stack.** ggml's HIP backend captures the graph as a HIP graph, and ROCm's
  `hip::Graph::ScheduleOneNode` recurses once per node. On a JVM thread (smaller stack than a CLI's 8 MB main
  thread) LTX segfaults in libamdhip64 during text encoding. `GGML_CUDA_DISABLE_GRAPHS=1` fixes it, and Fornax sets it by default (`-Dfornax.gpu.graphs=true` turns graphs back on).
- **Image options the engine ignores without a word.** `strength` only reaches the program when there is an
  `init_image`, and `control_strength` only with a `control_image`; otherwise both are accepted and dropped. (A
  `mask_image` without an init image does fail, but only once generation starts.) LoRAs flagged `is_high_noise`
  are skipped for images. `DiffusionModel.Request` has fields for these and refuses the combinations that would be
  ignored.
- **sd-cli and the C API take LoRAs differently.** sd-cli wants `--lora-model-dir` and `<lora:name:weight>` tags in
  the prompt, which it strips; the C API takes a `loras` array of paths and weights. The compiled program is the
  same (`scripts/compare-with-sdcli.sh` showed it for FLUX.2 klein and Qwen-Image-Edit 2511), so compare with the
  tag in sd-cli's prompt and without it in Fornax's.
- **A LoRA on fp8 weights aborted the process.** The default `lora_apply_mode=auto` merged the LoRA into fp8
  weights (FLUX.2 klein's `-fp8` files) "immediately", which needs a copy into F8_E4M3 that Vulkan can't run:
  `ggml-backend.cpp: pre-allocated tensor ... cannot run the operation (CPY)`, then SIGABRT. sd-cli does the same.
  `native/patches/stable-diffusion.cpp/0002` (upstream as leejet/stable-diffusion.cpp#2431, revised to the
  maintainer's review) makes `auto` apply LoRAs at run time for F8_E5M2 weights, and for F8_E4M3 weights wherever the
  backend can't copy F32 back to E4M3 (Vulkan, Metal); CPU and CUDA keep the immediate merge. Without the patch,
  `.set("lora_apply_mode", "at_runtime")` works around it.
- **Qwen-Image-Edit 2511 needs `zero_cond_t`.** The engine enables it when the weights contain an
  `__index_timestep_zero__` tensor (Comfy-Org's bf16 file does). Files without it need
  `model_args = "qwen_image_zero_cond_t=true"`, or edits degrade badly, again with no error.
- **Qwen-Image-Edit 2511 in bf16 doesn't fit Vulkan's GPU heap on the 8060S.** RADV exposes about 64 GB of the
  128 GB as device memory. The 41 GB bf16 transformer plus the text encoder tracked 50.7 GB, and the first sampling
  step then wanted 1.5 GB of compute with about 0.6 GB free (sd-cli fails the same way). `params_backend=te=cpu` did
  not change the total. `params_backend=*=cpu` (sd-cli's `--offload-to-cpu`) keeps a RAM copy of every weight and
  stages GPU copies on top; on unified memory that doubles the footprint, and the machine ran out of memory. Use an
  fp8 or GGUF build of the model, or raise the GTT limit.
- **sd-cli loads a mask as one grey channel** (stb_image's luma); `DiffusionModel` sends masks the same way, so a
  mask's threshold at 0.5 falls on the same pixels.
- **Fedora's ROCm lives in `/usr`**, not `/opt/rocm`; `native/build.sh hip` falls back to `/usr/lib64`.
- **Cancel only reaches a run that has started.** `sd_cancel_generation` is thread-safe, but an idle context ignores
  it, so a cancel that lands just before `sd_generate_image` begins is lost. Fornax asks again from the progress callback.
  A cancelled run returns `SD_STATUS_CANCELLED` (10) and no images.
- **The progress callback also reports loading.** sd.cpp loads weights on the first run after `sd_ctx_create`, and the
  process-wide progress callback counts the tensors it loads (hundreds of "steps") before the sampling steps. Loading
  doesn't check for a cancel; sampling does, at every step.
- **LTX audio is part of the model.** LTX-2.5 generates sound and picture together. Leaving out
  `audio_vae_path` removes the audio stream; that didn't cause the ghosting, but it isn't how the model was trained.
- **LTX audio to video is a Fornax patch.** At the pinned commit sd.cpp's `ref_audios` only drove Wan2.2 S2V and
  MiniMax-H3, and its LTX audio VAE had no encoder. `native/patches/stable-diffusion.cpp/0002` adds the encoder
  (host log-mel as ComfyUI's `MelSpectrogram`, torchaudio's sinc resampler to 16 kHz), puts the encoded track into the
  joint latent with denoise mask 0, and gives the audio stream its own timestep (0 when held). It also had to follow
  LTX-2/ComfyUI's cross-modal timesteps: the video's cross-attention scale/shift follows the video's timestep, and its
  prompt modulation and the video-to-audio gate the video's largest one. sd.cpp fed all of them the audio's timestep, which is the same
  thing until the two streams are at different noise levels. Checked against ComfyUI: the soundtrack of a Fornax render
  correlates 0.97 sample for sample with ComfyUI's own encode → decode of the same section.

## acestep.cpp

- **It can stop mid-run.** At the pinned commit, `ace_lm_generate`, `ace_synth_job_run_dit` and
  `ace_synth_job_run_vae` take a `bool (*cancel)(void *)` polled between LM tokens, DiT steps and VAE tiles. The shim
  passes one that reads a flag `fornax_acestep_set_cancel` sets. yue2.cpp's `pipeline_generate` takes the same kind of
  callback (AR tokens, NAR steps, VAE tiles), and its shim does the same.
- **Decode in 512-frame VAE tiles.** The default 1024-frame tiles hung the GPU on battery (the same compute
  timeout). The shim defaults to 512 and reads `FORNAX_ACESTEP_VAE_CHUNK`.

## whisper.cpp (fornax-whisper)

- **DTW needs flash attention off.** whisper.cpp quietly drops DTW token timestamps when flash attention is on (its
  default), so the shim turns flash attention off whenever an alignment preset is given. On 55 s of narration with
  large-v3-turbo on Vulkan, DTW made a transcription about 50% slower (1.8 s against 1.2 s, beam 5).
- **The new-segment callback is unreliable with DTW on.** Its DTW branch passes a segment index as `n_new`, and its
  loop bound skips segments after the first window. The shim counts the segments it has reported itself and hands
  over the rest when `whisper_full` returns.
- **Word times drift at pauses.** In running speech, DTW puts a word within a frame or two (20 ms), but a word after
  a pause started 0.06-0.27 s late, and the word before a pause ran on to the next word's start. `Words.snapToVoice`
  moves word edges onto the voice where there is silence to go by: sentence starts went from 0.16 s to 0.03 s off on
  average, ends from 0.76 s to 0.08 s (synthesized narration with known gaps). Whisper's timestamp tokens (no DTW)
  were worse: starts up to 0.5 s early or late, and ends that ran into the next sentence.
- **large-v3-turbo's segment times are coarse**, often whole seconds. Fornax fits each segment to its words.
- **`no_context` is per call, not per window.** It only clears the text a previous `whisper_full` call left in the
  state (and is on by default). Each 30 s window within a call is still prompted with the text before it, up to
  `n_max_text_ctx` tokens; OpenAI's `condition_on_previous_text=False` is `n_max_text_ctx = 0`. The shim always sets
  `no_context`, so transcriptions are independent, and `Options.conditionOnPreviousText(false)` sets
  `n_max_text_ctx` to 0.
- **VAD doesn't map DTW times.** With `vad` on, whisper.cpp transcribes the speech segments joined (each plus
  `samples_overlap` of the audio after it, with 0.1 s of silence between) and maps segment and token `t0`/`t1` back to
  the original audio, but not `t_dtw`. `Words.VadTimeline` rebuilds that layout from `whisper_full_get_vad_segment_t0`
  and `_t1` and maps DTW times the way `whisper_map_token_time_segment_aware` does. VAD runs on the CPU, and the
  state keeps the first VAD model it loads, whatever path a later call gives.
- **Repeats in silence.** On a sung vocal stem with a near-silent 30 s outro, large-v3-turbo repeated the last line 24
  times with the defaults. VAD, `n_max_text_ctx = 0`, `entropy_thold` 2.8 and `suppress_nst` each stopped it (the
  speech-to-text guide has the numbers). `suppress_nst` also took out most commas and full stops, though they aren't
  on its list.
- **SPIRV-Headers.** whisper.cpp v1.9.6's ggml finds SPIRV-Headers but doesn't link the target into ggml-vulkan, so the
  headers must be on the compiler's include path. Fedora's `spirv-headers-devel` puts them in `/usr/include`; for a
  private install, pass `CMAKE_EXTRA="-DCMAKE_CXX_FLAGS=-I<prefix>/include"`.

## stems.cpp (fornax-demucs)

- **Why stems.cpp.** It runs HTDemucs v4 on ggml (so on the GPU, through the same build as the other engines), its
  published GGUFs match PyTorch demucs sample for sample (71-126 dB SNR per stem on its parity clip), and it has
  progress and cancel callbacks in a stable C ABI. sevagh/demucs.cpp, the older port, is Eigen on the CPU only and
  says it trades speed for memory. Both are MIT.
- **It does demucs' whole separate path,** not just the network: normalise by the mono mix's mean and deviation,
  7.8 s segments with 25% overlap and demucs' triangular cross-fade, then scale back. So there is nothing for Fornax to
  add around it. torchaudio's `HDEMUCS_HIGH_MUSDB_PLUS` (Hybrid Demucs v3, which Endless Country episode 2 used, with
  10 s chunks and a 1 s linear cross-fade) is a different model: on that song, Fornax's HTDemucs vocals measured
  13.5 dB SDR against torchaudio's (15.8 dB on a 20 s excerpt), its backing 14.8 dB, and the stems add back up to
  the mix to -35 dB.
- **Its ggml is a fork** (betweentwomidnights/ggml, shared with sa3.cpp and audiocraft.cpp). The fork carries a fix
  for ggml-vulkan rounding F32 matmul operands to fp16 (betweentwomidnights/ggml#7), without which HTDemucs failed
  stems.cpp's parity check on Vulkan. It is linked statically and kept private like every engine's ggml.
- **It prints its backend** (`[stems] backend: Vulkan0 ...`) on stderr when a model loads, whatever `FORNAX_VERBOSE`
  says. ggml's own logs are quiet as usual.
- **Cancel lands between segments.** stems.cpp polls the cancel callback before each 7.8 s segment, under a second
  apart on the 8060S. The progress callback comes at the same points.

## ComfyUI parity for LTX-2.5 (Endless Country episode 2)

| ComfyUI | Fornax (`VideoClip -Drecipe=two-stage`) |
|---|---|
| Stage 1: half size, `ManualSigmas` 1.0 … 0.421875, 0, euler_ancestral, CFG 1 | `sample.sigmas`, `sample.method=euler_a` |
| `LTXVImgToVideoInplace` strength 0.7 | `strength=0.7` |
| `LTXVPreprocess` 18 | ffmpeg H.264 round trip at CRF 18 (no visible effect in tests) |
| `LTXVLatentUpsampler` with the x2 model | `hires.upscaler=model` plus `hires.model_path` |
| Stage 2: sigmas 0.85, 0.725, 0.4219, 0, first frame at 1.0 | `hires.sigmas`; sd.cpp holds the first frame fully in the refine |
| Tiled VAE decode | `vae_tiling.enabled` |
| `LTXVAudioVAEEncode` + `SetLatentNoiseMask` 0 in place of `LTXVEmptyLatentAudio` (`ltx_a2v.py`) | `VideoRequest.withAudio(song, start)` (`-Psong=… -Psong.start=…`) |

## llama.cpp text to speech (fornax-tts)

- **Kokoro streams one chunk.** It isn't autoregressive: in stream mode the whole text decodes in one step and arrives
  as a single chunk, so cancel and backpressure have nothing to act on. The autoregressive models (ZONOS3, LFM3-Audio,
  ...) stream frame by frame.

- **Call `postprocess_cpu_params` before `common_init_from_params`.** The parser fills in the thread counts after
  the arguments; without it the threadpool is sized from -1 and ggml tries to allocate 2^64 bytes. See
  [tts-notes.md](tts-notes.md) for the rest (models, voices, licences).

## llama.cpp language models (fornax-llama)

- **`chatJson`'s stream callback only fires on content.** Reasoning and tool-call tokens add no content, so a callback
  that only sees content can't stop them. The shim calls it with an empty piece for those tokens, which LlamaModel
  doesn't pass on but uses to check for a cancel.
- **Upcalls must not throw.** An exception that escapes an FFM upcall can't unwind through C++ frames: the JVM prints
  "Unrecoverable uncaught exception encountered" and exits. Every Fornax upcall catches it and stops the engine instead.

## llama.cpp vision (fornax-llama with an mmproj)

- **One path for every chat with images.** The shim converts image parts to media markers, renders the chat with the
  model's own Jinja template (`common_chat_session`, as llama-server does), splits the rendered prompt on the
  markers, and tokenizes text and images together with `mtmd_tokenize_from_parts`. Template text is parsed for
  special tokens and user text isn't, as in the text-only path. `mtmd_helper_eval_chunks` then runs the vision encoder
  and decodes everything; after that, generation is the usual loop. So `json_schema` grammars and tool calls work with
  images for free.
- **The marker is private.** The shim uses a random marker per process rather than `mtmd_default_marker()`
  (`<__media__>`), so user text that happens to contain `<__media__>` can't claim an image. Raw `complete` with images
  does use the default marker, since the caller writes the whole prompt. The shim counts those markers itself:
  `mtmd_tokenize` returns 2 for a marker/image mismatch (its header says 1), the same code as a failed image.
- **Text tokens after an image need only one position.** Qwen3-VL uses M-RoPE; `mtmd_helper_eval_chunks` returns the
  next position (fewer positions than tokens for an image), and text tokens after it take a single position each
  through `llama_batch_ext_set_pos`. So the shim counts context cells apart from positions: a ~1200-token image
  takes ~40 positions, and checking the position against `n_ctx` would overrun the KV cache.
- **Non-causal image attention needs the whole image in one ubatch.** Gemma 3 (MedGemma) attends over an image
  non-causally. Like `llama-mtmd-cli`, `fornax_llama_create_mm` caps the image's tokens at `n_ubatch` (512 by default)
  for such models, using `mtmd_get_memory_usage` (an unstable mtmd API that may move).
- **PaddleOCR-VL takes task prompts, not instructions.** It answers "OCR:", "Table Recognition:", "Chart
  Recognition:" and so on, and ignores a request to fill a schema: a grammar-constrained reply comes back as
  well-formed JSON with the wrong values. Read the text with PaddleOCR-VL, then extract with a text model
  (`./gradlew invoice -Pmodel=paddleocr-vl`). Its OCR of a 1000×1300 invoice took ~1.5 s at ~200 tokens/s on the
  8060S.
- **Qwen3-VL 4B grounds what is there, not what is missing.** Asked for defects on a tray of parts, it boxed a dented
  part accurately (with a schema that asks for a short description before each box) but never found an empty slot
  in the grid, with or without a schema. Its boxes are on a 0 to 1000 grid over the image.
- **Images are cheap on the GPU.** A 1000×1300 invoice is ~1200 Qwen3-VL tokens and is read (encoded and prefilled)
  in ~1.6 s; a 720×480 picture in ~0.4 s.

## llama.cpp embeddings and reranking (fornax-llama, Mode.EMBED and Mode.RERANK)

- **Set up as llama-embedding does.** A pooled handle has `embeddings` on, the GGUF's own pooling type (Qwen3-Embedding:
  last; EmbeddingGemma 2: mean; Qwen3-Reranker: rank), one unified KV cache, and `n_batch = n_ubatch = n_ctx`:
  pooling needs every token of a sequence in one ubatch, so the context size is also the longest input. Inputs go in
  as separate sequences, up to 256 (`llama_max_parallel_sequences()`) a batch, and the cache is cleared before each
  batch.
- **`n_seq_max` must not exceed `n_batch`.** llama.cpp sizes the output buffer for `max(n_outputs, n_seq_max)` and
  asserts that fits `n_outputs_max` (which defaults to `n_batch`); a 64-token context with 256 sequences aborted the
  JVM. The shim caps `n_seq_max` at the batch.
- **Batching moves the numbers a little.** A text embedded with others comes out at cosine ~0.9999 to the same text
  embedded alone, and a borderline rerank score moved from 0.598 alone to 0.626 in a batch of four. `llama-embedding`
  gives the same numbers both ways on the same build, so it is the GPU kernels' shapes, not leakage between sequences.
  Fornax's batched embeddings match `llama-embedding`'s batched ones to cosine 1.000000 (`EmbeddingsIT`;
  `scripts/embedding-reference.sh` regenerates the reference).
- **The rerank prompt is the GGUF's.** Qwen3-Reranker's GGUF carries a `tokenizer.chat_template.rerank` with
  `{query}` and `{document}`; the shim fills it in and tokenizes it whole with special tokens parsed, as llama-server's
  `format_prompt_rerank` does. The score is the first class output, which for Qwen3 is the softmax probability of "yes".
  Embedding inputs, unlike llama-embedding's, aren't parsed for special tokens (they're the caller's data).
- **Qwen3-Reranker 4B is a community GGUF.** Qwen publishes none, and many community conversions lack the classifier
  head, so they score everything near 0 (llama.cpp #16407). `Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp` has it
  (`qwen3.pooling_type = 4`, `cls.output.weight`, labels `yes`/`no`) and the 0.6B's `rerank` template, and its
  scores match `llama-embedding`'s on the same file (`Reranker4BIT`).
