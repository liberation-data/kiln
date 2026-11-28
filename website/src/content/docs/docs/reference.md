---
title: API overview
description: The public classes of each Fornax module, what they're for, and how to build the Javadoc.
---

Fornax's public API is small: one model class per engine, plus a few shared helpers. This page lists it by module.
For every method and parameter, build the Javadoc from a clone:

```sh
./gradlew javadoc        # one site per module, in <module>/build/docs/javadoc/index.html
```

All packages are under `org.liberationdata.fornax`. Every model class is `AutoCloseable` (its weights are freed on
`close()`, which is idempotent). A model can be called from any thread, one call at a time; the `...Async` methods run
on its own thread. See [Threading](/docs/guides/threading/).

## fornax-core

Package `org.liberationdata.fornax`. The engine modules depend on it.

| Type | |
|---|---|
| `HfCache` | finds files in the Hugging Face cache: `root()`, `home()`, `find(repo, path)`, `require(repo, path)`, `requireAny(repo, path, ...)`, and the cache's contents: `repos(root)`, `files(root, repo)` |
| `hub.FornaxHub` | pulls, finds, reports and evicts model sets: `standard()`, `builder()`, `pull(set, listener)`, `pullAsync`, `plan`, `status`, `find`, `require`, `evict`, `fetch(reference, listener)`, `isReference`. See [Downloading models](/docs/guides/models/) |
| `hub.ModelRef` | a model reference, `owner/repo[@revision][:tag]`: `parse`, `tryParse`, `pick`. See [By reference](/docs/guides/models/#by-reference) |
| `hub.ModelReferences` | what an engine's constructor does with a path or reference: `locate(model, engine)`, `path(model, engine)` for one file, `set(name, engine, runs)` for a catalogue set, `check(model, engine)` (its refusals, before any request), `isPath`, `file` |
| `hub.ModelSet` | a model as a set of files by role: `id`, `kind`, `title`, `engine`, `parts`, `license`, `gated`, `aliases`; `Part` (role and sources), `Source` (repo, file, revision) |
| `hub.ModelCatalog` | the models Fornax's examples, CLI and Studio use: `all()`, `of(kind)`, `byName`, `require` |
| `hub.ModelFolders` | folders searched after the cache (a ComfyUI `models/`): `configured()`, `find(part)` |
| `hub.PullProgress`, `PullTask`, `PullPlan`, `ModelStatus`, `Eviction`, `HubException`, `HubToken` | a pull's progress, a background pull, its sizes, what is on disk, what eviction did, failures by reason, and where the token comes from |
| `ModelPool` | keeps models loaded within a memory budget: `acquire(key, sizeBytes, loader)` returns a `Lease`; `unload`, `unloadAll`, `entries`, `loadedBytes`, `sizeOf(Path...)`. See [ModelPool and memory](/docs/guides/model-pool/) |
| `ModelPool.Lease<M>` | a borrowed model: `model()`; `close()` gives it back |
| `ModelPool.Entry` | a snapshot of one pooled model: `key`, `sizeBytes`, `loaded`, `inUse`, `lastUsed` |
| `Audio` | stereo float PCM in [-1, 1]: `left`, `right`, `sampleRate`, `seconds()`, `writeWav(File)` (16-bit), `readWav(File)`, `ofMono`, `mono()`, `resample(rate)` |
| `NativeLibrary` | finds and binds a native library, and sets the engines' tunables before the first one loads. You won't usually call it |
| `ModelThread` | what each model uses to run its calls one at a time, on the caller's thread or its own, and to cancel and close them. You won't usually call it |
| `Call` | one call into an engine: its stop flag and the first exception a callback threw. For engine bindings |
| `StreamOverflowException` | what a streaming subscriber gets when it falls a whole buffer behind, and the generation stops |
| `Vectors` | search over embeddings in memory: `topK(query, vectors, k)` (by cosine), `cosine`, `dot`, `normalize`, `rank(scores)`. See [Embeddings and reranking](/docs/guides/embeddings/) |
| `Scored` | an item's `index` and `score`; `BEST_FIRST` |
| `http.EventStream`, `http.Loopback` | the server-sent-events writer (with its keep-alives) and the Host check that fornax-server and fornax-studio-server share. For those servers, not for applications: they may change in any release |

## fornax-llama

Package `org.liberationdata.fornax.llama`. See [Language models](/docs/guides/language-models/).

| Type | |
|---|---|
| `LlamaModel` | a GGUF language model: `new LlamaModel(gguf)` or `(gguf, gpuLayers, contextSize)`; `(gguf, gpuLayers, contextSize, slots)` runs that many requests at once (see [slots](/docs/guides/language-models/#many-requests-at-once-slots)) |
| `chat(messages, options[, onPiece])` | a chat reply, optionally streamed; returns a `Result` |
| `chatJson(requestJson, onContent)` | an OpenAI-format request and reply, with tool calls and `json_schema` |
| `complete(prompt, options, onPiece)` | raw completion of a prompt you've formatted |
| `cancel()` | stops the call in progress (every one, with slots), from any thread |
| `chatAsync(messages, options[, subscriber])`, `chatStream(messages, options)` | `chat` on the model's own thread: a `CompletableFuture<Result>`, optionally streaming to a `Flow.Subscriber`; or a cold `Flow.Publisher<String>` |
| `chatJsonAsync(requestJson[, images[, audio][, subscriber]])`, `extractAsync(...)`, `completeAsync(prompt, images, options)` | the same for `chatJson`, `extract` and `complete` |
| `streamOn(executor, bufferSize)` | where streamed pieces are delivered and how far a subscriber may fall behind (default: virtual threads, 1024) |
| `new LlamaModel(Weights)`, `(gguf, mmproj)` | a vision language model with its mmproj; see [Vision](/docs/guides/vision/). `supportsVision()` |
| `extract(messages, jsonSchema, options)` | a reply constrained to a JSON schema, as JSON text; messages may carry images |
| `chatJson(requestJson, images, onContent)`, `complete(prompt, images, options, onPiece)` | `chatJson` and `complete` with images; `mediaMarker()` |
| `chatJson(requestJson, images, audio, onContent)` | `chatJson` with audio too (`{"type": "audio", "index": i}` or `input_audio` parts), for a model whose mmproj has an audio encoder; `supportsAudio()`, `audioSampleRate()`. See [Speech to text](/docs/guides/speech-to-text/#qwen3-asr-through-llamacpp) |
| `new LlamaModel(Weights, Mode)`, `(gguf, Mode)`, `(Weights or gguf, gpuLayers, contextSize, Mode)` | a model loaded for `Mode.CHAT`, `EMBED` or `RERANK`; `mode()`. See [Embeddings and reranking](/docs/guides/embeddings/) |
| `embed(texts[, embedOptions])`, `embed(text)`, `embedAsync(texts, embedOptions)` | embeddings (`float[][]`, L2-normalised by default); `embeddingSize()`. `embeddings(texts, embedOptions)`: the vectors with the tokens they took (`Embeddings`) |
| `rerank(query, documents)`, `rerankAsync(query, documents)` | the documents' relevance scores, best first, as `List<Scored>` |
| `LlamaModel.EmbedOptions` | `prefix`, `normalize`; `DEFAULT`, `query(task)` (Qwen3-Embedding's instruction format), `gemmaQuery(task)` and `gemmaDocument()` (EmbeddingGemma's prefixes), `withPrefix`, `withNormalize` |
| `LlamaModel.Message` | `role`, `content`, `images` and `audio`; `system(...)`, `user(...)`, `user(text, images...)`, `user(text, audio...)`, `assistant(...)` |
| `LlamaModel.Image` | packed RGB: `of(BufferedImage)`, `read(Path)`, `decode(bytes)` |
| `LlamaModel.Weights` | the models Fornax knows by name: `QWEN3_4B`, `QWEN3_VL_4B`, `PADDLEOCR_VL`, `MEDGEMMA_4B`, `OCRIA_1`, `QWEN3_EMBEDDING_0_6B`, `EMBEDDINGGEMMA_2`, `QWEN3_RERANKER_0_6B`, `QWEN3_RERANKER_4B`, `QWEN3_ASR_0_6B`, `QWEN3_ASR_1_7B`; `named(...)`, `vision()`, `audio()`, `model()`, `mmproj()`, `downloaded()`, `downloadCommand()` |
| `Json` | just enough JSON to read `chatJson` replies: `parse`, `get(doc, path...)`, `quote` |
| `LlamaModel.Options` | `maxTokens`, `temperature`, `topP`, `minP`, `seed`, `thinking`; `DEFAULT`, `withSeed`, `withMaxTokens`, `withTemperature`, `withThinking` |
| `LlamaModel.Result` | `text`, `promptTokens`, `generatedTokens` |

## fornax-tts

Package `org.liberationdata.fornax.tts`. See [Speech](/docs/guides/speech/).

| Type | |
|---|---|
| `TtsModel` | a TTS model: `new TtsModel(Weights)`, `(alias)`, `(model, mmproj)` or `(model, mmproj, gpuLayers, contextSize)`; paths or model references |
| `speak(text, voice)`, `(text, voice, onChunk)`, `(text, voice, options, onChunk)` | speech as `Audio`, optionally streamed in chunks; a voice of `@clip.wav` speaks like that clip |
| `voices()`, `sampleRate()`, `cancel()` | the built-in voices (the default first), the output rate, and stopping a streaming call |
| `speakAsync(text, voice, options[, subscriber])`, `speakStream(text, voice, options)` | `speak` on the model's own thread, optionally streaming chunks to a `Flow.Subscriber`; `streamOn(executor, bufferSize)` |
| `TtsModel.Weights` | the models Fornax knows by name: `KOKORO`, `ZONOS3`, `LFM3_AUDIO`, `TINY_VOCA`, `CHATTERBOX_TURBO`, `SOPRANO`, `VOXCPM2`, `QWEN3_TTS`; `named(...)`, `vision()`, `model()`, `mmproj()`, `downloaded()`, `downloadCommand()` |
| `TtsModel.Options` | `lang`, `temperature`, `seed`, `speed`, `maxFrames`; `DEFAULT` and a `with...` for each |

## fornax-sd

Package `org.liberationdata.fornax.sd`. See [Images](/docs/guides/images/) and [Video](/docs/guides/video/).

| Type | |
|---|---|
| `DiffusionModel` | a stable-diffusion.cpp model: `new DiffusionModel(Config)` |
| `generate(Request)` | images, as `List<BufferedImage>` |
| `generateVideo(VideoRequest)` | a `Video` |
| `generateAsync(Request)`, `generateVideoAsync(VideoRequest)`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops a run at the next step (it throws `CancellationException`) |
| `modelVersion()` | the model stable-diffusion.cpp detected |
| `onProgress(sink)`, `onLog(sink)` | process-wide progress (step, steps, seconds per step) and log callbacks |
| `DiffusionModel.Config` | `builder()`: `model(name)` for a catalogue set, `diffusionModel`, `vae`, `llm`, `controlNet`, and `set(key, value)` for any context option; a `_path` option may be a model reference. `LTX_BF16_TO_F32` (deprecated: not needed from 0.3.0) |
| `DiffusionModel.Request` | `prompt`, `width`, `height`, `steps`, `cfg`, `seed`, plus `refImages` and an `extra` options map; `withRefImages`, `withInitImage(image[, strength])` (img2img), `withMaskImage` (inpainting), `withControlImage(image[, strength])` (ControlNet), `withLoras`, `withExtra`. Settings the engine would ignore (a strength or mask without an init image, a control strength without a control image, a typed key in `extra`) throw `IllegalArgumentException` from `generate` |
| `DiffusionModel.Lora` | `file`, `weight` (default 1): a LoRA for one request |
| `DiffusionModel.VideoRequest` | the same, plus `frames` and `fps`; `withInitImage`, `withEndImage`, `withRefAudio`, `withExtra` |
| `DiffusionModel.Video` | `frames`, `fps`, `audio` (or `null`), `seconds()` |
| `Upscaler` | an ESRGAN-family upscaler: `new Upscaler(path[, tileSize])`, `scale()`, `upscale(image[, factor])` (a power of `scale()`), `upscaleAsync(...)` on its own thread; it can't be cancelled mid-pass |
| `Sd` | the raw bindings to stable-diffusion.cpp's C API; `Sd.SdException` is what a failed call throws |

Option keys are stable-diffusion.cpp's, from its
[C API guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/c_api.md).

## fornax-acestep

Package `org.liberationdata.fornax.acestep`. See [Songs](/docs/guides/songs/).

| Type | |
|---|---|
| `AceStepModel` | ACE-Step 1.5: `new AceStepModel(name)` or `(name, keepLoaded)` for a catalogue set, or `(lm, textEncoder, dit, vae, keepLoaded)` as paths or model references; `lm` may be `null` |
| `generate(requestJson)` | a song from acestep.cpp's JSON request; returns a `Result` |
| `generateAsync(requestJson)`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops it between LM tokens, DiT steps and VAE tiles (it throws `CancellationException`) |
| `AceStepModel.Result` | `audio` (48 kHz stereo) and `resolvedRequestJson` |

## fornax-yue2

Package `org.liberationdata.fornax.yue2`. See [Songs](/docs/guides/songs/#yue2).

| Type | |
|---|---|
| `Yue2Model` | YuE2: `new Yue2Model(name)` or `(name, keepLoaded, maxSeq)` for a catalogue set, or `(model, vae, keepLoaded, maxSeq)` as paths or model references; `maxSeq` 0 is the full context |
| `generate(requestJson)` | a song from yue2.cpp's JSON request; returns a `Result` |
| `generateAsync(requestJson)`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops it between LM tokens, diffusion steps and VAE tiles (it throws `CancellationException`) |
| `Yue2Model.Result` | `audio` (48 kHz stereo), `resolvedRequestJson` and `abcScore` |

## fornax-whisper

Package `org.liberationdata.fornax.whisper`. See [Speech to text](/docs/guides/speech-to-text/).

| Type | |
|---|---|
| `WhisperModel` | a Whisper model: `new WhisperModel(Weights)`, `(path)`, `(path, alignment)` or `(path, alignment, gpu)`; a path or model reference |
| `transcribe(audio[, options][, listener])` | a `Transcript` from `Audio` at any rate, or from 16 kHz mono `float[]` |
| `multilingual()`, `type()`, `cancel()` | English-only or not, the model size, and stopping a call from another thread |
| `WhisperModel.Weights` | the models Fornax knows by name: `LARGE_V3_TURBO`, `LARGE_V3_TURBO_Q8_0`, `BASE_EN`; `named(...)`, `path()`, `downloaded()`, `alignment` |
| `WhisperModel.Alignment` | the DTW alignment heads for a model size (`LARGE_V3_TURBO`, `BASE_EN`, ...), or `NONE` |
| `WhisperModel.Options` | `language`, `translate`, `initialPrompt`, `threads`, `beamSize`, `temperature`, `conditionOnPreviousText`, `suppressNonSpeech`, `temperatureIncrement`, `entropyThreshold`, `logProbThreshold`, `noSpeechThreshold`, `vad`; `DEFAULT` and a `with...` for each |
| `WhisperModel.Vad` | voice activity detection: `Vad.of(path)` or `Vad.of(VadWeights)`, then `withThreshold`, `withMinSpeechMs`, `withMinSilenceMs`, `withMaxSpeechSeconds`, `withSpeechPadMs`, `withOverlapSeconds` |
| `WhisperModel.VadWeights` | Silero VAD models by name: `SILERO_V6_2_0`, `SILERO_V5_1_2`; `named(...)`, `path()`, `downloaded()` |
| `WhisperModel.Listener` | `progress(percent)` and `segment(segment)` as the transcription runs |
| `Transcript` | `text`, `language`, `segments`; `words()`, `wordsTsv()`, `srt()`, `vtt()` |
| `Transcript.Segment`, `Transcript.Word` | `start`, `end` (seconds), `text`; a segment's `words`, a word's `probability` |

## fornax-demucs

Package `org.liberationdata.fornax.demucs`. See [Stem separation](/docs/guides/stem-separation/).

| Type | |
|---|---|
| `DemucsModel` | an HTDemucs model: `new DemucsModel(Weights)`, `(path)` or `(path, gpu, cpuThreads)`; a path or model reference |
| `separate(audio[, options][, progress])` | `Stems` from `Audio` at any rate, at its rate and length |
| `separateAsync(audio[, options, progress])`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops it between segments (it throws `CancellationException`) |
| `name()`, `backend()`, `sources()`, `sampleRate()` | the checkpoint, the ggml backend, the stem names in order, and the model's own rate (44100) |
| `DemucsModel.Weights` | the models Fornax knows by name: `HTDEMUCS`, `HTDEMUCS_F16`, `HTDEMUCS_6S`, `HTDEMUCS_FT`; `named(...)`, `path()`, `downloaded()` |
| `DemucsModel.Options` | `shifts`, `overlap`, `seed`; `DEFAULT` and a `with...` for each |
| `DemucsModel.Progress` | `update(done, total)`, segment passes, as the separation runs |
| `Stems` | `names()`, `get(name)`, `vocals()`, `backing()` (every stem but the vocals), `mix()`, `asMap()`, `sampleRate()` |

## fornax-examples

Package `org.liberationdata.fornax.examples`: runnable programs, each a Gradle task, and the best place to see the API
used end to end.

| Task | Class | |
|---|---|---|
| `single` | `Single` | an LLM writes a song, ACE-Step sings it, FLUX.2 [klein] paints the cover |
| `chat` | `Chat` | streams a chat reply |
| `speak`, `speakBatch` | `Speak`, `SpeakBatch` | text to speech, one line or many at one seed |
| `image` | `Image` | text to image or image editing: `klein`, `qwen-image`, `qwen-edit` |
| `video` | `VideoClip` | an LTX-2.5 clip with sound, one- or two-stage |
| `song` | `Song` | an ACE-Step song |
| `yue2` | `Yue2Song` | a YuE2 song and its score |
| `transcribe` | `Transcribe` | speech to text: prints the text, writes `<name>.words.tsv` and `<name>.srt` |
| `separate` | `Separate` | stem separation: writes `<name>-vocals.wav` and `<name>-backing.wav` (`--all` for every stem) |

## System properties and environment

| Name | |
|---|---|
| `-Dfornax.library.path`, `FORNAX_LIBRARY_PATH` | where to look for the native libraries |
| `-Dfornax.<name>.lib` | one library's exact file; the names are `fornax_llama`, `fornax_tts`, `stable-diffusion`, `fornax_acestep`, `fornax_yue2`, `fornax_whisper` and `fornax_demucs` |
| `-Dfornax.backend`, `FORNAX_BACKEND` | which natives jar's backend to load when several for this platform are on the classpath |
| `-Dfornax.cache.dir`, `FORNAX_CACHE_DIR` | where natives jars' libraries are extracted to |
| `-Dfornax.gpu.graphs` | ggml's CUDA and HIP graphs; `false` by default |
| `-Dfornax.vulkan.maxNodesPerSubmit` | graph nodes per Vulkan submission |
| `FORNAX_VERBOSE` | the engines' own logs; `debug` for everything from stable-diffusion.cpp |
| `FORNAX_ACESTEP_VAE_CHUNK` | ACE-Step's VAE decode tile, in frames (512) |
| `SD_FA_CHUNK_FLOPS` | stable-diffusion.cpp's largest flash-attention node, in FLOPs, before it splits by query rows (1e12 on Vulkan, off elsewhere unless set; `0`, negative or unparsable never splits) |
| `SD_FF_CHUNK_FLOPS` | stable-diffusion.cpp's largest linear-layer matmul, in FLOPs, before it splits by token rows (4e11 on Vulkan, off elsewhere unless set; `0`, negative or unparsable never splits) |
| `HF_HUB_CACHE`, `HF_HOME` | where `HfCache` and `FornaxHub` look |
| `-Dfornax.modelDirs`, `FORNAX_MODEL_DIRS` | model folders `FornaxHub` searches after the cache |
| `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`), `HF_TOKEN_PATH`, `HF_ENDPOINT`, `HF_HUB_OFFLINE` | `FornaxHub`'s token for gated repos, the Hub to download from, and no network |
| `FORNAX_JAVA`, `FORNAX_CLI_LIB` | the `fornax` launcher's Java and jars folder (see [The fornax command](/docs/guides/cli/#the-launcher)) |
