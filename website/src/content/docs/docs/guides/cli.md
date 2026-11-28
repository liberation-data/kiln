---
title: The fornax command
description: Fetch models with fornax pull, see what is on disk, chat with a language model, make images, songs and speech, transcribe audio, ask about images and serve models over an OpenAI-compatible API from a terminal, and how fornax is installed and launched.
---

`fornax` is Fornax's command-line tool. It fetches models into the Hugging Face cache and runs them without any code:

```sh
fornax pull qwen3-4b                                 # an alias from Fornax's catalogue: every file the model needs
fornax chat qwen3-4b "Write a haiku about a fornax"
fornax image flux2-klein "a brick kiln at dusk"      # like chat, downloads the model first if it isn't in the cache
```

## Installing it

- **Nothing to install, with [JBang](https://www.jbang.dev)** (from Fornax 0.3.0). JBang runs the `fornax` command from
  Maven Central, with the natives for your platform, on a Java 22 or newer that it fetches if you have none:

  ```sh
  jbang fornax@fornax4j chat qwen3-4b "Write a haiku about a kiln"
  jbang image@fornax4j flux2-klein "a brick kiln at dusk"      # also song@, speak@, transcribe@, see@ and server@
  ```

  `jbang app install fornax@fornax4j` puts it on your `PATH` as `fornax`. The aliases are in
  [jbang-catalog.json](https://github.com/liberation-data/fornax/blob/main/jbang-catalog.json), and run
  `org.fornax4j:fornax-cli`, which JBang can also run by those coordinates: `jbang org.fornax4j:fornax-cli:0.3.0 chat qwen3-4b`
  (add `--java 22+ -R--enable-native-access=ALL-UNNAMED`, as the aliases do).
- **With install.sh** (from Fornax 0.3.0), on Linux x86_64 or an Apple-silicon Mac, for a machine with no JBang, such
  as a server. It downloads the `fornax-cli` zip for your platform from the
  [GitHub release](https://github.com/liberation-data/fornax/releases), checks it against the release's `SHA256SUMS`,
  unpacks it into `~/.local/share/fornax` and links `~/.local/bin/fornax` to it. It needs no root and edits no shell
  startup file: if `~/.local/bin` isn't on your `PATH`, it prints the line to add. Read it first, then run it:

  ```sh
  curl -fsSLO https://fornax4j.org/install.sh
  less install.sh
  sh install.sh
  ```

  Or in one line, if you trust it unread:

  ```sh
  curl -fsSL https://fornax4j.org/install.sh | sh
  ```

  `fornax` runs on Java 22 or newer (`FORNAX_JAVA`, `JAVA_HOME`, or `java` on `PATH`). The script installs no Java:
  it checks for one, and if there is none it says how to get one (`sdk install java` with
  [SDKMAN](https://sdkman.io), `jbang jdk install 25`, or your package manager). On Linux the engines need the Vulkan
  loader and your GPU's Vulkan driver ([Installing GPU drivers](/docs/gpu-drivers/)). Run it again to update;
  `FORNAX_VERSION=0.3.0` installs that release, and `FORNAX_HOME` and `FORNAX_BIN_DIR` move the two folders
  (`XDG_DATA_HOME`, if set, moves the first to `$XDG_DATA_HOME/fornax`). To uninstall, delete those two:
  `~/.local/share/fornax` and `~/.local/bin/fornax`.
- **With Fornax Studio** (coming in the next Fornax Studio release): choose **Install fornax command** from Studio's menu.
  Studio writes a `fornax` launcher into a folder on your `PATH` (`~/.local/bin` on Linux; `/usr/local/bin`, or
  `~/.local/bin`, on macOS). That `fornax` runs on Studio's own Java and engines, so it needs nothing else installed.
- **From source**, for a machine without a desktop, such as a server. It needs JDK 25 to build and Java 22 or newer to
  run (`JAVA_HOME`, or `java` on `PATH`):

  ```sh
  git clone https://github.com/liberation-data/fornax.git && cd fornax
  native/build.sh vulkan               # the engines, or skip this and add -Pfornax.cli.nativesVersion=0.2.0 below
  ./gradlew :fornax-cli:installDist      # -> fornax-cli/build/install/fornax-cli/bin/fornax and lib/
  export PATH="$PWD/fornax-cli/build/install/fornax-cli/bin:$PATH"
  ```

  The distribution's `lib/` holds the Fornax jars and the natives jar for each platform built in `build/native/`.
  `-Pfornax.cli.nativesVersion=0.2.0` takes this platform's natives jar from Maven Central instead. Copy the
  `fornax-cli` folder anywhere; `bin/fornax` can also be reached through a symlink.

`fornax info` shows which Java, engines and cache it uses.

## Fetching models

```sh
fornax pull flux2-klein                                   # three files, from three repos
fornax pull unsloth/Qwen3-4B-Instruct-2507-GGUF           # lists the repo's GGUF models and asks which
fornax pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M    # a quant by its tag
fornax pull Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf README.md
fornax pull some/repo '*.json' --revision v1.0            # globs match paths in the repo
fornax pull some/repo@v1.0:Q8_0                           # a revision in the reference itself
```

`org/name[@REV][:TAG]` is the same reference `new LlamaModel("org/name:TAG")` and `FornaxHub.fetch` take, so a model
your code names can be fetched ahead of time with `fornax pull` and the same string. One difference: with no `:TAG`,
`fornax pull` lists the repo's GGUF models to choose from, where the code takes the repo's only one.

An alias is a model set from [the catalogue](/docs/guides/models/#the-catalogue); `fornax` with no arguments lists
them. `fornax pull` downloads with [`FornaxHub`](/docs/guides/models/): into the Hugging Face cache, in the layout
`hf download` uses, so `fornax`, `hf`, Fornax Studio and your own app share one copy of each file. Files already in the
cache, or in a [model folder](/docs/guides/models/#model-folders), are used as they are. An interrupted download
resumes; each file is checked against the Hub's hash. The progress line shows each file's bytes, the download rate
and the time left.

A gated repo (`ltx-2.5-distilled`, for one) needs its terms accepted on its huggingface.co page and a token in
`HF_TOKEN` (or from `hf auth login`). Without them, the pull stops with a message that names the repo and what to do.
With `HF_HUB_OFFLINE=1`, `fornax pull` touches no network: an alias stops at the first file not already on disk, and
a repo (`org/name`) stops at once, as listing its files needs the Hub.

```sh
fornax list            # each alias: downloaded, partly downloaded or not; sizes; other GGUF models in the cache
fornax rm flux2-klein  # delete a model's files from the cache, keeping any another downloaded model uses
```

`hf download` works as well: `fornax` sees what it downloads, and the other way round.

## Trying models

```sh
fornax chat qwen3-0.6b "What is 2+2?"     # one reply, streamed
fornax chat qwen3-0.6b                    # interactive: /reset clears the history, /exit quits
echo "Summarise this" | fornax chat qwen3-4b
fornax test qwen3-4b                      # load time, time to first token, tokens a second
fornax test flux2-klein                   # one 256x256 image in 4 steps
fornax test ace-step --seconds 5          # a short song
```

`<model>` is an alias, `org/name[@REV]:TAG`, `org/name:file.gguf`, `org/name` (when the repo has one GGUF model) or a
path to a `.gguf` file. `chat` and `test` download a model that isn't in the cache yet before loading it, with the same
progress line as `fornax pull`, and refuse one of another kind before downloading anything.

A reasoning model, such as Qwen3, thinks aloud in `<think>…</think>` before it answers. `fornax chat` hides that, and
shows only the answer; `--think` shows the reasoning too, and `--no-think` asks the model not to reason at all,
where its chat template can turn that off (Qwen3.5, Qwen3's hybrids). Some models (Qwen3 Thinking, DeepSeek-R1 distills) are
trained with the `<think>` in the prompt and only close the block. `fornax chat` holds the first couple of hundred
characters of a reply until it can tell, or longer while the reply opens the way reasoning does ("Okay, the user…").
If the reasoning runs past that, you see it, but the `</think>` is still hidden. Once a reply opens inside, an
interactive chat takes the model's later replies as starting inside too, until `/reset`. Other options: `--system`, `--max-tokens`,
`--temperature`, `--seed`, `--ctx` and `--gpu-layers` (`fornax chat --help`).

## Making things

:::note[Coming in 0.3.0]
These commands are on `main` and ship with the next release, 0.3.0. A `fornax` built from source has them now.
:::

```sh
fornax image flux2-klein "a brick kiln at dusk, a fox by the door"
fornax image flux2-klein "a fox" --width 768 --height 512 --steps 12 --seed 42 --out fox.png
fornax image qwen-image-edit "make it night" --ref kiln.png
fornax song ace-step-turbo "acoustic folk, fingerpicked guitar, warm male vocal" --lyrics words.txt --seconds 60
fornax song ace-step "upbeat synth pop about a robot"     # with its lyric LM, it writes the words too
fornax speak "Hello from Fornax"                          # kokoro, voice af_heart -> speech.wav
fornax speak "Good evening" --voice bm_george --out evening.wav
fornax speak --list-voices
fornax transcribe speech.wav                              # whisper-base.en: the text
fornax transcribe interview.wav --model whisper-large-v3-turbo --words   # start, end and word, a line each
fornax see qwen3-vl-4b kiln.png                           # "Describe this image."
fornax see qwen3-vl-4b receipt.jpg "What is the total?"
```

Each is a thin wrapper over its engine (`DiffusionModel`, `AceStepModel` or `Yue2Model`, `TtsModel`,
`WhisperModel` and a vision `LlamaModel`), and `--help` lists its options. Like `chat`, these download a model
that isn't in the cache yet before loading it, with the same progress line as `fornax pull`. A model of the wrong
kind is refused before anything is downloaded, with the ones that fit: `fornax image qwen3-4b` says qwen3-4b is a
chat model and names the image models.

- **image** takes an image alias (`flux2-klein`, `qwen-image`, `qwen-image-edit`, `z-image-turbo`, `ernie-image-turbo`) and draws
  1024x1024 in 20 steps (8 for Z-Image Turbo and ERNIE-Image-Turbo), at the guidance the model is tuned for (4.0 for FLUX.2
  [klein], 2.5 for Qwen-Image, 1.0 for the Turbos). Width and height are
  multiples of 16. `qwen-image-edit` edits the image given with `--ref`, at that image's size unless told otherwise.
  The seed is random unless given, and printed, so a good image can be made again.
- **song** takes a song alias and a description of the style: genre, instruments, voice, mood. `--lyrics` reads the
  words from a file (`-` for stdin), with section tags such as `[Verse]` and `[Chorus]`. Without it, `ace-step`
  and `ace-step-xl` (which have the 4B lyric LM) and `yue2` write their own words, and `ace-step-turbo` and
  `ace-step-xl-turbo` make an instrumental. 30 seconds unless `--seconds` says otherwise. YuE2's weights are
  CC BY-NC 4.0: no commercial use of them or of what they make.
- **speak** uses `kokoro` and its first voice, `af_heart`, unless `--model` and `--voice` say otherwise; without text
  it reads stdin. `zonos3-mini` has British and Irish English voices among its 27 (`--list-voices`); `voxcpm2` has
  none, and takes a description of one in parentheses before the text: `fornax speak --model voxcpm2 "(A warm
  Australian woman)G'day."`. A TTS model that isn't in the catalogue loads by reference or path with its `--mmproj`:
  `fornax speak --model ggml-org/tiny-voca-GGUF:tiny-voca-F32.gguf --mmproj ggml-org/tiny-voca-GGUF:mmproj-tiny-voca-F32.gguf "Hi"`.
- **transcribe** reads a WAV at any rate (convert other formats first: `ffmpeg -i in.mp3 in.wav`) and prints the
  text, or with `--words` each word's start and end in seconds and the word, tab-separated. `whisper-base.en` hears
  English only; `whisper-large-v3-turbo` hears 99 languages (`--language` names one, else it is detected).
  `--model` also takes `org/name:file.bin` or a path to a whisper.cpp model.
- **see** takes a vision alias (`qwen3-vl-4b`, `paddleocr-vl`, `medgemma-4b`, `ocria-1`), or a GGUF by reference or path with
  its `--mmproj`, an image (PNG, JPEG, BMP or GIF) and a question, and streams the answer as `chat` does, with
  `chat`'s options. Without a question it asks "Describe this image.", or for `paddleocr-vl` "OCR:", which reads
  the image's text out. For `ocria-1` it asks "Transcribe this page in markdown.", and turns its thinking off, which
  OCRia-1 needs.

`image`, `song` and `speak` write to `--out`: a file, which is written over, or a folder (an existing one, or a
name ending in `/`). Otherwise they write `image.png`, `song.wav` or `speech.wav` in the current folder, numbered
(`image-2.png`) rather than overwrite one an earlier run made. A wrong extension, or an input file that isn't
there, fails before any model loads. Timings and other notes go to stderr; stdout has only the file written, the
transcript or the answer, for scripts.

Set `FORNAX_VERBOSE=1` to see the engines' own logs.

## Serving models

:::note[Coming in 0.3.0]
These commands are on `main` and ship with the next release, 0.3.0. A `fornax` built from source has them now.
:::

```sh
fornax server -m qwen3-4b                                 # http://127.0.0.1:54154/v1
fornax server -m qwen3-4b -m qwen3-embedding-0.6b --port 8080 --budget-gb 32
fornax server --host 0.0.0.0 --api-key "$KEY"             # reachable from other machines, with a key
```

`fornax server` serves models over an OpenAI-compatible HTTP API: chat completions (streaming, tool calls, JSON
schemas, images for a vision model), models, embeddings, images, speech and transcription. Point an OpenAI client's
base URL at it. The `-m` models are downloaded if need be and loaded first; a request may also name a catalogue
model already downloaded, which is loaded on first use and kept loaded within `--budget-gb`. Any other is refused
(404) unless `--allow-downloads` lets requests name any model, downloaded on first use. It listens on 127.0.0.1 only, and refuses web
pages unless `--cors` allows their origin. See [OpenAI-compatible server](/docs/guides/openai-server/).

## The launcher

`bin/fornax` (and `bin/fornax.cmd` on Windows) runs `org.liberationdata.fornax.cli.Main` from three inputs. Each comes from
an environment variable, else from a value written into the launcher, else from a default:

| Input | Environment variable | Placeholder | Default |
|---|---|---|---|
| the `java` to run, Java 22 or newer | `FORNAX_JAVA` | `@FORNAX_JAVA@` | `$JAVA_HOME/bin/java`, else `java` on `PATH` |
| the folder of jars: `fornax-cli`, the `fornax-*` modules it uses and, for the engines, a `fornax-natives-<os>-<arch>-<backend>` jar | `FORNAX_CLI_LIB` | `@FORNAX_CLI_LIB@` | `../lib` next to the launcher, after following symlinks |
| a folder of native engine libraries, loaded before any natives jar | `FORNAX_LIBRARY_PATH` | `@FORNAX_LIBRARY_PATH@` | none |

`FORNAX_OPTS` and `JAVA_OPTS` add JVM options; the launcher always passes `--enable-native-access=ALL-UNNAMED`.

An installer such as Fornax Studio renders the launcher by replacing each placeholder with an absolute path (or
leaving it, for the default) and writes the result, executable, into a folder on `PATH`. Studio's `fornax` points at
Studio's bundled Java, its jars and its engines, and uses the same Hugging Face cache as everything else.
