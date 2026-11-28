---
title: Speech to text
description: Transcribe audio from Java with Whisper through whisper.cpp, with a start, end and probability for every word, and write word timings, SRT or WebVTT subtitles; or with Qwen3-ASR through llama.cpp.
---

`fornax-whisper` turns speech into text through [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Every word comes
with its own start and end time and a probability, which is what captions, lyric titles and editing to the word need.
The class is `WhisperModel`.

## Transcribe a file

```sh
hf download ggerganov/whisper.cpp ggml-large-v3-turbo-q5_0.bin
```

<!-- snippet: SpeechToText#transcribe -->
```java
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.whisper.Transcript;
import org.liberationdata.fornax.whisper.WhisperModel;

import java.io.File;

try (var stt = new WhisperModel(WhisperModel.Weights.LARGE_V3_TURBO)) {
    Transcript t = stt.transcribe(Audio.readWav(new File("talk.wav")));
    System.out.println(t.text());
    for (Transcript.Word w : t.words())
        System.out.printf("%6.2f %6.2f %4.2f %s%n", w.start(), w.end(), w.probability(), w.text());
}
```

Any `Audio` works: Fornax mixes it to mono and resamples it to Whisper's 16 kHz. `Audio.readWav` reads 8, 16, 24 and
32-bit PCM and float WAV files. For MP3 and other formats, convert first (`ffmpeg -i in.mp3 in.wav`). If you already
have 16 kHz mono samples, pass the `float[]` straight to `transcribe(pcm, options, listener)`.

A `Transcript` has:

- `text`: everything Whisper heard, with its punctuation;
- `language`: the language it transcribed, detected unless you gave one;
- `segments`: phrases or sentences, each with `start`, `end`, `text` and its `words`;
- `words()`: every `Word` in order, with `start`, `end`, `text` and `probability`. Punctuation stays attached to the
  word before it ("Hello,"). A low probability flags a word the model was unsure of.

Times are in seconds from the start of the audio. Word times never go backwards: each word starts at or after the end
of the one before.

## Subtitles and word timings

<!-- snippet: SpeechToText#subtitles -->
```java
Files.writeString(Path.of("talk.srt"), t.srt());          // one cue per segment
Files.writeString(Path.of("talk.vtt"), t.vtt());          // WebVTT
Files.writeString(Path.of("talk.words.tsv"), t.wordsTsv()); // start <TAB> end <TAB> word, one word per line
```

## How words are timed

Whisper itself predicts timestamps only every few seconds. Fornax times each word in two steps:

1. **DTW alignment.** A few of the decoder's attention heads line text up with audio. whisper.cpp runs dynamic time
   warping over them, which puts each token on a 20 ms audio frame. A word starts at its first token and ends where the
   next word starts. Every `Weights` entry knows its model's alignment heads.
2. **Onto the voice.** Alignment drifts at pauses: a word after a pause starts a little late, and the word before a
   pause would run on into it. Where the audio has silence (35 dB below its loud level), Fornax moves a word's start to
   where the voice starts and its end to where the voice stops. Speech over music, with no silence, is left as DTW
   placed it.

On 55 s of synthesized narration (eleven sentences, a known gap between each), sentence starts landed within 0.03 s of
the true onset on average (at most 0.12 s), and sentence ends within 0.08 s (at most 0.2 s).

For a model file that isn't one of the `Weights`, give its alignment heads, or `Alignment.NONE` to time words from
Whisper's timestamp tokens instead. Those are rougher: in the same test, starts were up to 0.5 s off and ends ran into
the next sentence.

<!-- snippet: SpeechToText#files -->
```java
new WhisperModel("/models/ggml-small.en.bin", WhisperModel.Alignment.SMALL_EN);
new WhisperModel("/models/ggml-custom.bin");                 // no DTW
new WhisperModel("/models/ggml-small.bin", WhisperModel.Alignment.SMALL, false);   // on the CPU
```

On a machine with no GPU, running on the CPU also needs `-Dfornax.allowCpu=true` (or `FORNAX_ALLOW_CPU=true`): otherwise
Fornax refuses to load its GPU build there ([Fornax says it found no GPU](/docs/gpu-drivers/#fornax-says-it-found-no-gpu)).

DTW needs whisper.cpp's flash attention off, so Fornax turns it off for a model with alignment heads. That costs some
speed: about 50% longer than without DTW in the test above.

## The models Fornax knows by name

`WhisperModel.Weights` names the files from whisper.cpp's own repository on Hugging Face (`ggerganov/whisper.cpp`).
`new WhisperModel(Weights)` finds the file in the Hugging Face cache; if it isn't there, it fails with the
`hf download` command that fetches it. `Weights.named("base.en")` finds one by name.

The path can also be a [model reference](/docs/guides/models/#by-reference) or a catalogue alias, downloaded on first
use. One that names a `Weights` file gets that entry's alignment:

<!-- snippet: SpeechToText#references -->
```java
new WhisperModel("whisper-base.en");                                   // an alias, with DTW
new WhisperModel("ggerganov/whisper.cpp:ggml-large-v3-turbo-q8_0.bin");  // a file in a repo, with DTW
new WhisperModel("ggerganov/whisper.cpp:ggml-small.en.bin", WhisperModel.Alignment.SMALL_EN);
```

| `Weights` | File | Size | Notes |
|---|---|---|---|
| `LARGE_V3_TURBO` | `ggml-large-v3-turbo-q5_0.bin` | 574 MB | the default: 99 languages, close to large-v3's accuracy, much faster |
| `LARGE_V3_TURBO_Q8_0` | `ggml-large-v3-turbo-q8_0.bin` | 874 MB | the same model at 8 bits |
| `BASE_EN` | `ggml-base.en.bin` | 148 MB | English only, small and fast |

Any other ggml Whisper model works too: pass its path (and its `Alignment` for DTW). The Whisper weights are MIT
licensed.

## Options

`WhisperModel.Options` is a record. `Options.DEFAULT` detects the language and decodes with a beam of 5, the way
whisper.cpp's CLI does.

<!-- snippet: SpeechToText#options -->
```java
var options = WhisperModel.Options.DEFAULT
        .withLanguage("de")             // null or "auto" detects it; an English-only model always transcribes English
        .withTranslate(true)            // German speech, English text
        .withInitialPrompt("Fornax, ggml, ComfyUI")   // names and spellings to expect
        .withBeamSize(1)                // greedy: faster; 5 (the default) is a little more accurate
        .withThreads(8)                 // 0 or less is whisper.cpp's default
        .withTemperature(-1)            // below 0 is the default: 0, raised when a window fails its checks
        .withVad(WhisperModel.VadWeights.SILERO_V6_2_0);   // transcribe only the speech (see below)
```

large-v3-turbo was trained to transcribe, not to translate, so it translates poorly; for translation, use a full
large or medium model.

The rest of the options decide what Whisper does when a 30 s window goes wrong. The defaults are whisper.cpp's.

| Option | Default | What it does |
|---|---|---|
| `vad` | none | a `Vad`: transcribe only the speech that voice activity detection finds |
| `conditionOnPreviousText` | `true` | prompt each 30 s window with the text before it, which keeps spelling, punctuation and casing consistent |
| `temperatureIncrement` | `0.2` | when a window fails one of the two checks below, decode it again this much hotter, up to 1.0 (0 never retries) |
| `entropyThreshold` | `2.4` | fail a window whose last 32 tokens have less entropy than this: repetitive text (the most is ln 32, about 3.47) |
| `logProbThreshold` | `-1` | fail a window whose average token log-probability is below this |
| `noSpeechThreshold` | `0.6` | drop a window more likely than this to be silence that also fails the log-probability check |
| `suppressNonSpeech` | `false` | never output non-speech tokens (♪, brackets, quotes, ...) |

Each window of a transcription can draw on the text before it, but one transcription never draws on another, so
transcribing the same audio twice gives the same text.

## Silence, music and repeated lines

In long silence, or under music with no voice, Whisper can hallucinate. Most often it repeats the last line it heard
until the audio ends, or writes "Thank you." The line usually comes from the window before: its text is the prompt
for the next window.

Voice activity detection (VAD) fixes this at the source. [Silero VAD](https://github.com/snakers4/silero-vad) finds the
speech first, and Whisper hears only that: the speech segments joined, with 0.1 s of silence between them. Times in
the `Transcript` are still on the original audio. Fornax maps the DTW word times back itself, because whisper.cpp maps
only its own timestamps. The VAD model is under 1 MB and runs on the CPU:

```sh
hf download ggml-org/whisper-vad ggml-silero-v6.2.0.bin
```

<!-- snippet: SpeechToText#vad -->
```java
var options = WhisperModel.Options.DEFAULT.withVad(WhisperModel.VadWeights.SILERO_V6_2_0);
// or a file, with whisper.cpp's VAD settings to tune
var tuned = WhisperModel.Options.DEFAULT.withVad(WhisperModel.Vad.of("/models/ggml-silero-v6.2.0.bin")
        .withThreshold(0.5f)            // speech probability for a 32 ms frame to count as speech
        .withMinSpeechMs(250)           // shorter speech is dropped
        .withMinSilenceMs(100)          // this much silence ends a segment
        .withSpeechPadMs(30));          // padding either side of each segment
```

`VadWeights` names `SILERO_V6_2_0` (the version whisper.cpp's README uses) and `SILERO_V5_1_2`. whisper.cpp keeps the
first VAD model it loads until the `WhisperModel` is closed.

The test case was a song's vocal stem (189 s). After the last lyric, at 158 s, the rest is near-silence and a few
held notes. With the defaults, large-v3-turbo wrote "Oh, what do you want me to kill me?" 24 times from 160 s to the
end. Each option on its own, scored against the 29 vocal onsets before 158 s:

| Options | Outro (after 158 s) | Word start vs onset: median | Within 100 ms |
|---|---|---|---|
| defaults | the line 24 times (217 words) | 28 ms | 84% |
| `withVad(SILERO_V6_2_0)` | nothing | 28 ms | 88% |
| `withConditionOnPreviousText(false)` | "Thank you." | 28 ms | 84% |
| `withEntropyThreshold(2.8f)` | ". . . Thank you." | 28 ms | 84% |
| `withSuppressNonSpeech(true)` | "Thank you." twice | 26 ms | 92% |

All of them stop the repeats. They differ in what else they change:

- **VAD** keeps the text and its punctuation, and is faster, since Whisper decodes less audio. It's the one to use
  for songs, stems, and recordings with long pauses. It isn't the default because it needs the extra model file.
- **`conditionOnPreviousText(false)`** costs nothing to set up, but without the text before it each window punctuates
  on its own, so sentences sometimes run together ("a multi-state sin How the hell"). Speech without long silences
  doesn't need it.
- **A higher `entropyThreshold`** kept the lyrics word for word here. A song's real refrains are repetitive too,
  though, and a window that fails is decoded again, hotter and less accurately, so raise it with care.
- **`suppressNonSpeech(true)`** also stopped the repeats, but it changed more than the symbols it suppresses: most
  commas and full stops went, and "ten" became "10".

## Progress and streaming

A `Listener` hears about the transcription as it runs, on the calling thread: `progress(percent)`, and each
`segment(segment)` once Whisper has finished it, with its words. The call still returns the whole `Transcript`.

<!-- snippet: SpeechToText#listener -->
```java
Transcript t = stt.transcribe(audio, WhisperModel.Options.DEFAULT, new WhisperModel.Listener() {
    @Override public void progress(int percent) { System.out.printf("%d%%%n", percent); }
    @Override public void segment(Transcript.Segment s) { System.out.println(s.text()); }
});
```

The last word of a streamed segment may end a little later in the returned transcript, which can time it against the
next segment. `cancel()`, from another thread (or from the listener), stops a call, which then throws
`CancellationException`. An exception thrown by the listener stops the call too, and comes back as the cause.

## Speed

On the Radeon 8060S with Vulkan, large-v3-turbo transcribes 55 s of narration (107 words) in about 1.8 s with a beam
of 5 (31× real time, 60 words a second), and in about 1.1 s greedy (52×, 100 words a second). The model loads in
under half a second once it is in the page cache.

## Qwen3-ASR through llama.cpp

Whisper isn't the only way. Qwen3-ASR is a speech recognition model in the form llama.cpp's `mtmd` runs: a small Qwen3
language model with an audio encoder in its mmproj. `LlamaModel` loads it like a vision model, and a user message
carries `Audio` instead of an image. It gives text only: no word timings, segments or subtitles.

```sh
hf download ggml-org/Qwen3-ASR-0.6B-GGUF Qwen3-ASR-0.6B-Q8_0.gguf mmproj-Qwen3-ASR-0.6B-Q8_0.gguf
```

<!-- snippet: SpeechToText#qwen3asr -->
```java
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;

import java.util.List;

try (var asr = new LlamaModel(LlamaModel.Weights.QWEN3_ASR_0_6B)) {
    var greedy = LlamaModel.Options.DEFAULT.withTemperature(0).withMaxTokens(1024);
    Audio talk = Audio.readWav(new File("talk.wav"));     // any rate and channels: resampled to 16 kHz mono
    String text = asr.chat(List.of(Message.user("Transcribe audio to text", talk)), greedy).text();
    System.out.println(text);
}
```

Any `Audio` works: `LlamaModel` mixes it to mono and resamples it to the rate the encoder takes (`audioSampleRate()`,
16 kHz). `supportsAudio()` says whether a loaded model takes audio; Qwen3-ASR's mmproj has no vision encoder, so it
takes no images. The model writes the language it heard before the transcript (`language English<asr_text>...`). Its
chat template marks that part as reasoning, so `chat` returns the transcript alone. Run it at temperature 0, and ask
with "Transcribe audio to text", as llama-server's transcription endpoint does (the prompt made little difference
here). Unlike Whisper, which works through 30 s windows, it takes the whole clip into its context, at 13 tokens a
second of audio, and the transcript it writes shares that context, so a long recording needs a context to match
(8192 tokens, the default, holds about eight minutes of speech with its transcript) and a `maxTokens` to match (1024
is about five minutes of speech), or cutting into pieces first.

`chatJson` takes audio as an `{"type": "audio", "index": i}` part with `chatJson(request, images, audio, onContent)`,
or as an OpenAI `input_audio` part holding a WAV, MP3 or FLAC file, which llama.cpp decodes and resamples itself (as
`llama-mtmd-cli --audio` reads a file). There `reasoning_content` holds the language:

<!-- snippet: SpeechToText#qwen3asrjson -->
```java
String mp3 = Base64.getEncoder().encodeToString(Files.readAllBytes(Path.of("talk.mp3")));
String reply = asr.chatJson("""
    {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "%s", "format": "mp3"}},
        {"type": "text", "text": "Transcribe audio to text"}]}],
     "temperature": 0, "max_tokens": 1024}""".formatted(mp3), null);
```

`LlamaModel.Weights` has `QWEN3_ASR_0_6B` and `QWEN3_ASR_1_7B` (both Q8_0, Apache-2.0). A model with slots, and a
model whose mmproj has no audio encoder, refuse audio.

### Against Whisper

The seven clips of narration in Fornax's explainer video (2.2 minutes, 336 words of synthesized American English) were
transcribed with each model and scored against their scripts: word error rate in lower case, without punctuation. The
scripts spell names as the voice says them ("Comfy U.I.", "Embaybel", "four J dot org"), and no model writes "Fornax"
as it should, so about half of every model's errors are names. Times are per clip, warm, without loading; Radeon
8060S (Strix Halo), Vulkan.

| Model | WER | Time per clip | Speed |
|---|---|---|---|
| Whisper large-v3-turbo (q5_0), beam 5, word timings | 8.0% (27 words) | 1.25 s | 15× real time |
| Whisper large-v3-turbo (q5_0), greedy, word timings | 8.0% (27 words) | 1.21 s | 16× real time |
| Qwen3-ASR 0.6B (Q8_0) | 9.2% (31 words) | 0.59 s | 32× real time |
| Qwen3-ASR 1.7B (Q8_0) | 8.9% (30 words) | 1.07 s | 18× real time |

Whisper stays the default: it is as accurate here, and it times every word. Qwen3-ASR 0.6B is about twice as fast, but
more of its errors are ordinary words ("a plain negative call", "stall the GPU"), where Whisper's are mostly names;
Whisper, though, dropped the last sentence of the final clip. Whisper's times include its DTW word timing, which costs
it about a third (see [How words are timed](#how-words-are-timed)). On the same files, Fornax's Qwen3-ASR transcripts
are the same as `llama-mtmd-cli`'s at temperature 0. To rerun the comparison, from the repository root, with ffmpeg
on the `PATH` and the models downloaded:

```sh
native/build.sh vulkan llama whisper
./gradlew :fornax-core:jar :fornax-llama:jar :fornax-whisper:jar
V=0.3.0-SNAPSHOT
CP=fornax-core/build/libs/fornax-core-$V.jar:fornax-llama/build/libs/fornax-llama-$V.jar:fornax-whisper/build/libs/fornax-whisper-$V.jar
J="java --enable-native-access=ALL-UNNAMED -Dfornax.library.path=build/native/linux-x86_64-vulkan -cp $CP scripts/AsrEval.java"
$J whisper large-v3-turbo        # beam 5; add 1 for greedy
$J llama qwen3-asr-0.6b          # or qwen3-asr-1.7b
```

## Try it

```sh
./gradlew transcribe --args='talk.wav'                                 # writes talk.words.tsv and talk.srt
./gradlew transcribe --args='talk.wav --language de --translate'
./gradlew transcribe --args='talk.wav --beam 1 --prompt "Fornax, ggml"' -Pmodel=base.en
./gradlew transcribe --args='vocals.wav --vad'                         # Silero VAD v6.2.0; or --vad silero-v5.1.2, --vad file.bin
./gradlew transcribe --args='talk.wav --no-context --entropy 2.8'      # also --suppress-nst, --temperature-inc, --logprob, --no-speech
./gradlew transcribe --args='talk.wav' -Pmodel=/path/ggml-model.bin       # any ggml Whisper file, timed without DTW
```
