---
title: OpenAI-compatible server
description: Serve Fornax's models over an OpenAI-compatible HTTP API with fornax server or FornaxServer, so any OpenAI client, agent framework or tool can use them with no Java code.
---

:::note[Coming in 0.3.0]
`fornax server` and the `fornax-server` module are on `main` and ship in the next release, 0.3.0. They aren't in
0.2.0 on Maven Central. Until then, build Fornax from source (`./gradlew :fornax-cli:installDist` for the command,
`./gradlew publishToMavenLocal` for the module).
:::

`fornax server` puts Fornax's engines behind an HTTP API in OpenAI's shape. Anything that talks to OpenAI (the
official Python and JavaScript clients, LangChain, an agent framework, a chat UI, `curl`) can then use local models
by pointing its base URL at the server:

```sh
fornax server -m qwen3-4b
curl -s localhost:54154/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:54154/v1", api_key="unused")
reply = client.chat.completions.create(model="qwen3-4b", messages=[{"role": "user", "content": "Hi"}])
print(reply.choices[0].message.content)
```

It is the module `fornax-server`, JDK only (`com.sun.net.httpserver`, no framework), which an app can also start
itself:

<!-- snippet: OpenAiServer#start -->
```java
import org.liberationdata.fornax.server.FornaxServer;

try (var server = FornaxServer.builder().port(54154).model("qwen3-4b").build().start()) {
    System.out.println("OpenAI base URL: " + server.url());   // http://127.0.0.1:54154/v1
    server.await();
}
```

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | chat with a language or vision model, streamed or not |
| `GET /v1/models` | the models it was started with, those loaded, and those downloaded |
| `POST /v1/embeddings` | embeddings, as floats or base64 |
| `POST /v1/images/generations` | images, as base64 PNGs (`b64_json`) |
| `POST /v1/audio/speech` | text to speech, as a WAV |
| `POST /v1/audio/transcriptions` | speech to text, from a WAV file |

`GET /health` answers `{"status": "ok"}`. Errors are OpenAI's: the HTTP status and
`{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`, so a client's own exceptions work
(`openai.NotFoundError` for a model that isn't there, for one). A prompt longer than the context (`--ctx`) is a 400
with OpenAI's code, `context_length_exceeded`.

### Chat completions

Chat runs [`LlamaModel.chatJson`](/docs/guides/language-models/) underneath: the model's own chat template and
tool-call format, with llama-server's templates and parsers. A request takes:

- `messages`, with `system`, `user`, `assistant` and `tool` roles, as OpenAI sends them
- `tools` and `tool_choice` (`auto`, `none`, `required`, or a function by name); the reply's `tool_calls` and
  `finish_reason: "tool_calls"` come back as OpenAI's, and the tool results go back as `tool` messages
- `response_format`: `json_schema` (the reply is constrained to the schema) or `json_object`. An object in a schema,
  here or in a tool's parameters, can have at most 256 optional properties (a 400 says so), as llama.cpp's grammar can't take more
- `temperature`, `top_p`, `seed`, `max_tokens` (or `max_completion_tokens`), and `top_k` and `min_p` beyond OpenAI's
- `stop`: a string or a list; the reply ends before the first one, and the model stops there
- `stream: true`: server-sent events in OpenAI's chunk format, ending with `data: [DONE]`;
  `stream_options: {"include_usage": true}` adds the token counts in a last chunk
- images, for a vision model (`qwen3-vl-4b`, `paddleocr-vl`, `medgemma-4b`, `ocria-1`): `image_url` content parts with a base64
  `data:` URI of a PNG, JPEG, GIF or BMP, up to 67 megapixels (8192 x 8192) in all a request; the server doesn't fetch URLs
- `chat_template_kwargs: {"enable_thinking": false}` turns a reasoning model's thinking off; its other keys go to the
  chat template as variables, as llama-server passes them
- `reasoning_effort` (`low`, `medium` or `high`) sets how long a model whose template takes it (gpt-oss) reasons

A reasoning model's thinking comes back in `reasoning_content`, apart from `content`. When streaming, the content
arrives as it is generated; the thinking and the tool calls (each call with its `index`) come at the end, before the finish reason. A tool
call's `id` is unique to it, for the `tool` message that answers it.
`n` must be 1, and there are no logprobs.

A stream's status goes out with its first event, or with a keep-alive comment (`: keep-alive`, every 3 seconds while
the model loads or thinks), so a failure after that (a model that won't load, an engine error) can't be a status: it
comes as OpenAI's error event, `data: {"error": {...}}`, with no `[DONE]`, which the OpenAI clients raise as an error.

When a client disconnects mid-stream, the generation stops, so the model is free for the next request. A
non-streaming request runs to the end.

### Embeddings

`input` is a string or a list of up to 2048 (as OpenAI's), none empty (token arrays aren't taken). The model is `qwen3-embedding-0.6b` (Qwen3-Embedding
0.6B, 1024 dimensions) unless a request names another embedding GGUF by `org/name:TAG` or path (one the server was
started with, or any with `--allow-downloads`). `dimensions` keeps
the first n and normalises them again, which Qwen3-Embedding is trained for. `encoding_format: "base64"`, which the
OpenAI clients ask for by default, is supported. Vectors are normalised, so a dot product is the cosine similarity.
`usage` counts the tokens the texts took, as OpenAI's does (`prompt_tokens` and `total_tokens`, the same).

### Images

```python
import base64
r = client.images.generate(model="flux2-klein", prompt="a brick kiln at dusk", size="512x512",
                           extra_body={"seed": 7, "steps": 20})
open("kiln.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```

`model` is an image alias (`flux2-klein`, `qwen-image`, `z-image-turbo`, `ernie-image-turbo`); `size` is `WIDTHxHEIGHT` in multiples of 16, up to 4096 a side and 4
megapixels (2048x2048, or 4096x1024; default 1024x1024); `n` makes up to 8, with seeds counting up from `seed`.
Beyond OpenAI's: `seed`, `steps` (default 20, 8 for the two Turbos; at most 100)
and `cfg_scale` (4.0 for flux2-klein, 2.5 for the Qwen-Image models, 1.0 for z-image-turbo and ernie-image-turbo). Images come back as `b64_json` PNGs; the
server keeps no files, so `response_format: "url"` is refused.

### Speech and transcription

`/v1/audio/speech` speaks `input` (up to 4096 characters, as OpenAI's) with `kokoro` unless `model` names another speech alias. `voice` is one of the
model's voices (`fornax speak --list-voices`); OpenAI's voice names (`alloy`, `nova`, ...) get the model's default.
`voxcpm2` has no voices: describe one in parentheses at the start of `input` (`"(A warm Australian woman)G'day."`).
The reply is a WAV, so ask for `response_format: "wav"` (an OpenAI client asking for MP3 gets an error).

`/v1/audio/transcriptions` takes a WAV `file` (multipart, as the clients send it; convert other formats first,
`ffmpeg -i in.mp3 in.wav`) at 8 to 384 kHz, with `whisper-base.en` unless `model` names another (`whisper-large-v3-turbo` for 99
languages), and `language` and `prompt`. `response_format` is `json` (the default), `text`, `srt`, `vtt` or
`verbose_json`, which has the segments, each word's start and end, and the language spoken by its English name
(`english`), as OpenAI gives it.

## Models

A request's `model` is one of the `-m` models, or a [catalogue model](/docs/guides/models/#the-catalogue) that is
already downloaded (`fornax pull`), as is `qwen3-embedding-0.6b`. Any other is refused with a 404 (`model_not_found`, whose message says how to
allow it), so that whoever can call the server can't make it download many gigabytes. `--allow-downloads`
(`.allowDownloads(true)`) lets a request name anything Fornax loads: a catalogue alias, an `org/name[:TAG]` reference
or a path to a GGUF, downloaded the first time it's asked for.

An alias in any case (`QWEN3-4B`) is the same model as its catalogue id (`qwen3-4b`), loaded once; an
`org/name:TAG` reference or a path is taken as given, as the Hub and the filesystem may care about its case.

A model is loaded on first use and kept loaded in a [`ModelPool`](/docs/guides/model-pool/): within `--budget-gb` (models' file sizes, plus the KV cache of a model with slots), the least recently used are
unloaded to make room, and a model unused for `--idle-minutes` is unloaded. A model answers one request at a time;
others wait for it, unless it has slots (below).

### Many clients at once: `--parallel`

`--parallel N` (`.parallel(N)`) loads each language model with N
[slots](/docs/guides/language-models/#many-requests-at-once-slots): it answers N chat requests at once, decoding them
together in one batch (continuous batching, as llama-server's `-np` does), and more wait for a free slot. Each slot has
its own `--ctx` tokens, so the context takes N times the memory, and `--budget-gb` counts it (the KV cache's size, from
the model's header, besides its files). A client that hangs up, or whose deadline passes, stops only its own request.
A streaming client that reads more slowly than the model generates is never waited for, which would hold up every
slot: once it falls a whole buffer behind (1024 pieces), its request stops with an error event
([overflow](/docs/guides/threading/#streaming)).
Vision models still answer one request at a time, since a model with slots takes no images yet. Under
`FORNAX_ISOLATION=process` ([isolation](/docs/guides/isolation/)) each model runs in a worker process of its own,
and one with slots batches there. The
[benchmarks](/docs/benchmarks/) show what it buys: tokens a second across clients, and their wait for the first token.

The `-m` models are downloaded first if need be (after the port is taken, so a port in use or a bad `--host` is said
before any download) and loaded before the server starts answering. The first chat model
answers a request that names none. A reference or path whose name has "embed" in it is loaded for embeddings; any
other for chat.

## Options

```text
fornax server [-m MODEL]... [options]
  -m, --model M        a model to load at start (repeat for more): an alias, org/name:TAG or a path
  --port N             default 54154
  --host ADDR          default 127.0.0.1 (this machine only); 0.0.0.0 for every interface, with an API key
  --api-key K          require Authorization: Bearer K (default FORNAX_API_KEY, if set)
  --cors ORIGIN        let web pages from ORIGIN call it (repeat for more; * for any)
  --budget-gb N        memory for loaded models, by their files' size (default 24)
  --idle-minutes N     unload a model unused this long (default 15; 0 = never)
  --ctx N              context size for language and embedding models (default 8192; 0 = the model's own)
  --parallel N         chat requests each language model answers at once, decoded together in one batch
                       (default 1); each gets its own --ctx, which --budget-gb counts. Vision models: 1
  --allow-downloads    let a request name any model, downloading it on first use (default: only the -m
                       models and catalogue models already downloaded)
```

`FornaxServer.builder()` has the same settings, plus the hub to download with, a download progress listener, a
log line consumer, the largest request body (`maxBodyBytes`) and the most request body held at once
(`maxBufferedBytes`).

## Limits

So that one request can't take all the memory, or a model for days, a request beyond these is refused (400, or 413
for the body):

- the body: 64 MB (`maxBodyBytes`), with JSON nested at most 128 levels deep
- chat: images of up to 67 megapixels in all; 4 `stop` sequences of up to 256 characters each; the reply is at most
  the context (`--ctx`), less the prompt
- embeddings: 2048 inputs, each within the context
- images: 4 megapixels each, 8 a request, 100 steps
- speech: 4096 characters
- transcription: a WAV at 8 to 384 kHz (the body limit is about 35 minutes of 16 kHz 16-bit mono). Decoded, a WAV
  is several times its size, so one is decoded at a time, while it is transcribed; others wait with just their bytes

A request's body is read whole before it waits for its model, and kept until it is answered. So that many at once
can't fill the heap, the bodies held at once are limited to 256 MB in all (`maxBufferedBytes`, at least
`maxBodyBytes`): past that, a request is answered 503 (`server_busy`, with `Retry-After: 1`) rather than queued, and
the OpenAI clients retry it.

A refused request's body (a 413, a 503, a 403) is read to its end first, up to 64 KB more of it, so that the client
gets the error rather than a reset connection. Past that, the error says `Connection: close`, and a client still
sending a large body may see the connection reset instead of the reply.

A client that sends its request slowly, a byte at a time, would hold a connection and a thread for as long as it
liked. `fornax server` gives a client 120 seconds to send its request, headers and body, and takes 512 connections at
once. These are the JDK HTTP server's own settings, which it reads once per JVM, so `FornaxServer` doesn't set them
for an application it runs in. To set them in yours, pass them to `java` (or set them before the first HTTP server is
made); for `fornax server`, a `-D` in `FORNAX_OPTS` overrides its values:

```text
java -Dsun.net.httpserver.maxReqTime=120 -Djdk.httpserver.maxConnections=512 ...
FORNAX_OPTS=-Dsun.net.httpserver.maxReqTime=600 fornax server -m qwen3-4b
```

Leave `sun.net.httpserver.maxRspTime` unset: its clock runs from the request's end to the reply's, so it would cut
off a long stream, or a request waiting for a model to load.

## Who may call it

The server listens on 127.0.0.1 unless `--host` says otherwise, so only programs on the same machine can reach it.
That alone doesn't keep out the web pages open in your browser, which can send requests to 127.0.0.1 too. So:

- **On loopback, the `Host` must be the server's own** (`127.0.0.1:54154`, any other `127.x.y.z:54154`,
  `localhost:54154` or `[::1]:54154`). A
  page that rebinds its own domain name to 127.0.0.1 sends its own name, and is refused (403).
- **A request from a web page is refused** (403): browsers send an `Origin` with every cross-origin request.
  `--cors https://app.example` allows that origin, which then gets CORS headers, its preflight answered and the
  response readable (a refusal too, so a page can tell a missing API key). `--cors '*'` allows any origin.
- **`--api-key`** (or `FORNAX_API_KEY`) makes every request need `Authorization: Bearer <key>` (401 otherwise), which
  is what the OpenAI clients send as their `api_key`. Set one before listening on another address: the `Host` check
  is off then, as other machines reach it by other names.
- **A server error (500) names the file that failed, not its directory**, so a caller doesn't learn the machine's
  paths.
