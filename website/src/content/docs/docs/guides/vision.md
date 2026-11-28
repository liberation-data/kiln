---
title: "Vision: images in, text or JSON out"
description: Send images to a vision language model from Java - Qwen3-VL, PaddleOCR-VL, MedGemma - ask about them, or extract their contents to JSON that follows a schema.
---

A vision language model is a language model with a vision encoder in front. In GGUF form it comes as two files: the
language model, and an **mmproj** that holds the vision encoder and its projector. Load both into a `LlamaModel` and
user messages can carry images. It runs through llama.cpp's `mtmd` library, the way `llama-mtmd-cli` and
`llama-server --mmproj` run these models.

Everything on the [Language models](/docs/guides/language-models/) page still applies: options, streaming,
`cancel()`, `chatJson`, JSON schemas and tool calls all work with images.

## Ask about an image

```sh
hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
```

<!-- snippet: Vision#chat -->
```java
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.nio.file.Path;
import java.util.List;

try (var vlm = new LlamaModel(LlamaModel.Weights.QWEN3_VL_4B)) {        // the GGUF and its mmproj, 8192 tokens of context
    var question = Message.user("What is in this picture?", Image.read(Path.of("photo.jpg")));
    vlm.chat(List.of(question), Options.DEFAULT.withTemperature(0).withMaxTokens(512), System.out::print);
}
```

`Message.user(text, images...)` puts the images before the text. An image counts against the context like text does:
a 1000×1300 invoice is about 1,200 Qwen3-VL tokens. With images, `chat` formats the conversation with the model's own
(Jinja) chat template, as `chatJson` does, and `Result.promptTokens()` includes the image's tokens.

`supportsVision()` tells you whether a loaded model takes images. An mmproj can hold an audio encoder instead, or as
well: `supportsAudio()` says so, and `Message.user(text, audio...)` sends `Audio` the same way
([Speech to text](/docs/guides/speech-to-text/#qwen3-asr-through-llamacpp)).

### Images

`Image` is packed RGB, three bytes a pixel; alpha is dropped. There's no need to resize: the model's mmproj scales
each image to what its encoder takes.

<!-- snippet: Vision#images -->
```java
import java.awt.image.BufferedImage;

Image a = Image.read(Path.of("scan.png"));          // a file ImageIO can read: PNG, JPEG, BMP, GIF, ...
Image b = Image.decode(bytes);                      // encoded bytes, from an upload say
Image c = Image.of(bufferedImage);                  // any Java2D image
```

## The models Fornax knows by name

`LlamaModel.Weights` names the models Fornax has been tried with. `new LlamaModel(Weights)` loads the model and, for a
vision model, its mmproj, from the Hugging Face cache. If they're missing it fails with the `hf download` command,
which `downloadCommand()` also gives you.

| `Weights` | What it's for | Licence |
|---|---|---|
| `QWEN3_VL_4B` | questions about images, documents and charts; grounding (boxes on a 0 to 1000 grid) | Apache-2.0 |
| `PADDLEOCR_VL` | PaddleOCR-VL 1.6 (0.9B): reading documents. It takes task prompts, not questions (below) | Apache-2.0 |
| `MEDGEMMA_4B` | MedGemma 1.5 4B: medical images and text. Using it to diagnose or treat patients may need a health regulator's authorization | Health AI Developer Foundations terms |
| `OCRIA_1` | OCRia-1 4B: a document page to markdown, tables as HTML; [run it with thinking off](#ocria-1) | OCRia-1 license, not open source: [read it first](#ocria-1) |
| `QWEN3_4B` | Qwen3 4B Instruct: text only | Apache-2.0 |
| `QWEN3_EMBEDDING_0_6B`, `EMBEDDINGGEMMA_2`, `QWEN3_RERANKER_0_6B`, `QWEN3_RERANKER_4B` | embeddings and reranking, loaded with `Mode.EMBED` or `Mode.RERANK`; see [Embeddings and reranking](/docs/guides/embeddings/) | Apache-2.0 |
| `QWEN3_ASR_0_6B`, `QWEN3_ASR_1_7B` | speech recognition: their mmproj is an audio encoder, so they take `Audio`, not images; see [Speech to text](/docs/guides/speech-to-text/#qwen3-asr-through-llamacpp) | Apache-2.0 |

Any other vision GGUF llama.cpp runs works too:

<!-- snippet: Vision#load -->
```java
try (var vlm = new LlamaModel("/models/model.gguf", "/models/mmproj.gguf")) {   // or (gguf, mmproj, gpuLayers, contextSize)
    // ...
}
```

The vision encoder runs on the GPU unless `gpuLayers` is 0.

## Extract to JSON

`extract` asks for a reply that follows a JSON Schema, and returns the JSON text. llama.cpp turns the schema into a
grammar, so the reply can only be JSON of that shape. A thinking model's reasoning is left out.

<!-- snippet: Vision#extract -->
```java
String schema = """
        {"type": "object",
         "properties": {
           "vendor": {"type": "string"},
           "invoice_number": {"type": "string"},
           "date": {"type": "string", "description": "YYYY-MM-DD"},
           "total": {"type": "number"}},
         "required": ["vendor", "invoice_number", "date", "total"]}""";

try (var vlm = new LlamaModel(LlamaModel.Weights.QWEN3_VL_4B)) {
    String json = vlm.extract(
            List.of(Message.user("Extract this invoice's fields.", Image.read(Path.of("invoice.png")))),
            schema, Options.DEFAULT.withTemperature(0).withMaxTokens(1024));

    Object invoice = org.liberationdata.fornax.llama.Json.parse(json);
    System.out.println(org.liberationdata.fornax.llama.Json.get(invoice, "total"));
}
```

If the reply runs out of tokens before the JSON is complete, `extract` throws an `IllegalStateException`; raise
`maxTokens`. Use any JSON library to read the result, or the small `Json` helper that comes with `fornax-llama`:
`Json.parse` gives maps, lists, strings, numbers, booleans and nulls, and `Json.get(doc, "a", "b", 0)` follows a
path.

On the Radeon 8060S with Vulkan, Qwen3-VL 4B read a synthetic 1000×1300 invoice to schema-checked JSON (vendor,
number, date, four line items, subtotal, tax and total, all correct) in about 6 seconds.

## Images in `chatJson`

In a `chatJson` request, an image part `{"type": "image", "index": i}` stands for the `i`-th image in the list you
pass. OpenAI's `image_url` parts with a base64 `data:` URI work too.

<!-- snippet: Vision#chat-json -->
```java
String reply = vlm.chatJson("""
        {"messages": [{"role": "user", "content": [
             {"type": "image", "index": 0},
             {"type": "text", "text": "Read this invoice."}]}],
         "json_schema": {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]},
         "temperature": 0}""",
        List.of(Image.read(Path.of("invoice.png"))), null);
```

Tool definitions work here as they do without images.

## OCRia-1

:::caution[Not open source]
OCRia-1's weights come under LlamaIndex's OCRia-1 license, a modified OpenRAIL-M. If your organisation had more than
US$2M revenue last year, or has raised more than US$2M, you may use it only for personal use or research. No one may
use it in a service that does OCR or document extraction in competition with LlamaParse or LlamaCloud. Read the
[license](https://huggingface.co/llamaindex/OCRia-1-GGUF/blob/main/LICENSE) before you use the model or what it makes.
:::

LlamaIndex's OCRia-1 (4B, built on Qwen3.5) reads a document page to markdown, with tables as HTML. Run it as its
model card says: its own system prompt, the page image with no text, thinking off and greedy. Fetch the prompt with
the weights:

```sh
hf download llamaindex/OCRia-1-GGUF OCRia-1-Q8_0.gguf mmproj-OCRia-1-f16.gguf prompt.txt
```

Turn thinking off with `Options.withThinking(false)`: with it on, the model writes the page inside its think block
and then again after it, so the reply takes twice as long and can run out of tokens. `fornax see ocria-1` and Fornax
Studio's Inspect window turn it off for you.

<!-- snippet: Vision#ocria -->
```java
try (var ocr = new LlamaModel(LlamaModel.Weights.OCRIA_1)) {
    var page = List.of(Message.system(java.nio.file.Files.readString(Path.of("prompt.txt"))),   // from the model's repo
                       Message.user("", Image.read(Path.of("page.png"))));                     // the page, no text
    String markdown = ocr.chat(page, Options.DEFAULT.withTemperature(0).withMaxTokens(4096).withThinking(false)).text();
}
```

On the Radeon 8060S with Vulkan, it read the synthetic 1000×1300 invoice exactly, line items as an HTML table, in
about 24 s (2,347 prompt tokens with the system prompt and the image, 518 generated). Its model card asks for a
32k context for large pages. `new LlamaModel(Weights)` gives 8192 tokens, so for those use
`new LlamaModel(LlamaModel.Weights.OCRIA_1, -1, 32768)`.

## Raw completion with images

`complete(prompt, images, options, onPiece)` takes a prompt you have formatted yourself. Put
`LlamaModel.mediaMarker()` in it once per image, in order, where the model's image tokens go.

## What the models are good at

From trying them on synthetic documents and parts:

- **PaddleOCR-VL takes task prompts, not instructions.** Ask it `OCR:`, `Table Recognition:`, `Formula Recognition:`,
  `Chart Recognition:`, `Seal Recognition:` or `Spotting:`, and use it greedy (temperature 0). It ignores a request to
  fill a schema: a grammar-constrained reply comes back as well-formed JSON with the wrong values. So read the text
  with PaddleOCR-VL, then extract with a text model. Its OCR of the same invoice took about 1.5 s, at about 200
  tokens a second.
- **Qwen3-VL 4B grounds what is there, not what is missing.** Asked for defects on a tray of parts, it boxed a dented
  part accurately, but never found an empty slot, with or without a schema. Its boxes are on a 0 to 1000 grid over
  the image.
- **MedGemma** (Gemma 3's vision) gives each image 256 tokens. It read the invoice to JSON correctly in about 6 s.
- **Images are quick to read.** The 1000×1300 invoice was read (encoded and prefilled) in about 1.6 s by Qwen3-VL;
  a 720×480 picture in about 0.4 s.

## Try it

```sh
hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
./gradlew see --args='photo.jpg "What is in this picture?"'           # -Pmodel=qwen3-vl-4b|paddleocr-vl|medgemma-4b
./gradlew see --args='form.png "Fill in the fields."' -Pschema=schema.json
./gradlew invoice     # draws a made-up invoice and reads it into JSON; -Pmodel=paddleocr-vl reads it with OCR first
./gradlew defects     # a tray of parts with a dented one: Qwen3-VL boxes the defects (out/defects-found.png)
```
