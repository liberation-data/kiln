// Small, stable C ABI over llama.cpp for Fornax (Java FFM). llama.cpp's own API passes param structs by
// value and is mid-migration (llama_batch -> llama_batch_ext), so Java binds to this instead.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fornax_llama fornax_llama;

// Called for each generated piece of text. Return non-zero to stop generation.
typedef int (*fornax_llama_token_cb)(const char * piece, void * user);

const char * fornax_llama_last_error(void);

// n_gpu_layers: -1 = all. n_ctx: 0 = model default.
fornax_llama * fornax_llama_create(const char * model_path, int n_gpu_layers, int n_ctx);

// Formats messages with the model's chat template, then generates. Each call starts from a fresh context.
// roles[i] is "system", "user" or "assistant". *out_text is malloc'd; free it with fornax_llama_free.
// seed: 0xFFFFFFFF = random. temperature <= 0 = greedy.
int fornax_llama_chat(fornax_llama * h, const char ** roles, const char ** contents, int n_messages,
                    int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                    fornax_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated);

// Raw completion of an already-formatted prompt.
int fornax_llama_complete(fornax_llama * h, const char * prompt, int max_tokens, float temperature, float top_p,
                        float min_p, uint32_t seed, fornax_llama_token_cb cb, void * user, char ** out_text,
                        int * out_n_prompt, int * out_n_generated);

// OpenAI-style chat with tool calling, through llama.cpp's chat templates and parsers (as llama-server does).
// request_json: {"messages": [...], "tools": [...], "tool_choice": "auto|required|none", "parallel_tool_calls": bool,
//                "max_tokens": n, "temperature": t, "top_p": p, "min_p": p, "top_k": k, "seed": s, "enable_thinking": bool,
//                "json_schema": {...}, "chat_template_kwargs": {...}, "reasoning_effort": "low|medium|high"}
// messages and tools use OpenAI's chat completions format.
// *out_response_json: {"message": {"role", "content", "reasoning_content", "tool_calls": [...]},
//                      "finish_reason": "stop|length|tool_calls", "usage": {"prompt_tokens", "completion_tokens"}}
// cb (optional) receives each new piece of the message's content as it's parsed, and an empty piece for each token that
// adds no content (reasoning, a tool call), so that returning non-zero can stop those too. Free the response with
// fornax_llama_free.
int fornax_llama_chat_json(fornax_llama * h, const char * request_json, fornax_llama_token_cb cb, void * user,
                         char ** out_response_json);

// --- Images (vision language models, through llama.cpp's mtmd) ---
//
// Images go in as packed RGB, 3 bytes a pixel, row by row: rgb[i] holds widths[i] * heights[i] * 3 bytes.

// A vision (or audio) language model: the language model's GGUF plus its mmproj GGUF (the vision or audio encoder,
// or both, and the projector).
// A handle from fornax_llama_create_mm also does everything a fornax_llama_create handle does.
fornax_llama * fornax_llama_create_mm(const char * model_path, const char * mmproj_path, int n_gpu_layers, int n_ctx);

// 1 if the handle takes images (it was created with an mmproj that has a vision encoder), else 0.
int fornax_llama_supports_vision(fornax_llama * h);

// The marker fornax_llama_complete_images replaces with an image (llama.cpp's mtmd_default_marker()).
const char * fornax_llama_media_marker(void);

// fornax_llama_chat with images: message i carries the next image_counts[i] images (in order), placed before its text.
// The chat is formatted with the model's own (Jinja) chat template, as fornax_llama_chat_json does.
int fornax_llama_chat_images(fornax_llama * h, const char ** roles, const char ** contents, const int * image_counts,
                           int n_messages, const unsigned char ** rgb, const uint32_t * widths, const uint32_t * heights,
                           int n_images, int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                           fornax_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated);

// fornax_llama_complete with images: the prompt holds fornax_llama_media_marker() once for each image, in order.
// out_n_prompt counts the image tokens too.
int fornax_llama_complete_images(fornax_llama * h, const char * prompt, const unsigned char ** rgb, const uint32_t * widths,
                               const uint32_t * heights, int n_images, int max_tokens, float temperature, float top_p,
                               float min_p, uint32_t seed, fornax_llama_token_cb cb, void * user, char ** out_text,
                               int * out_n_prompt, int * out_n_generated);

// fornax_llama_chat_json with images. In a message's content array, {"type": "image", "index": i} stands for the i-th
// RGB buffer given here, and an OpenAI {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} part
// is decoded too (PNG, JPEG, BMP, GIF). "json_schema" constrains the reply to a JSON schema, as in fornax_llama_chat_json.
int fornax_llama_chat_json_images(fornax_llama * h, const char * request_json, const unsigned char ** rgb,
                                const uint32_t * widths, const uint32_t * heights, int n_images,
                                fornax_llama_token_cb cb, void * user, char ** out_response_json);

// --- Audio (speech recognition and audio language models, through llama.cpp's mtmd) ---
//
// Audio goes in as mono float PCM in [-1, 1] at the model's own sample rate (fornax_llama_audio_sample_rate): pcm[i]
// holds n_samples[i] floats. mtmd doesn't resample raw PCM, so audio at another rate must be resampled first.

// 1 if the handle takes audio (it was created with an mmproj that has an audio encoder), else 0.
int fornax_llama_supports_audio(fornax_llama * h);

// The sample rate the audio encoder takes, in Hz (16000 for Whisper-style encoders), or -1 when the handle takes no
// audio.
int fornax_llama_audio_sample_rate(fornax_llama * h);

// fornax_llama_chat_json_images with audio too. In a message's content array, {"type": "audio", "index": i} stands for
// the i-th PCM buffer given here, and an OpenAI {"type": "input_audio", "input_audio": {"data": "<base64>",
// "format": "wav"}} part is decoded (WAV, MP3, FLAC, told apart by their bytes) and resampled to the model's rate.
// Image parts work as in fornax_llama_chat_json_images. A model whose mmproj has no audio encoder refuses audio parts.
int fornax_llama_chat_json_media(fornax_llama * h, const char * request_json, const unsigned char ** rgb,
                               const uint32_t * widths, const uint32_t * heights, int n_images, const float ** pcm,
                               const uint32_t * n_samples, int n_audio, fornax_llama_token_cb cb, void * user,
                               char ** out_response_json);

// --- Embeddings and reranking (pooled output, as llama-embedding and llama-server's /embedding and /rerank) ---
//
// A pooled handle runs the model with llama.cpp's embeddings output and the pooling type its GGUF sets (mean, cls,
// last; rank for a reranker), and takes only the calls below: fornax_llama_chat and the rest fail on it. Several inputs
// go in one batch, each as its own sequence; one input may be up to n_ctx tokens. cb (optional) is called with an
// empty piece after each batch; returning non-zero stops the call, which then fails with "cancelled" (rc 13).

// rerank 0: an embedding model (its GGUF sets a pooling type other than rank). rerank 1: a reranker (rank pooling,
// with a classifier head). n_gpu_layers: -1 = all. n_ctx: 0 = model default.
fornax_llama * fornax_llama_create_pooled(const char * model_path, int n_gpu_layers, int n_ctx, int rerank);

// The length of one embedding (floats).
int fornax_llama_n_embd(fornax_llama * h);

// Embeds n texts into out (n * fornax_llama_n_embd floats, row by row). normalize non-zero: each row L2-normalised.
// Texts are tokenized with the vocab's own BOS/EOS added, and not parsed for special tokens. out_tokens, if not NULL,
// gets each text's token count (n ints), as a usage report counts them.
int fornax_llama_embed(fornax_llama * h, const char ** texts, int n, int normalize, float * out, int * out_tokens,
                     fornax_llama_token_cb cb, void * user);

// Scores n documents against a query into out_scores (n floats, in the documents' order), as llama-server's /rerank:
// the GGUF's "rerank" prompt template with {query} and {document} filled in. For Qwen3-Reranker a score is the
// probability (0 to 1) that the document answers the query.
int fornax_llama_rerank(fornax_llama * h, const char * query, const char ** docs, int n, float * out_scores,
                      fornax_llama_token_cb cb, void * user);

// ---- continuous batching: several requests decoded together, as llama-server's parallel slots are ----
//
// A batched handle holds n_slots sequences, each with n_ctx_slot tokens of context of its own. Requests are added to
// free slots and advance together, one fornax_llama_batch_step at a time: each step decodes one token for every
// running request plus chunks of new prompts, in one batch, then samples each request with its own sampler (and its
// own grammar, stops and seed). A handle is used from one thread at a time; the serial generation calls refuse it.

// Called with each request's new text (seq: its slot). Return non-zero to stop that request.
typedef int (*fornax_llama_piece_cb)(int seq, const char * piece, void * user);
// Called once a request ends: rc 0 with its response JSON (as fornax_llama_chat_json's, or for a raw prompt
// {"text", "finish_reason", "usage"}), or the error code and message. The slot is free again by then.
typedef void (*fornax_llama_done_cb)(int seq, int rc, const char * response, void * user);

// n_slots: 1 to llama_max_parallel_sequences(). n_ctx_slot: each slot's context (> 0).
fornax_llama * fornax_llama_create_batched(const char * model_path, int n_gpu_layers, int n_ctx_slot, int n_slots);
int fornax_llama_n_slots(fornax_llama * h);       // 0 for a handle that isn't batched
int fornax_llama_n_ctx_slot(fornax_llama * h);    // a slot's context (the whole context when not batched)

// Adds a request to a free slot (*out_seq). It starts with the next step. Fails (15) when every slot is busy, and as
// the serial call would for a bad request or a prompt longer than a slot's context; the other slots are unaffected.
int fornax_llama_batch_add_json(fornax_llama * h, const char * request_json, int * out_seq);  // as fornax_llama_chat_json; no images
int fornax_llama_batch_add_raw(fornax_llama * h, const char * prompt, int max_tokens, float temperature, float top_p,
                               float min_p, uint32_t seed, int * out_seq);               // as fornax_llama_complete
int fornax_llama_batch_add_chat(fornax_llama * h, const char ** roles, const char ** contents, int n_messages,
                                int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                                int * out_seq);                                          // as fornax_llama_chat
// Ends a request at the next step, with what it has generated.
void fornax_llama_batch_cancel(fornax_llama * h, int seq);
// One step for every busy slot. Returns how many are still busy, or minus an error code when the decode failed
// (the requests in that batch have each had on_done with the error).
int fornax_llama_batch_step(fornax_llama * h, fornax_llama_piece_cb on_piece, fornax_llama_done_cb on_done, void * user);

// An estimate of the KV cache n_ctx tokens of context take (F16 keys and values, every layer), from the GGUF's header
// alone: no weights are loaded. -1 when the file can't be read.
int64_t fornax_llama_kv_bytes(const char * model_path, int n_ctx);

void fornax_llama_free(void * p);
void fornax_llama_destroy(fornax_llama * h);

#ifdef __cplusplus
}
#endif
