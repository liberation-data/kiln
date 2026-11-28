#include "fornax_llama.h"
#include "llama.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {
thread_local std::string g_err;
int fail(const std::string & e, int rc) { g_err = e; return rc; }
char * dup(const std::string & s) {
    char * p = (char *) malloc(s.size() + 1);
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}
}  // namespace

// One sequence of a batched handle (fornax_llama_create_batched): a request from its prompt to its last token.
struct fornax_llama_slot {
    bool busy = false, cancelled = false;
    std::unique_ptr<common_chat_session> session;  // a chat request; null for a raw prompt
    common_sampler * chat_smpl = nullptr;           // with a session: its grammar, stops and seed
    llama_sampler * raw_smpl = nullptr;             // a raw prompt's
    std::vector<llama_token> prompt;
    size_t fed = 0;                                 // prompt tokens decoded so far
    llama_token next = LLAMA_TOKEN_NULL;            // sampled, to decode in the next step
    int pos = 0, generated = 0, max_tokens = -1, i_batch = -1, n_adding = 0;
    std::string raw, sent_content, text;            // text: a raw prompt's output
};

struct fornax_llama {
    common_chat_templates_ptr tmpls;
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_batch_ext * batch = nullptr;
    mtmd_context *  mctx  = nullptr;  // only for a handle from fornax_llama_create_mm
    enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_NONE;  // not NONE: a handle from fornax_llama_create_pooled
    int pooled_max = 0;  // a pooled handle's largest context: it starts smaller and grows to fit (run_pooled)
    std::vector<fornax_llama_slot> slots;  // not empty: a handle from fornax_llama_create_batched
};

namespace {
// A pooled handle (fornax_llama_create_pooled) has no logits to sample from.
int require_generative(fornax_llama * h) {
    if (!h->slots.empty()) return fail("this model was loaded with slots: its requests go through fornax_llama_batch_add", 16);
    if (h->pooling != LLAMA_POOLING_TYPE_NONE)
        return fail(h->pooling == LLAMA_POOLING_TYPE_RANK ? "this model was loaded as a reranker: it can only rerank"
                                                          : "this model was loaded for embeddings: it can only embed", 14);
    return 0;
}

llama_sampler * simple_sampler(float temperature, float top_p, float min_p, uint32_t seed) {
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (top_p > 0 && top_p < 1) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }
    return smpl;
}

// Decodes `cur` (one batch) at positions pos.., with logits for its last token.
int decode(fornax_llama * h, const llama_token * cur, int n_cur, int pos) {
    llama_batch_ext_clear(h->batch);
    for (int i = 0; i < n_cur; i++) {
        int idx = llama_batch_ext_add_token(h->batch, 0, cur[i]);
        llama_pos p = pos + i;
        llama_batch_ext_set_pos(h->batch, idx, &p);
    }
    llama_batch_ext_set_output_logits(h->batch, n_cur - 1, true);
    return llama_process(h->ctx, LLAMA_PROCESS_TYPE_DECODE, h->batch);
}

// Samples and decodes until end of turn, max_tokens or a full context, starting from logits already computed for
// the prompt (which ends at position pos and fills `cells` of the context). Positions and cells differ after an
// M-RoPE image (Qwen-VL: ~40 positions for ~1200 tokens), and it's the cells that run out.
int generate(fornax_llama * h, llama_sampler * smpl, int pos, int cells, int max_tokens, fornax_llama_token_cb cb,
             void * user, std::string & out, int & generated) {
    const int n_ctx = (int) llama_n_ctx(h->ctx);
    while (max_tokens <= 0 || generated < max_tokens) {
        llama_token tok = llama_sampler_sample(smpl, h->ctx, -1);
        if (llama_vocab_is_eog(h->vocab, tok)) break;
        char buf[256];
        int m = llama_token_to_piece(h->vocab, tok, buf, sizeof buf, 0, true);
        if (m < 0) return fail("token_to_piece failed", 4);
        std::string piece(buf, m);
        out += piece;
        generated++;
        if (cb && cb(piece.c_str(), user) != 0) break;
        if (cells + 1 > n_ctx) break;  // context full: return what we have
        if (int r = decode(h, &tok, 1, pos); r != 0) return fail("llama_process failed: " + std::to_string(r), 3);
        pos++;
        cells++;
    }
    return 0;
}

// Packed RGB buffers (3 bytes a pixel) as mtmd bitmaps, in order.
std::vector<mtmd::bitmap_ptr> rgb_bitmaps(const unsigned char ** rgb, const uint32_t * widths, const uint32_t * heights, int n) {
    std::vector<mtmd::bitmap_ptr> out;
    for (int i = 0; i < n; i++) {
        if (!rgb[i] || widths[i] == 0 || heights[i] == 0) throw std::invalid_argument("image " + std::to_string(i) + " is empty");
        out.emplace_back(mtmd_bitmap_init(widths[i], heights[i], rgb[i]));
    }
    return out;
}

std::vector<mtmd_bitmap *> raw(const std::vector<mtmd::bitmap_ptr> & v) {
    std::vector<mtmd_bitmap *> out;
    for (const auto & b : v) out.push_back(b.get());
    return out;
}

std::string base64_decode(const std::string & in) {
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        if (isspace(c)) continue;
        size_t d = chars.find(c == '-' ? '+' : c == '_' ? '/' : (char) c);
        if (d == std::string::npos) throw std::invalid_argument("invalid base64 in a media part");
        val = ((val << 6) | (int) d) & 0xFFFFFF;
        bits += 6;
        if (bits >= 0) { out.push_back((char) ((val >> bits) & 0xFF)); bits -= 8; }
    }
    return out;
}

// The marker for media inside a chat: no user text will contain it, so only Fornax's own media parts become media.
const std::string & private_marker() {
    static const std::string m = [] {
        std::random_device rd;
        char buf[64];
        snprintf(buf, sizeof buf, "<__fornax_media_%08x%08x__>", rd(), rd());
        return std::string(buf);
    }();
    return m;
}

// Mono float PCM buffers, at the model's sample rate, as mtmd audio bitmaps, in order.
std::vector<mtmd::bitmap_ptr> pcm_bitmaps(const float ** pcm, const uint32_t * n_samples, int n) {
    std::vector<mtmd::bitmap_ptr> out;
    for (int i = 0; i < n; i++) {
        if (!pcm[i] || n_samples[i] == 0) throw std::invalid_argument("audio " + std::to_string(i) + " is empty");
        out.emplace_back(mtmd_bitmap_init_from_audio(n_samples[i], pcm[i]));
    }
    return out;
}

// Images and audio found in a request. `owned` keeps the ones decoded here alive; `all` is every one, in prompt order.
struct request_media {
    std::vector<mtmd::bitmap_ptr> owned;
    std::vector<mtmd_bitmap *> all;
};

// A media file decoded by mtmd (images: PNG, JPEG, ...; audio: WAV, MP3, FLAC), refused if it isn't the kind the
// part said it was: an image_url that holds audio would otherwise reach the model as audio, and the reverse.
mtmd_bitmap * decode_media(fornax_llama * h, const std::string & bytes, bool audio, const char * part) {
    auto w = mtmd_helper_bitmap_init_from_buf(h->mctx, (const unsigned char *) bytes.data(), bytes.size(), false,
                                              mtmd_helper_init_opt_default());
    if (w.video_ctx) {  // a video's bitmap reads its frames from video_ctx as it's tokenized: freed together, unused
        if (w.bitmap) mtmd_bitmap_free(w.bitmap);
        mtmd_helper_video_free(w.video_ctx);
        throw std::invalid_argument(std::string("an ") + part + " part holds a video, which isn't supported");
    }
    if (!w.bitmap)
        throw std::invalid_argument(std::string("could not decode an ") + part + (audio ? " audio file" : " image"));
    if (mtmd_bitmap_is_audio(w.bitmap) != audio) {
        mtmd_bitmap_free(w.bitmap);
        throw std::invalid_argument(std::string("an ") + part + " part holds " + (audio ? "an image" : "audio"));
    }
    return w.bitmap;
}

// Replaces the image and audio parts of the request's messages with media markers, collecting the media in order:
// {"type": "image", "index": i} is images[i]; {"type": "image_url", ...} is a base64 data: URI (PNG, JPEG, ...);
// {"type": "audio", "index": i} is audio[i]; {"type": "input_audio", "input_audio": {"data": base64, "format": ...}}
// is an encoded audio file (WAV, MP3, FLAC: mtmd tells them apart by their bytes, so format isn't needed).
void take_media(fornax_llama * h, common_json & messages, const std::vector<mtmd_bitmap *> & images,
                const std::vector<mtmd_bitmap *> & audio, request_media & out) {
    if (!messages.is_array()) return;
    for (auto & m : messages) {
        if (!m.is_object() || !m.contains("content") || !m.at("content").is_array()) continue;
        for (auto & p : m.at("content")) {
            std::string type = p.is_object() ? p.value("type", "") : "";
            const bool is_image = type == "image" || type == "image_url";
            const bool is_audio = type == "audio" || type == "input_audio";
            if (!is_image && !is_audio) continue;
            if (is_image && (!h->mctx || !mtmd_support_vision(h->mctx)))
                throw std::invalid_argument(h->mctx ? "this model takes no images (its mmproj has no vision encoder)"
                                                    : "this model takes no images (load it with its mmproj)");
            if (is_audio && (!h->mctx || !mtmd_support_audio(h->mctx)))
                throw std::invalid_argument(h->mctx ? "this model takes no audio (its mmproj has no audio encoder)"
                                                    : "this model takes no audio (load it with its mmproj)");
            if (type == "image" || type == "audio") {
                const auto & given = type == "image" ? images : audio;
                int i = p.at("index").get<int>();
                if (i < 0 || i >= (int) given.size())
                    throw std::invalid_argument(type + " index " + std::to_string(i) + " out of range (" +
                                                std::to_string(given.size()) + " given)");
                out.all.push_back(given[i]);
            } else if (type == "image_url") {
                std::string url = p.at("image_url").at("url").get<std::string>();
                size_t comma = url.find(',');
                if (url.rfind("data:", 0) != 0 || comma == std::string::npos || comma < 6 ||
                    url.compare(comma - 6, 6, "base64") != 0)
                    throw std::invalid_argument("image_url must be a base64 data: URI");
                out.owned.emplace_back(decode_media(h, base64_decode(url.substr(comma + 1)), false, "image_url"));
                out.all.push_back(out.owned.back().get());
            } else {
                std::string data = p.at("input_audio").at("data").get<std::string>();
                out.owned.emplace_back(decode_media(h, base64_decode(data), true, "input_audio"));
                out.all.push_back(out.owned.back().get());
            }
            p = common_json{{"type", "media_marker"}, {"text", private_marker()}};
        }
    }
}

// A rendered chat prompt with media markers, tokenized with the images and audio in their places. Template text may
// carry special tokens; user-supplied text must not be parsed for them.
int tokenize_chat(fornax_llama * h, const common_chat_prompt & prompt, const std::vector<mtmd_bitmap *> & bmps,
                  mtmd::input_chunks_ptr & chunks) {
    const std::string & marker = private_marker();
    struct seg { std::string text; bool special; int media; };  // media: the one that follows the text, or -1
    std::vector<seg> segs;
    int next = 0;
    for (const auto & part : prompt.parts) {
        size_t start = 0, at;
        while ((at = part.text.find(marker, start)) != std::string::npos) {
            segs.push_back({part.text.substr(start, at - start), !part.is_input, next++});
            start = at + marker.size();
        }
        segs.push_back({part.text.substr(start), !part.is_input, -1});
    }
    if (next != (int) bmps.size())
        return fail("the chat template placed " + std::to_string(next) + " of " + std::to_string(bmps.size()) + " media", 9);
    std::vector<mtmd_input_text> texts(segs.size());
    std::vector<mtmd_input_part> parts;
    for (size_t i = 0; i < segs.size(); i++) {
        texts[i] = {segs[i].text.data(), segs[i].text.size(), false, segs[i].special};
        if (!segs[i].text.empty()) parts.push_back({&texts[i], nullptr});
        if (segs[i].media >= 0) parts.push_back({nullptr, bmps[segs[i].media]});
    }
    std::vector<const mtmd_input_part *> ptrs;
    for (const auto & p : parts) ptrs.push_back(&p);
    chunks.reset(mtmd_input_chunks_init());
    if (int r = mtmd_tokenize_from_parts(h->mctx, chunks.get(), ptrs.data(), ptrs.size(), true); r != 0)
        return fail("tokenizing the prompt with media failed: " + std::to_string(r), 10);
    return 0;
}

// Runs a prompt with media through the model from position 0 (images and audio through their encoders, then
// everything through the language model), leaving the logits of its last token. *out_pos is the position after it, and
// *out_n_prompt the number of tokens (context cells) it took.
int eval_chunks(fornax_llama * h, const mtmd_input_chunks * chunks, int * out_pos, int * out_n_prompt) {
    const int n_ctx = (int) llama_n_ctx(h->ctx);
    const int n = (int) mtmd_helper_get_n_tokens(chunks);
    if (n >= n_ctx) return fail("prompt (" + std::to_string(n) + " tokens) exceeds context " + std::to_string(n_ctx), 2);
    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_pos new_pos = 0;
    if (int r = mtmd_helper_eval_chunks(h->mctx, h->ctx, chunks, 0, 0, (int) llama_n_batch(h->ctx), true, &new_pos); r != 0)
        return fail("evaluating the prompt and media failed: " + std::to_string(r), 8);
    *out_pos = (int) new_pos;
    *out_n_prompt = n;
    return 0;
}

// Messages formatted with the model's built-in chat template (fornax_llama_chat's), ending with the assistant's turn.
int template_prompt(fornax_llama * h, const char ** roles, const char ** contents, int n_messages, std::string & out) {
    std::vector<llama_chat_message> msgs(n_messages);
    for (int i = 0; i < n_messages; i++) msgs[i] = {roles[i], contents[i]};
    const char * tmpl = llama_model_chat_template(h->model, nullptr);
    int need = llama_chat_apply_template(tmpl, msgs.data(), n_messages, true, nullptr, 0);
    if (need < 0) return fail("failed to apply the chat template", 5);
    std::vector<char> buf(need + 1);
    llama_chat_apply_template(tmpl, msgs.data(), n_messages, true, buf.data(), (int) buf.size());
    out.assign(buf.data(), need);
    return 0;
}

// The template inputs of an OpenAI-style chat request: messages, tools, tool choice, thinking, a JSON schema, and
// variables for the template: chat_template_kwargs as llama-server takes them, and reasoning_effort (gpt-oss's
// low, medium or high), which is one of them.
common_chat_templates_inputs chat_inputs(const common_json & req) {
    common_chat_templates_inputs in;
    in.messages = common_chat_msgs_parse_oaicompat(req.at("messages"));
    if (req.contains("tools") && !req.at("tools").is_null()) in.tools = common_chat_tools_parse_oaicompat(req.at("tools"));
    if (req.contains("tool_choice") && req.at("tool_choice").is_string())
        in.tool_choice = common_chat_tool_choice_parse_oaicompat(req.at("tool_choice").get<std::string>());
    in.parallel_tool_calls = req.value("parallel_tool_calls", true);
    in.enable_thinking     = req.value("enable_thinking", true);
    if (req.contains("chat_template_kwargs") && req.at("chat_template_kwargs").is_object()) {
        for (const auto & [k, v] : req.at("chat_template_kwargs").items()) {
            in.chat_template_kwargs[k] = v.dump();
            if (k == "enable_thinking" && v.is_boolean() && !req.contains("enable_thinking")) in.enable_thinking = v.get<bool>();
        }
        // the template reads kwargs over in.enable_thinking: drop it, so the prompt and the parser agree on one value
        in.chat_template_kwargs.erase("enable_thinking");
    }
    if (req.contains("reasoning_effort") && req.at("reasoning_effort").is_string())
        in.chat_template_kwargs["reasoning_effort"] = req.at("reasoning_effort").dump();
    in.reasoning_format    = COMMON_REASONING_FORMAT_DEEPSEEK;  // thinking goes to reasoning_content
    if (req.contains("json_schema") && !req.at("json_schema").is_null()) in.json_schema = req.at("json_schema").dump();
    return in;
}

// A request's sampling options, with the session's grammar (tools, a JSON schema) and its triggers.
common_params_sampling chat_sampling(const common_json & req, const common_chat_session & session) {
    common_params_sampling sp;
    sp.temp  = (float) req.value("temperature", 0.7);
    sp.top_p = (float) req.value("top_p", 0.95);
    sp.min_p = (float) req.value("min_p", 0.05);
    sp.top_k = req.value("top_k", 40);
    int64_t seed = req.value("seed", (int64_t) -1);
    sp.seed = seed < 0 ? LLAMA_DEFAULT_SEED : (uint32_t) seed;
    session.apply_sampling(sp);
    return sp;
}

// A text-only prompt's tokens. Template text may carry special tokens; user-supplied text must not be parsed for them.
std::vector<llama_token> chat_tokens(const llama_vocab * vocab, const common_chat_session & session) {
    std::vector<llama_token> prompt;
    bool first = true;
    for (const auto & part : session.prompt().parts) {
        auto t = common_tokenize(vocab, part.text, first, !part.is_input);
        prompt.insert(prompt.end(), t.begin(), t.end());
        first = false;
    }
    return prompt;
}

// Feeds a generated token to the session and hands new message content (or "", when there is none: reasoning, a
// tool call) to emit, which returns non-zero to stop. True when the generation should stop: emit asked to, or the
// text ends with one of the template's extra stops.
template <class Emit>
bool chat_feed(const llama_vocab * vocab, common_chat_session & session, llama_token tok, std::string & raw,
               std::string & sent_content, Emit emit) {
    std::string piece = common_token_to_piece(vocab, tok, true);
    raw += piece;
    common_chat_input chunk;
    chunk.append(piece, tok);
    const common_chat_msg & m = session.feed(chunk);
    if (m.content.size() > sent_content.size() && m.content.compare(0, sent_content.size(), sent_content) == 0) {
        std::string delta = m.content.substr(sent_content.size());
        sent_content = m.content;
        if (emit(delta.c_str()) != 0) return true;
    } else if (emit("") != 0) {
        return true;
    }
    for (const auto & s : session.additional_stops())
        if (!s.empty() && raw.size() >= s.size() && raw.compare(raw.size() - s.size(), s.size(), s) == 0) return true;
    return false;
}

// The response JSON: the parsed message (content, reasoning, tool calls), why it ended, and the token counts.
std::string chat_response(common_chat_session & session, std::string finish, int n_prompt, int generated) {
    common_chat_msg msg = session.finish();
    msg.role = "assistant";
    for (size_t i = 0; i < msg.tool_calls.size(); i++)
        if (msg.tool_calls[i].id.empty()) msg.tool_calls[i].id = "call_" + std::to_string(i);
    if (!msg.tool_calls.empty() && finish == "stop") finish = "tool_calls";
    common_json res = common_json::object();
    res["message"] = msg.to_json_oaicompat();
    res["finish_reason"] = finish;
    common_json usage = common_json::object();
    usage["prompt_tokens"] = (int64_t) n_prompt;
    usage["completion_tokens"] = (int64_t) generated;
    res["usage"] = usage;
    return res.dump();
}

// fornax_llama_chat_json, with the images and audio that {"type": "image" | "audio", "index": i} parts refer to.
int chat_json_impl(fornax_llama * h, const char * request_json, const std::vector<mtmd_bitmap *> & images,
                   const std::vector<mtmd_bitmap *> & audio, fornax_llama_token_cb cb, void * user,
                   char ** out_response_json) {
    *out_response_json = nullptr;
    if (int rc = require_generative(h); rc != 0) return rc;
    common_sampler * smpl = nullptr;
    try {
        common_json req = common_json::parse(request_json);
        request_media media;
        if (req.contains("messages")) take_media(h, req.at("messages"), images, audio, media);
        const bool with_media = !media.all.empty();

        common_chat_session session(h->tmpls.get(), h->vocab, chat_inputs(req));

        std::vector<llama_token> prompt;
        mtmd::input_chunks_ptr chunks;
        int n_prompt = 0;
        const int n_ctx = (int) llama_n_ctx(h->ctx);
        if (with_media) {
            if (int rc = tokenize_chat(h, session.prompt(), media.all, chunks); rc != 0) return rc;
            n_prompt = (int) mtmd_helper_get_n_tokens(chunks.get());
        } else {
            prompt = chat_tokens(h->vocab, session);
            n_prompt = (int) prompt.size();
        }
        if (n_prompt >= n_ctx)
            return fail("prompt (" + std::to_string(n_prompt) + " tokens) exceeds context " + std::to_string(n_ctx), 2);

        common_params_sampling sp = chat_sampling(req, session);
        smpl = common_sampler_init(h->model, sp);
        if (!smpl) return fail("failed to initialise the sampler (bad grammar?)", 6);

        const int max_tokens = req.value("max_tokens", -1);
        llama_memory_clear(llama_get_memory(h->ctx), true);
        std::string finish = "stop", raw, sent_content;
        int pos = 0, generated = 0, cells = 0;  // cells: of the context used (more than pos after an M-RoPE image)
        std::vector<llama_token> cur = prompt;  // empty when the prompt went in with its media
        if (with_media) {
            if (int rc = eval_chunks(h, chunks.get(), &pos, &cells); rc != 0) {
                common_sampler_free(smpl);
                return rc;
            }
        }
        while (true) {
            if (max_tokens > 0 && generated >= max_tokens) { finish = "length"; break; }
            if (!cur.empty()) {
                if (cells + (int) cur.size() > n_ctx) { finish = "length"; break; }
                if (int r = decode(h, cur.data(), (int) cur.size(), pos); r != 0) {
                    common_sampler_free(smpl);
                    return fail("llama_process failed: " + std::to_string(r), 3);
                }
                pos += (int) cur.size();
                cells += (int) cur.size();
            }
            llama_token tok = common_sampler_sample(smpl, h->ctx, -1);
            common_sampler_accept(smpl, tok, true);
            if (llama_vocab_is_eog(h->vocab, tok)) break;
            generated++;
            // an empty piece (no new content: reasoning, a tool call) still lets the caller stop
            if (chat_feed(h->vocab, session, tok, raw, sent_content, [&](const char * p) { return cb ? cb(p, user) : 0; })) break;
            cur.assign(1, tok);
        }
        common_sampler_free(smpl);
        smpl = nullptr;

        *out_response_json = dup(chat_response(session, finish, n_prompt, generated));
        return 0;
    } catch (const std::exception & e) {
        if (smpl) common_sampler_free(smpl);
        return fail(std::string("chat: ") + e.what(), 7);
    }
}

// Process-wide setup (once), then the model's weights. The caller creates the context.
fornax_llama * load_model(const char * model_path, int n_gpu_layers) {
    static bool inited = false;
    if (!inited) {
        // Quiet by default: only errors, unless FORNAX_VERBOSE is set.
        if (!getenv("FORNAX_VERBOSE")) {
            llama_log_set([](ggml_log_level level, const char * text, void *) {
                if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
            }, nullptr);
        }
        ggml_backend_load_all();
        llama_backend_init();
        inited = true;
    }
    auto * h = new fornax_llama();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers < 0 ? 999 : n_gpu_layers;
    if (!getenv("FORNAX_VERBOSE")) mp.progress_callback = [](float, void *) { return true; };  // no loading dots
    h->model = llama_model_load_from_file(model_path, mp);
    if (!h->model) { g_err = std::string("unable to load model: ") + model_path; delete h; return nullptr; }
    h->vocab = llama_model_get_vocab(h->model);
    return h;
}

// The context a pooled handle starts with (or its largest, if smaller): enough for a query or a few short texts.
constexpr int POOLED_START = 512;

// A pooled handle's context of n tokens: as llama-embedding's, one unified KV cache over as many sequences as llama.cpp
// allows, and the whole batch in one ubatch (pooling needs every token of a sequence at once), so one input can be up
// to n tokens.
llama_context * pooled_context(llama_model * model, uint32_t n) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = n;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;  // the model's own (GGUF <arch>.pooling_type)
    cp.kv_unified   = true;
    cp.n_batch = cp.n_ubatch = cp.n_ctx;
    // no more sequences than outputs: llama.cpp sizes its output buffer for n_seq_max and asserts it fits n_batch
    cp.n_seq_max    = std::min((uint32_t) llama_max_parallel_sequences(), cp.n_batch);
    return llama_init_from_model(model, cp);
}

// Runs inputs through a pooled (embeddings or rerank) context, each as its own sequence, packing as many into one
// batch as fit (n_batch tokens, n_seq_max sequences), as llama-embedding does. Calls out(i, pooled) for input i, with
// the n_embd_out floats (or n_cls_out scores) llama.cpp pooled for it. Between batches, cb("") lets the caller stop.
int run_pooled(fornax_llama * h, const std::vector<std::vector<llama_token>> & inputs,
               const std::function<void(int, const float *)> & out, fornax_llama_token_cb cb, void * user) {
    // Grow the context to fit the call: its longest input, and as much of the whole call as the largest context
    // holds, so a call of many short texts still goes through in few batches. Rebuilding a context (not the weights)
    // takes well under 0.1 s (#109); it never shrinks, as the model has shown it gets inputs this size. The inputs are
    // checked first, so one that can't fit fails without growing the context for nothing.
    size_t longest = 0, total = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].empty()) return fail("input " + std::to_string(i) + " has no tokens", 15);
        if ((int) inputs[i].size() > h->pooled_max)
            return fail("input " + std::to_string(i) + " has " + std::to_string(inputs[i].size()) +
                        " tokens, more than the context of " + std::to_string(h->pooled_max) + " (raise the context size or split it)", 2);
        longest = std::max(longest, inputs[i].size());
        total += inputs[i].size();
    }
    const size_t want = std::max(longest, std::min(total, (size_t) h->pooled_max));
    if (want > llama_n_batch(h->ctx)) {  // want is at most pooled_max, so n stays under 2^32
        uint32_t n = llama_n_batch(h->ctx);
        while (n < want) n *= 2;
        n = std::min(n, (uint32_t) h->pooled_max);
        llama_context * bigger = pooled_context(h->model, n);
        if (!bigger) return fail("failed to grow the context to " + std::to_string(n) + " tokens", 5);
        llama_batch_ext_free(h->batch);
        llama_free(h->ctx);
        h->ctx = bigger;
        h->batch = llama_batch_ext_init(h->ctx);
    }
    const int n_batch = (int) llama_n_batch(h->ctx);
    const int n_seq_max = (int) llama_n_seq_max(h->ctx);
    size_t next = 0;
    while (next < inputs.size()) {
        const size_t first = next;
        int n_tokens = 0;
        llama_batch_ext_clear(h->batch);
        while (next < inputs.size() && (int) (next - first) < n_seq_max && n_tokens + (int) inputs[next].size() <= n_batch) {
            const llama_seq_id seq = (llama_seq_id) (next - first);
            const auto & toks = inputs[next];
            for (size_t j = 0; j < toks.size(); j++) {
                int idx = llama_batch_ext_add_token(h->batch, seq, toks[j]);
                if (idx < 0) return fail("llama_batch_ext_add_token failed: " + std::to_string(idx), 3);
                llama_pos p = (llama_pos) j;
                llama_batch_ext_set_pos(h->batch, idx, &p);
                llama_batch_ext_set_output_embd(h->batch, idx, true);
            }
            n_tokens += (int) toks.size();
            next++;
        }
        llama_memory_clear(llama_get_memory(h->ctx), true);  // each batch stands alone
        if (int r = llama_process(h->ctx, LLAMA_PROCESS_TYPE_DECODE, h->batch); r != 0)
            return fail("llama_process failed: " + std::to_string(r), 3);
        for (size_t i = first; i < next; i++) {
            const float * v = llama_get_embeddings_seq(h->ctx, (llama_seq_id) (i - first));
            if (!v) return fail("no pooled output for input " + std::to_string(i), 16);
            out((int) i, v);
        }
        if (cb && cb("", user) != 0) return fail("cancelled", 13);
    }
    return 0;
}


// ---- continuous batching ----

// Frees a slot's sampler and its cells in the context, ready for the next request.
void release_slot(fornax_llama * h, int seq) {
    fornax_llama_slot & s = h->slots[seq];
    if (s.chat_smpl) common_sampler_free(s.chat_smpl);
    if (s.raw_smpl) llama_sampler_free(s.raw_smpl);
    llama_memory_seq_rm(llama_get_memory(h->ctx), seq, -1, -1);
    s = fornax_llama_slot();
}

// Ends a slot's request with its response (a chat response, or for a raw prompt {"text", "finish_reason", "usage"}).
void finish_slot(fornax_llama * h, int seq, const std::string & finish, fornax_llama_done_cb done, void * user) {
    fornax_llama_slot & s = h->slots[seq];
    std::string res;
    int rc = 0;
    try {
        if (s.session) {
            res = chat_response(*s.session, finish, (int) s.prompt.size(), s.generated);
        } else {
            common_json r = common_json::object();
            r["text"] = s.text;
            r["finish_reason"] = finish;
            r["usage"] = {{"prompt_tokens", (int64_t) s.prompt.size()}, {"completion_tokens", (int64_t) s.generated}};
            res = r.dump();
        }
    } catch (const std::exception & e) {
        rc = fail(std::string("chat: ") + e.what(), 7);
        res = g_err;
    }
    release_slot(h, seq);
    if (done) done(seq, rc, res.c_str(), user);
}

void fail_slot(fornax_llama * h, int seq, int rc, fornax_llama_done_cb done, void * user) {
    std::string e = g_err;
    release_slot(h, seq);
    if (done) done(seq, rc, e.c_str(), user);
}

// A free slot for a new request, or minus the error code (with the error set).
int free_slot(fornax_llama * h) {
    if (h->slots.empty()) return -fail("this model wasn't loaded with slots (fornax_llama_create_batched)", 16);
    for (size_t i = 0; i < h->slots.size(); i++) if (!h->slots[i].busy) return (int) i;
    return -fail("every slot is busy", 15);
}

// Takes a tokenized request into slot seq, or fails (and frees it) when the prompt doesn't fit a slot's context.
int start_slot(fornax_llama * h, int seq, fornax_llama_slot && s) {
    const int n_ctx_seq = (int) llama_n_ctx_seq(h->ctx);
    if ((int) s.prompt.size() >= n_ctx_seq) {
        if (s.chat_smpl) common_sampler_free(s.chat_smpl);
        if (s.raw_smpl) llama_sampler_free(s.raw_smpl);
        return fail("prompt (" + std::to_string(s.prompt.size()) + " tokens) exceeds a slot's context " + std::to_string(n_ctx_seq), 2);
    }
    if (s.prompt.empty()) {
        if (s.chat_smpl) common_sampler_free(s.chat_smpl);
        if (s.raw_smpl) llama_sampler_free(s.raw_smpl);
        return fail("empty prompt", 1);
    }
    s.busy = true;
    h->slots[seq] = std::move(s);
    return 0;
}
}  // namespace

extern "C" {

const char * fornax_llama_last_error(void) { return g_err.c_str(); }

fornax_llama * fornax_llama_create(const char * model_path, int n_gpu_layers, int n_ctx) {
    auto * h = load_model(model_path, n_gpu_layers);
    if (!h) return nullptr;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx > 0 ? n_ctx : 0;
    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) { g_err = "failed to create llama_context"; fornax_llama_destroy(h); return nullptr; }
    h->batch = llama_batch_ext_init(h->ctx);
    try {
        h->tmpls = common_chat_templates_init(h->model, "");
    } catch (const std::exception & e) {
        g_err = std::string("chat template: ") + e.what();
        fornax_llama_destroy(h);
        return nullptr;
    }
    return h;
}

int fornax_llama_complete(fornax_llama * h, const char * prompt, int max_tokens, float temperature, float top_p,
                        float min_p, uint32_t seed, fornax_llama_token_cb cb, void * user, char ** out_text,
                        int * out_n_prompt, int * out_n_generated) {
    *out_text = nullptr;
    if (int rc = require_generative(h); rc != 0) return rc;
    llama_memory_clear(llama_get_memory(h->ctx), true);

    const int len = (int) strlen(prompt);
    int n = -llama_tokenize(h->vocab, prompt, len, nullptr, 0, true, true);
    std::vector<llama_token> toks(n);
    if (llama_tokenize(h->vocab, prompt, len, toks.data(), n, true, true) < 0) return fail("tokenize failed", 1);
    const int n_ctx = (int) llama_n_ctx(h->ctx);
    if (n >= n_ctx) return fail("prompt (" + std::to_string(n) + " tokens) exceeds context " + std::to_string(n_ctx), 2);
    if (out_n_prompt) *out_n_prompt = n;

    llama_sampler * smpl = simple_sampler(temperature, top_p, min_p, seed);

    std::string out;
    int pos = 0, generated = 0, rc = 0;
    const llama_token * cur = toks.data();
    int n_cur = n;
    llama_token tok;
    while (max_tokens <= 0 || generated < max_tokens) {
        if (pos + n_cur > n_ctx) break;  // context full: return what we have
        if (int r = decode(h, cur, n_cur, pos); r != 0) {
            rc = fail("llama_process failed: " + std::to_string(r), 3);
            break;
        }
        pos += n_cur;
        tok = llama_sampler_sample(smpl, h->ctx, -1);
        if (llama_vocab_is_eog(h->vocab, tok)) break;
        char buf[256];
        int m = llama_token_to_piece(h->vocab, tok, buf, sizeof buf, 0, true);
        if (m < 0) { rc = fail("token_to_piece failed", 4); break; }
        std::string piece(buf, m);
        out += piece;
        generated++;
        if (cb && cb(piece.c_str(), user) != 0) break;
        cur = &tok;
        n_cur = 1;
    }
    llama_sampler_free(smpl);
    if (out_n_generated) *out_n_generated = generated;
    *out_text = dup(out);
    return rc;
}

int fornax_llama_chat(fornax_llama * h, const char ** roles, const char ** contents, int n_messages,
                    int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                    fornax_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated) {
    std::string prompt;
    if (int rc = template_prompt(h, roles, contents, n_messages, prompt); rc != 0) return rc;
    return fornax_llama_complete(h, prompt.c_str(), max_tokens, temperature, top_p, min_p, seed, cb, user, out_text,
                               out_n_prompt, out_n_generated);
}

int fornax_llama_chat_json(fornax_llama * h, const char * request_json, fornax_llama_token_cb cb, void * user,
                         char ** out_response_json) {
    return chat_json_impl(h, request_json, {}, {}, cb, user, out_response_json);
}

fornax_llama * fornax_llama_create_mm(const char * model_path, const char * mmproj_path, int n_gpu_layers, int n_ctx) {
    fornax_llama * h = fornax_llama_create(model_path, n_gpu_layers, n_ctx);
    if (!h) return nullptr;
    if (!getenv("FORNAX_VERBOSE")) {
        mtmd_helper_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
        }, nullptr);
    }
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu       = n_gpu_layers != 0;
    mp.print_timings = getenv("FORNAX_VERBOSE") != nullptr;
    mp.warmup        = false;
    try {
        // As llama-mtmd-cli does: a model whose image attention is non-causal needs the whole image in one ubatch.
        const int n_ubatch = (int) llama_n_ubatch(h->ctx);
        auto mem = mtmd_get_memory_usage(mmproj_path, mp);
        if (mem.use_non_causal && mem.image_max_tokens > n_ubatch) {
            mp.image_max_tokens = n_ubatch;
            mp.image_min_tokens = std::min(mp.image_min_tokens, n_ubatch);
        }
    } catch (const std::exception &) {}  // mtmd_init_from_file reports a bad mmproj below
    h->mctx = mtmd_init_from_file(mmproj_path, h->model, mp);
    if (!h->mctx) { g_err = std::string("unable to load mmproj: ") + mmproj_path; fornax_llama_destroy(h); return nullptr; }
    return h;
}

int fornax_llama_supports_vision(fornax_llama * h) { return h->mctx && mtmd_support_vision(h->mctx) ? 1 : 0; }

int fornax_llama_supports_audio(fornax_llama * h) { return h->mctx && mtmd_support_audio(h->mctx) ? 1 : 0; }

int fornax_llama_audio_sample_rate(fornax_llama * h) { return h->mctx ? mtmd_get_audio_sample_rate(h->mctx) : -1; }

const char * fornax_llama_media_marker(void) { return mtmd_default_marker(); }

int fornax_llama_chat_images(fornax_llama * h, const char ** roles, const char ** contents, const int * image_counts,
                           int n_messages, const unsigned char ** rgb, const uint32_t * widths, const uint32_t * heights,
                           int n_images, int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                           fornax_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated) {
    *out_text = nullptr;
    char * res = nullptr;
    int rc;
    try {
        auto bmps = rgb_bitmaps(rgb, widths, heights, n_images);
        common_json msgs = common_json::array();
        int next = 0;
        for (int i = 0; i < n_messages; i++) {
            int k = image_counts ? image_counts[i] : 0;
            if (k <= 0) { msgs.push_back({{"role", roles[i]}, {"content", contents[i]}}); continue; }
            common_json parts = common_json::array();
            for (int j = 0; j < k; j++) parts.push_back({{"type", "image"}, {"index", next++}});
            if (contents[i] && *contents[i]) parts.push_back({{"type", "text"}, {"text", contents[i]}});
            msgs.push_back({{"role", roles[i]}, {"content", parts}});
        }
        if (next != n_images) return fail(std::to_string(n_images) + " images given, messages use " + std::to_string(next), 11);
        // the same sampling as fornax_llama_chat: no top-k; temperature <= 0 is greedy
        common_json req = {{"messages", msgs}, {"max_tokens", max_tokens}, {"temperature", temperature > 0 ? temperature : 0.0f},
                           {"top_p", top_p}, {"min_p", min_p}, {"top_k", 0},
                           {"seed", seed == 0xFFFFFFFFu ? (int64_t) -1 : (int64_t) seed}};
        rc = chat_json_impl(h, req.dump().c_str(), raw(bmps), {}, cb, user, &res);
        if (rc != 0) return rc;
        std::string json(res);
        free(res);
        common_json out = common_json::parse(json);
        const auto & content = out.at("message").at("content");
        *out_text = dup(content.is_string() ? content.get<std::string>() : "");
        if (out_n_prompt) *out_n_prompt = out.at("usage").at("prompt_tokens").get<int>();
        if (out_n_generated) *out_n_generated = out.at("usage").at("completion_tokens").get<int>();
        return 0;
    } catch (const std::exception & e) {
        return fail(std::string("chat: ") + e.what(), 7);
    }
}

int fornax_llama_complete_images(fornax_llama * h, const char * prompt, const unsigned char ** rgb, const uint32_t * widths,
                               const uint32_t * heights, int n_images, int max_tokens, float temperature, float top_p,
                               float min_p, uint32_t seed, fornax_llama_token_cb cb, void * user, char ** out_text,
                               int * out_n_prompt, int * out_n_generated) {
    *out_text = nullptr;
    if (int rc = require_generative(h); rc != 0) return rc;
    if (!h->mctx || !mtmd_support_vision(h->mctx))
        return fail(h->mctx ? "this model takes no images (its mmproj has no vision encoder)"
                            : "this model takes no images (load it with its mmproj)", 12);
    std::vector<mtmd::bitmap_ptr> bmps;
    try { bmps = rgb_bitmaps(rgb, widths, heights, n_images); }
    catch (const std::exception & e) { return fail(e.what(), 11); }
    // Counted here: mtmd_tokenize reports a marker mismatch as 2, the same as a failed image, despite its header.
    const std::string marker = mtmd_default_marker(), text_in = prompt;
    int n_markers = 0;
    for (size_t at = text_in.find(marker); at != std::string::npos; at = text_in.find(marker, at + marker.size())) n_markers++;
    if (n_markers != n_images)
        return fail("the prompt has " + std::to_string(n_markers) + " image markers for " + std::to_string(n_images) + " images", 9);
    auto ptrs = raw(bmps);
    std::vector<const mtmd_bitmap *> cptrs(ptrs.begin(), ptrs.end());
    mtmd_input_text text = {prompt, strlen(prompt), true, true};
    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    int r = mtmd_tokenize(h->mctx, chunks.get(), &text, cptrs.data(), cptrs.size());
    if (r != 0) return fail("tokenizing the prompt with images failed: " + std::to_string(r), 10);
    int pos = 0, n_prompt = 0;
    if (int rc = eval_chunks(h, chunks.get(), &pos, &n_prompt); rc != 0) return rc;
    if (out_n_prompt) *out_n_prompt = n_prompt;
    llama_sampler * smpl = simple_sampler(temperature, top_p, min_p, seed);
    std::string out;
    int generated = 0;
    int rc = generate(h, smpl, pos, n_prompt, max_tokens, cb, user, out, generated);
    llama_sampler_free(smpl);
    if (out_n_generated) *out_n_generated = generated;
    *out_text = dup(out);
    return rc;
}

int fornax_llama_chat_json_images(fornax_llama * h, const char * request_json, const unsigned char ** rgb,
                                const uint32_t * widths, const uint32_t * heights, int n_images,
                                fornax_llama_token_cb cb, void * user, char ** out_response_json) {
    *out_response_json = nullptr;
    std::vector<mtmd::bitmap_ptr> bmps;
    try { bmps = rgb_bitmaps(rgb, widths, heights, n_images); }
    catch (const std::exception & e) { return fail(e.what(), 11); }
    return chat_json_impl(h, request_json, raw(bmps), {}, cb, user, out_response_json);
}

int fornax_llama_chat_json_media(fornax_llama * h, const char * request_json, const unsigned char ** rgb,
                               const uint32_t * widths, const uint32_t * heights, int n_images, const float ** pcm,
                               const uint32_t * n_samples, int n_audio, fornax_llama_token_cb cb, void * user,
                               char ** out_response_json) {
    *out_response_json = nullptr;
    std::vector<mtmd::bitmap_ptr> imgs, sounds;
    try {
        imgs = rgb_bitmaps(rgb, widths, heights, n_images);
        sounds = pcm_bitmaps(pcm, n_samples, n_audio);
    } catch (const std::exception & e) { return fail(e.what(), 11); }
    return chat_json_impl(h, request_json, raw(imgs), raw(sounds), cb, user, out_response_json);
}

fornax_llama * fornax_llama_create_pooled(const char * model_path, int n_gpu_layers, int n_ctx, int rerank) {
    auto * h = load_model(model_path, n_gpu_layers);
    if (!h) return nullptr;
    // n_ctx is the largest the context grows to; it starts at POOLED_START (or n_ctx, if smaller) and grows to fit
    // what it's given (run_pooled), so short inputs don't reserve the compute buffer of long ones.
    h->pooled_max = n_ctx > 0 ? n_ctx : (int) llama_model_n_ctx_train(h->model);
    h->ctx = pooled_context(h->model, (uint32_t) std::min(h->pooled_max, POOLED_START));
    if (!h->ctx) { g_err = "failed to create llama_context"; fornax_llama_destroy(h); return nullptr; }
    h->pooling = llama_pooling_type(h->ctx);
    if (rerank && h->pooling != LLAMA_POOLING_TYPE_RANK) {
        g_err = "not a reranker: its GGUF has no rank pooling (a classifier head)";
        fornax_llama_destroy(h);
        return nullptr;
    }
    if (!rerank && h->pooling == LLAMA_POOLING_TYPE_RANK) {
        g_err = "this is a reranker: load it to rerank, not to embed";
        fornax_llama_destroy(h);
        return nullptr;
    }
    if (!rerank && h->pooling == LLAMA_POOLING_TYPE_NONE) {
        g_err = "not an embedding model: its GGUF sets no pooling type";
        fornax_llama_destroy(h);
        return nullptr;
    }
    h->batch = llama_batch_ext_init(h->ctx);
    return h;
}

int fornax_llama_n_embd(fornax_llama * h) { return llama_model_n_embd_out(h->model); }

int fornax_llama_embed(fornax_llama * h, const char ** texts, int n, int normalize, float * out, int * out_tokens,
                     fornax_llama_token_cb cb, void * user) {
    if (h->pooling == LLAMA_POOLING_TYPE_NONE) return fail("this model was not loaded for embeddings", 14);
    if (h->pooling == LLAMA_POOLING_TYPE_RANK) return fail("this model was loaded as a reranker: it can only rerank", 14);
    try {
        // As llama-embedding: the vocab's own BOS/EOS added (Qwen3-Embedding pools the EOS it appends). Unlike it,
        // the text isn't parsed for special tokens: it is the caller's data, not markup.
        std::vector<std::vector<llama_token>> inputs;
        for (int i = 0; i < n; i++) inputs.push_back(common_tokenize(h->vocab, texts[i], true, false));
        if (out_tokens) for (int i = 0; i < n; i++) out_tokens[i] = (int) inputs[i].size();
        const int dim = llama_model_n_embd_out(h->model);
        return run_pooled(h, inputs, [&](int i, const float * v) {
            common_embd_normalize(v, out + (size_t) i * dim, dim, normalize ? 2 : -1);
        }, cb, user);
    } catch (const std::exception & e) {
        return fail(std::string("embed: ") + e.what(), 7);
    }
}

int fornax_llama_rerank(fornax_llama * h, const char * query, const char ** docs, int n, float * out_scores,
                      fornax_llama_token_cb cb, void * user) {
    if (h->pooling != LLAMA_POOLING_TYPE_RANK) return fail("this model was not loaded as a reranker", 14);
    try {
        // The pair's prompt as llama-server's /rerank builds it (format_prompt_rerank): the GGUF's "rerank" template
        // with {query} and {document} filled in, tokenized whole with special tokens parsed; with no template,
        // [BOS] query [EOS] [SEP] doc [EOS] by what the vocab adds.
        const char * tmpl = llama_model_chat_template(h->model, "rerank");
        std::vector<std::vector<llama_token>> inputs;
        for (int i = 0; i < n; i++) {
            if (tmpl) {
                std::string t = tmpl, prompt;
                for (size_t p = 0; p < t.size();) {
                    if (t.compare(p, 7, "{query}") == 0) { prompt += query; p += 7; }
                    else if (t.compare(p, 10, "{document}") == 0) { prompt += docs[i]; p += 10; }
                    else prompt += t[p++];
                }
                inputs.push_back(common_tokenize(h->vocab, prompt, false, true));
            } else {
                llama_token eos = llama_vocab_eos(h->vocab);
                if (eos == LLAMA_TOKEN_NULL) eos = llama_vocab_sep(h->vocab);
                std::vector<llama_token> t;
                if (llama_vocab_get_add_bos(h->vocab)) t.push_back(llama_vocab_bos(h->vocab));
                auto q = common_tokenize(h->vocab, query, false, false), d = common_tokenize(h->vocab, docs[i], false, false);
                t.insert(t.end(), q.begin(), q.end());
                if (llama_vocab_get_add_eos(h->vocab)) t.push_back(eos);
                if (llama_vocab_get_add_sep(h->vocab)) t.push_back(llama_vocab_sep(h->vocab));
                t.insert(t.end(), d.begin(), d.end());
                if (llama_vocab_get_add_eos(h->vocab)) t.push_back(eos);
                inputs.push_back(std::move(t));
            }
        }
        // llama-server's score: the first class output (for Qwen3-Reranker, the softmax probability of "yes")
        return run_pooled(h, inputs, [&](int i, const float * v) { out_scores[i] = v[0]; }, cb, user);
    } catch (const std::exception & e) {
        return fail(std::string("rerank: ") + e.what(), 7);
    }
}

fornax_llama * fornax_llama_create_batched(const char * model_path, int n_gpu_layers, int n_ctx_slot, int n_slots) {
    if (n_slots < 1 || n_slots > (int) llama_max_parallel_sequences()) {
        g_err = "slots must be 1 to " + std::to_string(llama_max_parallel_sequences());
        return nullptr;
    }
    if (n_ctx_slot <= 0) { g_err = "a batched model needs a context size for each slot"; return nullptr; }
    auto * h = load_model(model_path, n_gpu_layers);
    if (!h) return nullptr;
    llama_context_params cp = llama_context_default_params();
    // One KV stream per sequence (not unified), so each slot has n_ctx_slot cells of its own (llama_n_ctx_seq) and
    // can never take another's: a slot that fills its context ends with "length", whatever the others are doing.
    cp.n_ctx      = (uint32_t) n_ctx_slot * (uint32_t) n_slots;
    cp.n_seq_max  = (uint32_t) n_slots;
    cp.kv_unified = false;
    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) { g_err = "failed to create llama_context"; fornax_llama_destroy(h); return nullptr; }
    h->batch = llama_batch_ext_init(h->ctx);
    try {
        h->tmpls = common_chat_templates_init(h->model, "");
    } catch (const std::exception & e) {
        g_err = std::string("chat template: ") + e.what();
        fornax_llama_destroy(h);
        return nullptr;
    }
    h->slots.resize(n_slots);
    return h;
}

int fornax_llama_n_slots(fornax_llama * h) { return (int) h->slots.size(); }

int fornax_llama_n_ctx_slot(fornax_llama * h) { return h->slots.empty() ? (int) llama_n_ctx(h->ctx) : (int) llama_n_ctx_seq(h->ctx); }

int fornax_llama_batch_add_json(fornax_llama * h, const char * request_json, int * out_seq) {
    *out_seq = -1;
    const int seq = free_slot(h);
    if (seq < 0) return -seq;
    fornax_llama_slot s;
    try {
        common_json req = common_json::parse(request_json);
        s.session = std::make_unique<common_chat_session>(h->tmpls.get(), h->vocab, chat_inputs(req));
        s.prompt = chat_tokens(h->vocab, *s.session);
        s.max_tokens = req.value("max_tokens", -1);  // before the sampler, which nothing frees if this throws
        common_params_sampling sp = chat_sampling(req, *s.session);
        s.chat_smpl = common_sampler_init(h->model, sp);
        if (!s.chat_smpl) return fail("failed to initialise the sampler (bad grammar?)", 6);
    } catch (const std::exception & e) {
        return fail(std::string("chat: ") + e.what(), 7);
    }
    if (int rc = start_slot(h, seq, std::move(s)); rc != 0) return rc;
    *out_seq = seq;
    return 0;
}

int fornax_llama_batch_add_raw(fornax_llama * h, const char * prompt, int max_tokens, float temperature, float top_p,
                               float min_p, uint32_t seed, int * out_seq) {
    *out_seq = -1;
    const int seq = free_slot(h);
    if (seq < 0) return -seq;
    fornax_llama_slot s;
    const int len = (int) strlen(prompt);
    int n = -llama_tokenize(h->vocab, prompt, len, nullptr, 0, true, true);
    s.prompt.resize(n);
    if (n > 0 && llama_tokenize(h->vocab, prompt, len, s.prompt.data(), n, true, true) < 0) return fail("tokenize failed", 1);
    s.raw_smpl = simple_sampler(temperature, top_p, min_p, seed);
    s.max_tokens = max_tokens;
    if (int rc = start_slot(h, seq, std::move(s)); rc != 0) return rc;
    *out_seq = seq;
    return 0;
}

int fornax_llama_batch_add_chat(fornax_llama * h, const char ** roles, const char ** contents, int n_messages,
                                int max_tokens, float temperature, float top_p, float min_p, uint32_t seed, int * out_seq) {
    *out_seq = -1;
    std::string prompt;
    if (int rc = template_prompt(h, roles, contents, n_messages, prompt); rc != 0) return rc;
    return fornax_llama_batch_add_raw(h, prompt.c_str(), max_tokens, temperature, top_p, min_p, seed, out_seq);
}

void fornax_llama_batch_cancel(fornax_llama * h, int seq) {
    if (seq >= 0 && seq < (int) h->slots.size() && h->slots[seq].busy) h->slots[seq].cancelled = true;
}

int fornax_llama_batch_step(fornax_llama * h, fornax_llama_piece_cb on_piece, fornax_llama_done_cb on_done, void * user) {
    if (h->slots.empty()) return fail("this model wasn't loaded with slots (fornax_llama_create_batched)", 16);
    const int n_slots = (int) h->slots.size();
    const int n_ctx_seq = (int) llama_n_ctx_seq(h->ctx);
    const int n_batch = (int) llama_n_batch(h->ctx);
    // Cancelled requests end with what they have, as a serial call stopped by its callback does.
    for (int i = 0; i < n_slots; i++)
        if (h->slots[i].busy && h->slots[i].cancelled) finish_slot(h, i, "stop", on_done, user);

    llama_batch_ext_clear(h->batch);
    int n = 0;
    // Running generations first, one token each, so a long prompt arriving never stalls them...
    for (int i = 0; i < n_slots; i++) {
        fornax_llama_slot & s = h->slots[i];
        if (!s.busy || s.next == LLAMA_TOKEN_NULL) continue;
        if ((s.max_tokens > 0 && s.generated >= s.max_tokens) || s.pos + 1 > n_ctx_seq) {
            finish_slot(h, i, "length", on_done, user);
            continue;
        }
        int idx = llama_batch_ext_add_token(h->batch, i, s.next);
        llama_pos p = s.pos;
        llama_batch_ext_set_pos(h->batch, idx, &p);
        llama_batch_ext_set_output_logits(h->batch, idx, true);
        s.i_batch = idx;
        s.n_adding = 1;
        n++;
    }
    // ...then prompts, in chunks, in what is left of the batch. A prompt gets its logits once its last chunk is in.
    for (int i = 0; i < n_slots && n < n_batch; i++) {
        fornax_llama_slot & s = h->slots[i];
        if (!s.busy || s.fed >= s.prompt.size()) continue;
        int take = std::min((int) (s.prompt.size() - s.fed), n_batch - n);
        int idx = -1;
        for (int k = 0; k < take; k++) {
            idx = llama_batch_ext_add_token(h->batch, i, s.prompt[s.fed + k]);
            llama_pos p = s.pos + k;
            llama_batch_ext_set_pos(h->batch, idx, &p);
        }
        if (s.fed + take == s.prompt.size()) {
            llama_batch_ext_set_output_logits(h->batch, idx, true);
            s.i_batch = idx;
        }
        s.n_adding = take;
        n += take;
    }
    if (n > 0) {
        if (int r = llama_process(h->ctx, LLAMA_PROCESS_TYPE_DECODE, h->batch); r != 0) {
            // The batch's requests fail; the others (none in this batch) carry on.
            const std::string e = "llama_process failed: " + std::to_string(r);
            for (int i = 0; i < n_slots; i++)
                if (h->slots[i].busy && h->slots[i].n_adding > 0) { fail(e, 3); fail_slot(h, i, 3, on_done, user); }
            return -fail(e, 3);
        }
    }
    for (int i = 0; i < n_slots; i++) {
        fornax_llama_slot & s = h->slots[i];
        if (!s.busy || s.n_adding == 0) continue;
        s.pos += s.n_adding;
        if (s.next == LLAMA_TOKEN_NULL) s.fed += s.n_adding;  // a prompt chunk, not a generated token
        s.n_adding = 0;
        if (s.i_batch < 0) continue;  // more prompt to go
        const int idx = s.i_batch;
        s.i_batch = -1;
        // As chat_json_impl does, a request whose grammar rejects a token, or whose output the chat parser can't
        // read, fails alone: an exception must not cross the C boundary into the JVM.
        try {
            llama_token tok;
            if (s.session) {
                tok = common_sampler_sample(s.chat_smpl, h->ctx, idx);
                common_sampler_accept(s.chat_smpl, tok, true);
            } else {
                tok = llama_sampler_sample(s.raw_smpl, h->ctx, idx);
            }
            if (llama_vocab_is_eog(h->vocab, tok)) { finish_slot(h, i, "stop", on_done, user); continue; }
            s.generated++;
            auto emit = [&](const char * p) { return on_piece ? on_piece(i, p, user) : 0; };
            bool stop;
            if (s.session) {
                stop = chat_feed(h->vocab, *s.session, tok, s.raw, s.sent_content, emit);
            } else {
                std::string piece = common_token_to_piece(h->vocab, tok, true);
                s.text += piece;
                stop = emit(piece.c_str()) != 0;
            }
            if (stop) { finish_slot(h, i, "stop", on_done, user); continue; }
            s.next = tok;
        } catch (const std::exception & e) {
            fail_slot(h, i, fail(std::string("chat: ") + e.what(), 7), on_done, user);
        }
    }
    int busy = 0;
    for (const auto & s : h->slots) busy += s.busy ? 1 : 0;
    return busy;
}

int64_t fornax_llama_kv_bytes(const char * model_path, int n_ctx) {
    gguf_init_params gp = {/*no_alloc*/ true, /*ctx*/ nullptr};
    gguf_context * g = gguf_init_from_file(model_path, gp);
    if (!g) { g_err = std::string("can't read the GGUF header of ") + model_path; return -1; }
    auto u32 = [&](const std::string & key, int64_t dflt) -> int64_t {
        int64_t id = gguf_find_key(g, key.c_str());
        if (id < 0) return dflt;
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
            case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
            default: return dflt;   // per-layer arrays (some hybrid models): take the default
        }
    };
    int64_t arch_id = gguf_find_key(g, "general.architecture");
    std::string a = arch_id >= 0 ? gguf_get_val_str(g, arch_id) : "llama";
    const int64_t n_layer = u32(a + ".block_count", 0), n_embd = u32(a + ".embedding_length", 0);
    const int64_t n_head = u32(a + ".attention.head_count", 1), n_head_kv = u32(a + ".attention.head_count_kv", n_head);
    const int64_t head = n_head > 0 ? n_embd / n_head : 0;
    const int64_t k_len = u32(a + ".attention.key_length", head), v_len = u32(a + ".attention.value_length", head);
    gguf_free(g);
    return n_layer * (int64_t) n_ctx * n_head_kv * (k_len + v_len) * 2;   // F16 keys and values, as llama.cpp's default
}

void fornax_llama_free(void * p) { free(p); }

void fornax_llama_destroy(fornax_llama * h) {
    if (!h) return;
    for (int i = 0; i < (int) h->slots.size(); i++)
        if (h->slots[i].busy) release_slot(h, i);
    h->tmpls.reset();
    if (h->mctx) mtmd_free(h->mctx);
    if (h->batch) llama_batch_ext_free(h->batch);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

}  // extern "C"
