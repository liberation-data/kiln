package org.liberationdata.fornax.jitllm;

import org.liberationdata.fornax.Json;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

/**
 * {@link JitLlmModel#chatJson}'s request and response, in the shape {@code LlamaModel.chatJson} takes and gives, so
 * fornax-server serves either engine. Needs no jitllm classes.
 */
final class ChatJson {
    private ChatJson() {}

    /** Return code 7 is what fornax-server reports as a 400 (bad input), as it does for llama.cpp's. */
    static IllegalStateException badRequest(String message) {
        return new IllegalStateException("jitllm (rc=7): " + message);
    }

    /**
     * A chat: the whole conversation and its sampling. maxTokens {@link Integer#MAX_VALUE} for no limit but the
     * context; seed below 0 for a random one; topP 0 for the engine's own; thinking null for the model's default.
     */
    record Request(List<Message> messages, int maxTokens, float temperature, float topP, long seed, List<String> stop,
                   Boolean thinking) {
        Request {
            messages = List.copyOf(messages);
            stop = List.copyOf(stop);
            if (messages.isEmpty()) throw badRequest("messages is empty");
            if (maxTokens < 1) throw badRequest("max_tokens must be at least 1");
        }

        /** maxTokens 0 (or less) generates until the model stops or the context fills, as LlamaModel does. */
        static Request of(List<Message> messages, Options o) {
            return new Request(messages, o.maxTokens() > 0 ? o.maxTokens() : Integer.MAX_VALUE, o.temperature(), o.topP(),
                    o.seed(), List.of(), null);
        }
    }

    /** What the engine gave. finishReason: OpenAI's "stop" or "length". opensInside: see {@link #splitThinking}. */
    record Reply(String text, int promptTokens, int generatedTokens, String finishReason, boolean opensInside) {
        Result result() { return new Result(text, promptTokens, generatedTokens); }
    }

    private static final Set<String> ROLES = Set.of("system", "developer", "user", "assistant");

    /**
     * An OpenAI chat completions request's messages and sampling, as fornax-server passes it on; what jitllm can't do
     * is refused. {@code min_p} and {@code top_k} are ignored: jitllm samples with temperature and top-p only. Without
     * {@code max_tokens} a reply runs to the end of its turn or a full context, as with llama.cpp.
     */
    static Request parse(String json) {
        Object doc;
        try { doc = Json.parse(json); }
        catch (RuntimeException e) { throw badRequest("the request isn't JSON: " + e.getMessage()); }
        if (!(doc instanceof Map<?, ?> req)) throw badRequest("the request isn't a JSON object");
        for (String unsupported : List.of("json_schema", "grammar"))
            if (req.get(unsupported) != null) throw badRequest(unsupported + " isn't supported by the jitllm engine");
        boolean noTools = "none".equals(req.get("tool_choice"));   // tools sent along, but none may be called
        if (!noTools && req.get("tools") instanceof List<?> tools && !tools.isEmpty())
            throw badRequest("tools aren't supported by the jitllm engine");
        if (req.get("response_format") instanceof Map<?, ?> rf && !"text".equals(rf.get("type")))
            throw badRequest("response_format " + rf.get("type") + " isn't supported by the jitllm engine");
        if (!(req.get("messages") instanceof List<?> list)) throw badRequest("messages must be an array");
        List<Message> messages = new ArrayList<>(list.size());
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> m) || !(m.get("role") instanceof String role))
                throw badRequest("each message needs a role");
            if (!ROLES.contains(role.toLowerCase(Locale.ROOT)))
                throw badRequest("the jitllm engine takes system, developer, user and assistant messages, not " + role);
            if (m.get("tool_calls") instanceof List<?> calls && !calls.isEmpty())
                throw badRequest("tool calls aren't supported by the jitllm engine");
            messages.add(new Message(role, content(m.get("content"))));
        }
        Object max = req.get("max_completion_tokens") != null ? req.get("max_completion_tokens") : req.get("max_tokens");
        Object thinking = req.get("enable_thinking");
        return new Request(messages,
                max == null ? Integer.MAX_VALUE : clamp(number(max, "max_tokens")),
                req.get("temperature") == null ? Options.DEFAULT.temperature() : number(req.get("temperature"), "temperature").floatValue(),
                req.get("top_p") == null ? 0f : number(req.get("top_p"), "top_p").floatValue(),
                req.get("seed") == null ? -1 : number(req.get("seed"), "seed").longValue(),
                stops(req.get("stop")),
                thinking instanceof Boolean b ? b : null);
    }

    /**
     * The response, as LlamaModel.chatJson gives it. A reply's reasoning, a leading {@code <think>} block (or, for a
     * model whose prompt opens it, the text before its {@code </think>}), is {@code reasoning_content}, as llama.cpp's
     * parser gives it.
     */
    static String response(Reply r) {
        String[] parts = splitThinking(r.text(), r.opensInside());
        StringBuilder b = new StringBuilder("{\"message\": {\"role\": \"assistant\", \"content\": ").append(Json.quote(parts[1]));
        if (parts[0] != null) b.append(", \"reasoning_content\": ").append(Json.quote(parts[0]));
        return b.append("}, \"finish_reason\": ").append(Json.quote(r.finishReason()))
                .append(", \"usage\": {\"prompt_tokens\": ").append(r.promptTokens())
                .append(", \"completion_tokens\": ").append(r.generatedTokens()).append("}}").toString();
    }

    private static final String OPEN = "<think>", CLOSE = "</think>";

    /**
     * {reasoning or null, content}. A reply that starts with {@code <think>} has its block as reasoning. With
     * {@code opensInside} (DeepSeek-R1 distills, whose prompt has the opening {@code <think>}) the reply starts inside
     * the reasoning, which ends at its {@code </think>}. Any other {@code </think>} is just text.
     */
    static String[] splitThinking(String text, boolean opensInside) {
        String t = text.stripLeading();
        boolean opened = t.startsWith(OPEN);
        if (!opened && !opensInside) return new String[]{null, text};
        int from = opened ? OPEN.length() : 0;
        int end = t.indexOf(CLOSE, from);
        if (end < 0) return new String[]{t.substring(from).strip(), ""};
        return new String[]{t.substring(from, end).strip(), t.substring(end + CLOSE.length()).stripLeading()};
    }

    /**
     * Streams what {@link #response} puts in {@code content}, as llama.cpp streams only the content: the reasoning
     * (a leading {@code <think>} block, or with {@code opensInside} everything up to the first {@code </think>}) is
     * held back, the rest passes as it comes. Only the first characters of an ordinary reply wait, until they show it
     * doesn't start with {@code <think>}. {@link #finish} lets out a start still waiting once the reply is complete.
     */
    static final class Content implements Consumer<String> {
        private enum State { START, THINKING, CONTENT }

        private final Consumer<String> out;
        private final StringBuilder held = new StringBuilder();
        private State state;
        private int scanned;     // THINKING: how much of held has been searched for </think>
        private boolean trim;    // after a </think>: whitespace before the content is dropped

        Content(Consumer<String> out, boolean opensInside) {
            this.out = out;
            this.state = opensInside ? State.THINKING : State.START;
        }

        @Override public void accept(String piece) {
            switch (state) {
                case CONTENT -> emit(piece);
                case START -> {
                    held.append(piece);
                    String t = held.toString().stripLeading();   // short: only the first characters wait here
                    if (t.isEmpty() || OPEN.startsWith(t)) return;
                    if (t.startsWith(OPEN)) {
                        held.setLength(0);
                        held.append(t, OPEN.length(), t.length());
                        state = State.THINKING;
                        scanThinking();
                    } else release();
                }
                case THINKING -> {
                    held.append(piece);
                    scanThinking();
                }
            }
        }

        /** The reply is complete: a start still waiting was content after all; reasoning never closed is dropped. */
        void finish() {
            if (state == State.START) release();
        }

        /** Looks for </think> only in what wasn't searched yet (and the few characters a tag could straddle). */
        private void scanThinking() {
            int end = held.indexOf(CLOSE, Math.max(0, scanned - CLOSE.length() + 1));
            if (end < 0) { scanned = held.length(); return; }
            String rest = held.substring(end + CLOSE.length());
            held.setLength(0);
            state = State.CONTENT;
            trim = true;
            emit(rest);
        }

        private void emit(String piece) {
            if (trim) {
                piece = piece.stripLeading();
                if (piece.isEmpty()) return;
                trim = false;
            }
            out.accept(piece);
        }

        private void release() {
            state = State.CONTENT;
            String text = held.toString();
            held.setLength(0);
            if (!text.isEmpty()) out.accept(text);
        }
    }

    /** A message's content: a string, or OpenAI's array of parts, of which only text is taken. */
    private static String content(Object c) {
        if (c == null) return "";
        if (c instanceof String s) return s;
        if (c instanceof List<?> parts) {
            StringBuilder b = new StringBuilder();
            for (Object p : parts) {
                if (!(p instanceof Map<?, ?> part)) throw badRequest("a content part must be an object");
                Object type = part.get("type");
                if ("text".equals(type)) b.append(part.get("text") instanceof String s ? s : "");
                else throw badRequest("content of type " + type + " isn't supported by the jitllm engine (it takes text only)");
            }
            return b.toString();
        }
        throw badRequest("content must be a string or an array of parts");
    }

    private static List<String> stops(Object stop) {
        if (stop == null) return List.of();
        if (stop instanceof String s) return List.of(s);
        if (stop instanceof List<?> l) {
            List<String> out = new ArrayList<>();
            for (Object o : l) if (o instanceof String s && !s.isEmpty()) out.add(s);
            return out;
        }
        throw badRequest("stop must be a string or an array of strings");
    }

    /** A token count from JSON: a huge one is no limit at all rather than one that wraps around. */
    private static int clamp(Number n) {
        double d = n.doubleValue();
        return d >= Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) d;
    }

    private static Number number(Object v, String name) {
        if (v instanceof Number n) return n;
        throw badRequest(name + " must be a number");
    }
}
