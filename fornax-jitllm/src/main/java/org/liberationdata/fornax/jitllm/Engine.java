package org.liberationdata.fornax.jitllm;

import org.beehive.jitllm.api.CancellationToken;
import org.beehive.jitllm.api.ChatMessage;
import org.beehive.jitllm.api.ChatRole;
import org.beehive.jitllm.api.EmbeddingSession;
import org.beehive.jitllm.api.FinishReason;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.InsufficientDeviceMemoryException;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.SessionOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.api.ThinkingMode;
import java.nio.charset.StandardCharsets;
import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.llama.LlamaModel.Message;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * One jitllm model loaded in this JVM, and the session its calls run on. Calls come one at a time from the model's
 * {@code ModelThread}; {@link #cancel()} may come from any thread. Only this class touches jitllm's API, so a JVM that
 * runs the model in a worker never loads it.
 */
final class Engine implements AutoCloseable {
    private final LocalModel model;
    private final TextGenerationModel text;
    private final EmbeddingSession embeddings;    // embed, or null
    private final int contextLength;             // tokens: the one asked for, else the model's own
    private final boolean opensInside;           // the prompt opens a think block, so replies start inside it
    private final boolean thinkingControl;       // the model's template can turn thinking on and off
    private GenerationSession session;           // chat, or null; opened with sessionThinking
    private boolean sessionThinking;             // the session thinks (for a model with thinking control)
    private volatile CancellationToken running;  // the call in progress, or null

    Engine(Path gguf, int contextSize, boolean embed) {
        ModelOptions.Builder options = ModelOptions.builder();
        if (contextSize > 0) options.contextLength(contextSize);
        try { model = LocalModels.load(gguf, options.build()); }
        catch (IOException e) { throw new UncheckedIOException(gguf + ": " + e.getMessage(), e); }
        catch (RuntimeException e) { throw unsupported(outOfMemory(e)); }
        try {
            if (!(model instanceof TextGenerationModel t))
                throw new IllegalArgumentException(gguf + " isn't a language model jitllm can generate with");
            text = t;
            contextLength = contextSize > 0 ? contextSize : model.info().contextLength();
            // jitllm opens the reasoning in the prompt for these (DeepSeek-R1 distills), as their template does
            opensInside = model.info().architecture().toLowerCase(Locale.ROOT).startsWith("deepseek_r1");
            thinkingControl = model.info().capabilities().thinkingControl();
            embeddings = embed ? EmbeddingSession.open(t) : null;
            session = embed ? null : open(true);
        } catch (RuntimeException e) {
            model.close();
            throw outOfMemory(e);
        }
    }

    int embeddingSize() { return embeddings == null ? 0 : embeddings.dimension(); }

    /** Whether a reply starts inside a think block that its prompt opened, and has only the closing tag. */
    boolean opensInside() { return opensInside; }

    /** A chat, from an empty context: the request carries the whole conversation. */
    ChatJson.Reply chat(Call call, ChatJson.Request r, Consumer<String> onPiece) {
        CancellationToken token = new CancellationToken();
        running = token;
        try {
            if (call.stopRequested()) token.cancel();   // a cancel that came before the call started
            GenerationSession s = session(r.thinking());
            s.reset();
            GenerationRequest.Builder b = GenerationRequest.builder()
                    .messages(messages(r.messages()))
                    .maxNewTokens(Math.min(r.maxTokens(), contextLength))
                    .temperature(Math.max(0f, r.temperature()))
                    .cancellation(token)
                    .onToken(piece -> {
                        if (call.stopRequested()) token.cancel();
                        if (onPiece != null && !piece.isEmpty()) onPiece.accept(piece);
                    });
            if (r.topP() > 0f && r.topP() <= 1f) b.topP(r.topP());
            if (r.seed() >= 0) b.seed(r.seed());
            if (!r.stop().isEmpty()) b.stopSequences(r.stop());
            GenerationResult g;
            try { g = s.generate(b.build()); }
            catch (RuntimeException e) { throw outOfMemory(e); }
            if (g.finishReason() == FinishReason.CONTEXT_FULL && g.generatedTokens() == 0 && !token.isCancelled())
                throw new IllegalStateException("jitllm (rc=2): the prompt is " + g.promptTokens()
                        + " tokens, more than the context holds (" + contextLength + " tokens)");
            return new ChatJson.Reply(g.text(), g.promptTokens(), g.generatedTokens(), finishReason(g), opensInside);
        } finally {
            running = null;
        }
    }

    /**
     * The session for a request's thinking: the one open, else (for a model whose template turns thinking on and off,
     * when a request asks for the other setting) a new one in its place. One session at a time: a second would
     * reserve a second context's memory, while a switch costs well under a second (0.7 s for Qwen3 0.6B on an M4 Pro).
     */
    private GenerationSession session(Boolean thinking) {
        boolean want = !Boolean.FALSE.equals(thinking);   // on unless the request turns it off, as llama.cpp
        if (session != null && (want == sessionThinking || !thinkingControl)) return session;
        if (session != null) session.close();
        session = null;   // until one opens: a failure leaves none, and the next call tries again
        session = open(want);
        return session;
    }

    /**
     * A chat session: for a model whose template turns thinking on and off, thinking as asked (on by default, as
     * llama.cpp's {@code enable_thinking}); for any other, the model's own.
     */
    private GenerationSession open(boolean thinking) {
        try {
            GenerationSession s = !thinkingControl ? text.newSession() : text.newSession(SessionOptions.builder()
                    .thinkingMode(thinking ? ThinkingMode.ENABLED : ThinkingMode.DISABLED).build());
            sessionThinking = thinking;
            return s;
        } catch (RuntimeException e) { throw outOfMemory(e); }
    }

    /**
     * One embedding per text; a text longer than the context is a bad request ({@code rc=2}), as it is for llama.cpp.
     * A text is tokenized for the check only when it could be too long: a token is at least one byte of it.
     */
    float[][] embed(List<String> texts) {
        int room = Math.max(1, contextLength - 5);   // EmbeddingSession's own room: the end-of-text token and a margin
        for (int i = 0; i < texts.size(); i++) {
            String t = texts.get(i);
            if (t.length() <= room && t.getBytes(StandardCharsets.UTF_8).length <= room) continue;
            int n = embeddings.tokens(t).size();
            if (n > room)
                throw new IllegalStateException("jitllm (rc=2): input " + i + " is " + n + " tokens, more than the context holds ("
                        + contextLength + " tokens)");
        }
        try { return embeddings.embedAll(texts).toArray(float[][]::new); }
        catch (RuntimeException e) { throw outOfMemory(e); }
    }

    /** Stops the call in progress: jitllm ends it with the text so far. */
    void cancel() {
        CancellationToken t = running;
        if (t != null) t.cancel();
    }

    @Override public void close() {
        try {
            if (session != null) session.close();
            if (embeddings != null) embeddings.close();
        } finally {
            model.close();
        }
    }

    /**
     * jitllm's out-of-device-memory error, with Fornax's way to give it more (its advice names the jitllm command's
     * flags); anything else as it is.
     */
    static RuntimeException outOfMemory(RuntimeException e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof InsufficientDeviceMemoryException) {
                String budget = System.getProperty("tornado.device.memory");   // a worker always has one
                return new IllegalStateException("jitllm ran out of GPU memory ("
                        + (budget == null ? "TornadoVM's default budget" : "TornadoVM's budget is " + budget)
                        + "): set -Dtornado.device.memory=<N>GB on the JVM that runs Fornax, or use a smaller context or "
                        + "quantization", e);
            }
        }
        return e;
    }

    /**
     * A file jitllm has no model for (a vision or embedding-only architecture, say) as a bad request ({@code rc=7}):
     * jitllm says so before it allocates anything.
     */
    static RuntimeException unsupported(RuntimeException e) {
        String m = String.valueOf(e.getMessage());
        return e instanceof IllegalStateException && m.contains("GPUL-MOD-002")
                ? ChatJson.badRequest(m.substring(m.indexOf(']') + 1).strip() + "; vision models run on llama.cpp")
                : e;
    }

    /** OpenAI's finish reasons: "length" when the reply was cut short, else "stop". */
    private static String finishReason(GenerationResult g) {
        return switch (g.finishReason()) {
            case MAX_TOKENS, CONTEXT_FULL -> "length";
            default -> "stop";
        };
    }

    private static List<ChatMessage> messages(List<Message> messages) {
        List<ChatMessage> out = new ArrayList<>(messages.size());
        for (Message m : messages) out.add(ChatMessage.of(role(m.role()), m.content() == null ? "" : m.content()));
        return out;
    }

    private static ChatRole role(String role) {
        return switch (role.toLowerCase(Locale.ROOT)) {
            case "system", "developer" -> ChatRole.SYSTEM;
            case "user" -> ChatRole.USER;
            case "assistant" -> ChatRole.ASSISTANT;
            default -> throw ChatJson.badRequest("the jitllm engine takes system, developer, user and assistant messages, not " + role);
        };
    }
}
