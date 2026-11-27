package org.liberationdata.fornax.jitllm;

import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.Isolation;
import org.liberationdata.fornax.ModelThread;
import org.liberationdata.fornax.Worker;
import org.liberationdata.fornax.hub.ModelReferences;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;
import java.util.function.Consumer;

/**
 * A GGUF language model running on the GPU through <a href="https://github.com/beehive-lab/jitllm">jitllm</a>, a
 * pure-Java inference engine whose kernels TornadoVM compiles at run time for CUDA, OpenCL or Metal. It takes the same
 * {@link Message}s and {@link Options} as {@code LlamaModel} and gives the same {@link Result}, so code written for
 * one runs on the other: {@code chat}, {@code chatJson} (what fornax-server serves), {@code embed}, streaming, the
 * async methods and {@code cancel}. jitllm samples with temperature and top-p only: {@link Options#minP()} is
 * ignored. Images, tools, JSON-schema constrained output and reranking stay with {@code LlamaModel}.
 *
 * <p>jitllm needs a TornadoVM SDK ({@code TORNADOVM_HOME}) and the JVM options it lists. When this JVM was started
 * with them, {@code --add-modules jdk.incubator.vector}, {@code -Duse.tornadovm=true} and
 * {@code -Dtornado.tvm.maxbytecodesize=65536} (or more), the model loads in-process. Otherwise (a plain {@code java}
 * or {@code ./gradlew run}) it loads in a
 * {@link Worker}, a child JVM started with TornadoVM's options (see {@link TornadoJvm}), behind this same API; so does
 * every JitLlmModel with {@code -Dfornax.isolation=process}.
 *
 * <p>The model's path can be a file, an {@code org/name:TAG} reference or a catalogue alias, as for
 * {@code LlamaModel}: it is downloaded into the Hugging Face cache on first use.
 *
 * <p>Threads: calls run one at a time, in arrival order, from any thread; {@link #cancel()} and {@link #close()} are
 * safe from any thread. Each call starts from an empty context: the messages are the whole conversation.
 */
public final class JitLlmModel implements AutoCloseable {
    /** What a model is loaded for. */
    public enum Mode { CHAT, EMBED }

    private final Mode mode;
    private final ModelThread thread;
    private final Engine local;              // in-process, or null
    private final Worker.Instance remote;    // in a worker, or null
    private final Worker worker;
    private final int embeddingSize;

    /** A chat model with 8192 tokens of context. */
    public JitLlmModel(String ggufPath) { this(ggufPath, 8192, Mode.CHAT); }

    /** @param contextSize tokens of context (the KV cache); 0 for the model's own maximum */
    public JitLlmModel(String ggufPath, int contextSize) { this(ggufPath, contextSize, Mode.CHAT); }

    /** A chat or embedding model, in-process if this JVM has TornadoVM, else in a worker of its own. */
    public JitLlmModel(String ggufPath, int contextSize, Mode mode) {
        this(locate(sdkFirst(ggufPath)), contextSize, mode, inProcessPossible() ? null : workerFor(ggufPath), true);
    }

    /** The path as given, once a model that will need a worker has its TornadoVM SDK: said before any download. */
    private static String sdkFirst(String ggufPath) {
        checkRuntime();
        return ggufPath;
    }

    /**
     * Throws an IllegalStateException that says what to set if jitllm couldn't run here: neither this JVM has
     * TornadoVM (see {@link #inProcessPossible}) nor is there an SDK for a worker. For a caller to check before it
     * downloads a model.
     */
    public static void checkRuntime() {
        if (!inProcessPossible()) TornadoJvm.home();
    }

    /** The model in a worker process of its own, whether or not this JVM has TornadoVM. */
    public static JitLlmModel isolated(String ggufPath, int contextSize, Mode mode) {
        TornadoJvm.home();   // before any download
        return new JitLlmModel(locate(ggufPath), contextSize, mode, workerFor(ggufPath), true);
    }

    /** In this JVM whatever fornax.isolation says: what a worker loads. */
    static JitLlmModel inProcess(String ggufPath, int contextSize, Mode mode) {
        if (!tornadoJvm())
            throw new IllegalStateException("jitllm in this JVM needs TornadoVM's tornado-argfile, --add-modules "
                    + "jdk.incubator.vector, -Duse.tornadovm=true and -Dtornado.tvm.maxbytecodesize=" + MAX_BYTECODE);
        return new JitLlmModel(ggufPath, contextSize, mode, null, false);
    }

    private JitLlmModel(String ggufPath, int contextSize, Mode mode, Worker worker, boolean ownsWorker) {
        if (contextSize < 0) throw new IllegalArgumentException("contextSize must be 0 or more");
        this.mode = java.util.Objects.requireNonNull(mode, "mode");
        String name = "jitllm-" + fileName(ggufPath);
        if (worker != null) {
            Worker.Instance r;
            try {
                r = worker.open(JitLlmService.class.getName(), JitLlmService.config(ggufPath, contextSize, mode));
                embeddingSize = mode == Mode.EMBED ? JitLlmService.decodeInt(r.invoke("info", null)) : 0;
            } catch (RuntimeException e) {
                if (ownsWorker) worker.close();
                throw e;
            }
            Worker.Instance instance = r;
            this.remote = instance;
            this.local = null;
            this.worker = worker;
            thread = new ModelThread(name, () -> {
                try { instance.close(); } finally { if (ownsWorker) worker.close(); }
            }, instance::cancel, null);
            return;
        }
        this.remote = null;
        this.worker = null;
        Engine e = new Engine(Path.of(ggufPath), contextSize, mode == Mode.EMBED);
        this.local = e;
        embeddingSize = e.embeddingSize();
        thread = new ModelThread(name, e::close, e::cancel, null);
    }

    /** Whether this model runs in a worker process. */
    public boolean isIsolated() { return remote != null; }

    /** The worker this model runs in, or null for one in-process. */
    public Worker worker() { return worker; }

    public Mode mode() { return mode; }

    /** The length of an embedding, for a model loaded in {@link Mode#EMBED}. */
    public int embeddingSize() {
        requireMode(Mode.EMBED, "embeddingSize");
        return embeddingSize;
    }

    /** Stops the call in progress, if any: it returns the text so far. */
    public void cancel() { thread.cancel(); }

    /**
     * Stops the engine's generation in progress, prompt processing included: for a worker, whose calls run on the
     * worker's own thread rather than this model's.
     */
    void cancelEngine() {
        if (local != null) local.cancel();
        else thread.cancel();
    }

    // ---- chat ----

    public Result chat(List<Message> messages, Options o) { return chat(messages, o, null); }

    /**
     * Formats the messages with the model's chat template and generates. onPiece (optional) streams text as it is
     * produced. Messages with images are refused: jitllm runs language models only.
     */
    public Result chat(List<Message> messages, Options o, Consumer<String> onPiece) {
        return thread.run(c -> chatOn(c, messages, o, onPiece));
    }

    public CompletableFuture<Result> chatAsync(List<Message> messages, Options o) {
        return thread.submit(c -> chatOn(c, messages, o, null));
    }

    public CompletableFuture<Result> chatAsync(List<Message> messages, Options o, Flow.Subscriber<? super String> pieces) {
        return thread.submit((Call c, Consumer<String> sink) -> chatOn(c, messages, o, sink), pieces);
    }

    public Flow.Publisher<String> chatStream(List<Message> messages, Options o) {
        return thread.publisher((Call c, Consumer<String> sink) -> chatOn(c, messages, o, sink));
    }

    Result chatOn(Call call, List<Message> messages, Options o, Consumer<String> onPiece) {
        requireMode(Mode.CHAT, "chat");
        if (messages.stream().anyMatch(m -> !m.images().isEmpty()))
            throw new UnsupportedOperationException("jitllm runs language models only: use LlamaModel for images");
        if (remote != null)
            return JitLlmService.decodeResult(remote(call, "chat", JitLlmService.chatArgs(messages, o, onPiece != null), onPiece));
        return local.chat(call, ChatJson.Request.of(messages, o), onPiece).result();
    }

    /**
     * OpenAI-style chat, as {@code LlamaModel.chatJson}: the request's {@code messages}, {@code max_tokens} (or
     * {@code max_completion_tokens}), {@code temperature}, {@code top_p}, {@code seed} and {@code stop}; the response
     * <pre>
     * {"message": {"role": "assistant", "content": "...", "reasoning_content": "..."},
     *  "finish_reason": "stop|length", "usage": {"prompt_tokens": n, "completion_tokens": n}}
     * </pre>
     * onContent (optional) streams the content. A request with tools, images or a JSON schema is refused.
     */
    public String chatJson(String requestJson, Consumer<String> onContent) {
        return thread.run(c -> chatJsonOn(c, requestJson, onContent));
    }

    public CompletableFuture<String> chatJsonAsync(String requestJson) {
        return thread.submit(c -> chatJsonOn(c, requestJson, null));
    }

    public CompletableFuture<String> chatJsonAsync(String requestJson, Flow.Subscriber<? super String> content) {
        return thread.submit((Call c, Consumer<String> sink) -> chatJsonOn(c, requestJson, sink), content);
    }

    String chatJsonOn(Call call, String requestJson, Consumer<String> onContent) {
        requireMode(Mode.CHAT, "chat");
        if (remote != null)
            return JitLlmService.decodeString(remote(call, "chatJson", JitLlmService.chatJsonArgs(requestJson, onContent != null), onContent));
        ChatJson.Request request = ChatJson.parse(requestJson);
        ChatJson.Content content = onContent == null ? null : new ChatJson.Content(onContent, local.opensInside());
        ChatJson.Reply reply = local.chat(call, request, content);
        if (content != null) content.finish();
        return ChatJson.response(reply);
    }

    // ---- embeddings ----

    /** One embedding per text, L2-normalised. */
    public float[][] embed(List<String> texts) {
        return thread.run(c -> embedOn(c, texts));
    }

    public float[] embed(String text) { return embed(List.of(text))[0]; }

    public CompletableFuture<float[][]> embedAsync(List<String> texts) {
        return thread.submit(c -> embedOn(c, texts));
    }

    float[][] embedOn(Call call, List<String> texts) {
        requireMode(Mode.EMBED, "embed");
        if (remote != null) return JitLlmService.decodeEmbeddings(remote.invoke("embed", JitLlmService.embedArgs(texts), call, null));
        return local.embed(texts);
    }

    @Override public void close() { thread.close(); }

    // ---- plumbing ----

    private byte[] remote(Call call, String method, byte[] args, Consumer<String> onPiece) {
        return remote.invoke(method, args, call, onPiece == null ? null : b -> onPiece.accept(JitLlmService.decodeString(b)));
    }

    private void requireMode(Mode m, String what) {
        if (mode != m) throw new IllegalStateException(what + " needs a model loaded in " + m + " mode; this one is " + mode);
    }

    /**
     * Whether jitllm can run on the GPU in this JVM: it was started with TornadoVM's modules, the Vector API,
     * {@code -Duse.tornadovm=true} (without which jitllm runs on the CPU) and a bytecode buffer big enough for prompt
     * ingestion, and fornax.isolation doesn't ask for a worker. A worker itself always runs its models in-process.
     */
    static boolean inProcessPossible() {
        return Isolation.configured() == Isolation.IN_PROCESS && tornadoJvm();
    }

    /** The interpreter bytecode buffer jitllm's prompt ingestion needs: TornadoVM's default is smaller. */
    static final int MAX_BYTECODE = 65536;

    private static boolean tornadoJvm() {
        return ModuleLayer.boot().findModule("tornado.runtime").isPresent()
                && ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()
                && Boolean.getBoolean("use.tornadovm")
                && Integer.getInteger("tornado.tvm.maxbytecodesize", 0) >= MAX_BYTECODE;
    }

    private static Worker workerFor(String ggufPath) {
        return new Worker(fileName(ggufPath), Worker.Options.DEFAULT.withJvmArgs(TornadoJvm.workerArgs().toArray(String[]::new)));
    }

    private static String fileName(String path) {
        return path.substring(Math.max(path.lastIndexOf('/'), path.lastIndexOf(java.io.File.separatorChar)) + 1);
    }

    /** A file as given, or the GGUF a reference or catalogue alias names, downloaded the first time. */
    private static String locate(String gguf) {
        Optional<Map<String, Path>> fetched = ModelReferences.locate(gguf, ENGINE);
        return fetched.map(files -> files.get("gguf").toString()).orElse(gguf);
    }

    private static final ModelReferences.Engine ENGINE = new ModelReferences.Engine("JitLlmModel", "language models",
            Set.of(ModelSet.Kind.CHAT), Set.of("gguf"), "a GGUF language model");
}
