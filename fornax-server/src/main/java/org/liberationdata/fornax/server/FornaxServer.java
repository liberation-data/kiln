package org.liberationdata.fornax.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.PullListener;
import org.liberationdata.fornax.server.Engines.ImageJob;
import org.liberationdata.fornax.server.Engines.Stopper;
import org.liberationdata.fornax.server.Engines.Use;
import org.liberationdata.fornax.whisper.Transcript;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;

/**
 * An OpenAI-compatible HTTP server over Fornax's engines, JDK only ({@code com.sun.net.httpserver}), so any OpenAI
 * client, agent framework or tool can use local models with no Java code of its own:
 *
 * <pre>{@code
 * try (var server = FornaxServer.builder().port(54154).model("qwen3-4b").build().start()) {
 *     // POST http://127.0.0.1:54154/v1/chat/completions ...
 *     server.await();
 * }
 * }</pre>
 *
 * <ul>
 *   <li>{@code POST /v1/chat/completions}: messages, tools and {@code tool_calls}, {@code response_format} (a JSON
 *       schema), {@code temperature}, {@code top_p}, {@code seed}, {@code max_tokens}, {@code stop}, images as
 *       {@code image_url} data: URIs for a vision model, and {@code stream: true} as server-sent events</li>
 *   <li>{@code GET /v1/models}: the models it was started with, those loaded, and those downloaded</li>
 *   <li>{@code POST /v1/embeddings}: a string or an array of them, as floats or base64</li>
 *   <li>{@code POST /v1/images/generations}: PNGs as {@code b64_json}; {@code size}, {@code n}, and {@code seed},
 *       {@code steps} and {@code cfg_scale} beyond OpenAI's</li>
 *   <li>{@code POST /v1/audio/speech} (WAV) and {@code POST /v1/audio/transcriptions} (a WAV file, multipart)</li>
 * </ul>
 *
 * A request may name the models given to {@link Builder#model}, which are downloaded and loaded at {@link #start}
 * (the first chat one answers a request that names no model), and catalogue models already downloaded; any other is
 * a 404 ({@code model_not_found}) unless {@link Builder#allowDownloads} lets a request name any model Fornax can load
 * (a catalogue alias, an {@code org/name[:TAG]} reference or a path), downloaded the first time. Models are kept
 * loaded in a {@link ModelPool} (least recently used unloaded to stay within the budget, and idle ones after the idle
 * timeout).
 *
 * <p>Who may call it: it listens on 127.0.0.1 unless told otherwise; on loopback, it refuses a request whose
 * {@code Host} isn't a loopback name or address with its port (DNS rebinding); it refuses a request from a web page
 * (an {@code Origin}) unless that origin is allowed with {@link Builder#corsOrigin}; and with an {@link Builder#apiKey},
 * it refuses a request without {@code Authorization: Bearer <key>}. Errors are OpenAI's: {@code {"error": {"message",
 * "type", "param", "code"}}} with the HTTP status; a 500's message has each local path cut to its file name (the
 * {@link Builder#log} keeps it). A refused request's body is read first, up to 64 KB more of it, so that its client
 * gets the error rather than a reset; past that, the error says {@code Connection: close}.
 *
 * <p>Slow clients: the JDK HTTP server never times out a client that sends its request a byte at a time. Its limits
 * are JVM-wide system properties, read when its classes load, so this class leaves them to the application: pass
 * {@code -Dsun.net.httpserver.maxReqTime=120} (seconds to send a request) and {@code -Djdk.httpserver.maxConnections=512}
 * to {@code java}, as {@code fornax server} sets them. Not {@code sun.net.httpserver.maxRspTime}, which would cut off
 * a long stream.
 *
 * <p>One property it does set, unless given: {@code sun.net.httpserver.nodelay=true} (see {@link #NODELAY}).
 */
public final class FornaxServer implements AutoCloseable {
    /**
     * Before JDK 25, the JDK HTTP server writes a reply's headers and its body in two writes, and with Nagle's algorithm
     * on (its default) the body waits for the client to acknowledge the headers, which a client delays by up to 40 ms:
     * every reply took 40 ms more, a stream's pieces were held back, and a reply sent before the request was read whole
     * (a 413) could be lost to the reset as the connection closed, its body still unsent. TCP_NODELAY sends each write
     * at once; JDK 25 buffers the headers with the body, and loses nothing by it. The JDK reads it once per JVM, when its
     * server classes load, so it is set as this class loads; an explicit {@code -Dsun.net.httpserver.nodelay} wins.
     */
    static final String NODELAY = "sun.net.httpserver.nodelay";

    static {
        if (System.getProperty(NODELAY) == null) System.setProperty(NODELAY, "true");
    }

    /** The port it listens on unless told otherwise. */
    public static final int DEFAULT_PORT = 54154;
    /** The embedding model {@code /v1/embeddings} knows by name (Qwen3-Embedding 0.6B), and uses when a request names none. */
    public static final String EMBEDDING_MODEL = PooledEngines.EMBEDDING;

    private final HttpServer http;
    private final ExecutorService executor;
    private final Engines engines;
    private final Guard guard;
    private final Consumer<String> log;
    private final long maxBody;
    /** The request bodies held at once, in KB: past it, a request is answered 503 rather than buffered too. */
    private final java.util.concurrent.Semaphore buffering;
    /** All of {@link #buffering}'s room, in KB. */
    private final int bufferingKb;
    /** The KB of {@link #buffering} each request in hand holds, given back when it is answered. */
    private final Map<HttpExchange, Integer> held = new java.util.concurrent.ConcurrentHashMap<>();
    private final Runnable warm;
    /** Held while a transcription's audio is decoded and transcribed: one at a time (see {@link #transcriptions}). */
    private final java.util.concurrent.Semaphore transcribing = new java.util.concurrent.Semaphore(1);
    /** The chat generations running, which {@link #close} stops. */
    private final Set<Stopper> generating = java.util.concurrent.ConcurrentHashMap.newKeySet();
    private final Object stopped = new Object();
    private boolean running, started;

    private FornaxServer(Builder b) throws IOException {
        InetAddress address = InetAddress.getByName(b.host);
        this.guard = new Guard(address.isLoopbackAddress(), b.cors, b.apiKey);
        this.log = b.log;
        this.maxBody = Math.min(b.maxBodyBytes, Integer.MAX_VALUE - 8);   // an array's most
        if (b.maxBufferedBytes < b.maxBodyBytes)
            throw new IllegalArgumentException("maxBufferedBytes (" + b.maxBufferedBytes + ") must be at least maxBodyBytes ("
                    + b.maxBodyBytes + "), or a body that large could never be taken");
        this.bufferingKb = (int) Math.min(Integer.MAX_VALUE, kb(b.maxBufferedBytes));
        this.buffering = new java.util.concurrent.Semaphore(bufferingKb);
        if (b.engines != null) {
            this.engines = b.engines;
            this.warm = () -> {};
        } else {
            var pooled = new PooledEngines(b.hub != null ? b.hub : FornaxHub.standard(), new ModelPool(b.budgetBytes, b.idleTimeout),
                    b.models, b.contextSize, b.parallel, b.pulls, b.log, b.allowDownloads, b.languageEngine);
            this.engines = pooled;
            this.warm = pooled::warm;
        }
        this.http = HttpServer.create(new InetSocketAddress(address, b.port), 0);   // binds now: a port in use fails here
        // Platform threads: a generation is one long native call, which would pin a virtual thread's carrier throughout.
        this.executor = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "fornax-server-http");
            t.setDaemon(true);
            return t;
        });
        http.setExecutor(executor);
        http.createContext("/", this::handle);
    }

    public static Builder builder() { return new Builder(); }

    /**
     * Downloads and loads the models given to {@link Builder#model}, then answers requests. Returns this.
     *
     * @throws RuntimeException as the engines throw, when a model won't download or load (the server is closed then)
     */
    public FornaxServer start() {
        try { warm.run(); }
        catch (ApiError e) { close(); throw new IllegalStateException(e.getMessage(), e); }
        catch (RuntimeException | Error e) { close(); throw e; }
        synchronized (stopped) {
            http.start();
            started = running = true;
        }
        return this;
    }

    /** The port it listens on (the one chosen, for port 0). */
    public int port() { return http.getAddress().getPort(); }

    /** The API's base URL, as an OpenAI client's base URL: {@code http://127.0.0.1:54154/v1}. */
    public String url() {
        InetAddress a = http.getAddress().getAddress();
        // ::1 as [::1], not [0:0:0:0:0:0:0:1]: the Host check takes the short form
        String host = a.isAnyLocalAddress() ? "127.0.0.1" : a.isLoopbackAddress() && a instanceof java.net.Inet6Address ? "[::1]"
                : a instanceof java.net.Inet6Address ? "[" + a.getHostAddress() + "]" : a.getHostAddress();
        return "http://" + host + ":" + port() + "/v1";
    }

    /** Waits until the server is closed (from another thread, or a shutdown hook). */
    public void await() throws InterruptedException {
        synchronized (stopped) { while (running) stopped.wait(); }
    }

    /** Stops answering, waits a moment for requests in flight, stops the chat generations still running, and unloads every model. */
    @Override public void close() {
        // the listening socket is only released by the server's dispatcher, so one never started (a model that
        // wouldn't load) is started to be stopped, or its port stays taken
        synchronized (stopped) { if (!started) { started = true; http.start(); } }
        http.stop(1);
        generating.forEach(Stopper::stop);   // else unloading waits for each to run to its end
        executor.shutdownNow();
        engines.close();
        synchronized (stopped) { running = false; stopped.notifyAll(); }
    }

    // ---- routing ----

    private interface Route { void handle(HttpExchange ex) throws IOException; }

    private Route route(String method, String path) {
        boolean get = method.equals("GET") || method.equals("HEAD"), post = method.equals("POST");
        if (path.equals("/health")) return get ? ex -> json(ex, 200, Json.obj("status", "ok")) : null;
        if (path.equals("/v1/models")) return get ? this::models : null;
        if (path.startsWith("/v1/models/")) return get ? ex -> model(ex, path.substring("/v1/models/".length())) : null;
        return switch (path) {
            case "/v1/chat/completions" -> post ? this::chat : null;
            case "/v1/embeddings" -> post ? this::embeddings : null;
            case "/v1/images/generations" -> post ? this::images : null;
            case "/v1/audio/speech" -> post ? this::speech : null;
            case "/v1/audio/transcriptions" -> post ? this::transcriptions : null;
            default -> throw new ApiError(404, "invalid_request_error", "no such endpoint: " + method + " " + path, null, "not_found");
        };
    }

    private void handle(HttpExchange ex) {
        long t0 = System.nanoTime();
        String method = ex.getRequestMethod(), path = ex.getRequestURI().getPath();
        int[] status = {0};
        try {
            var h = ex.getRequestHeaders();
            boolean preflight = method.equals("OPTIONS") && h.containsKey("Origin") && h.containsKey("Access-Control-Request-Method");
            // CORS headers first, so a page from an allowed origin can read why it was refused (a missing API key)
            guard.cors(h, ex.getResponseHeaders(), preflight);
            ApiError refused = guard.refusal(h, port(), preflight);
            if (refused != null) {
                status[0] = error(ex, refused);
                return;
            }
            if (preflight) {
                ex.sendResponseHeaders(status[0] = 204, -1);
                return;
            }
            Route r = route(method, path);
            if (r == null) {
                ex.getResponseHeaders().set("Allow", path.equals("/health") || path.startsWith("/v1/models") ? "GET, HEAD" : "POST");
                status[0] = error(ex, new ApiError(405, "invalid_request_error", method + " isn't allowed on " + path, null, "method_not_allowed"));
                return;
            }
            r.handle(ex);
            status[0] = ex.getResponseCode();
        } catch (ApiError e) {
            status[0] = error(ex, e);
            // a 500's body has its paths cut (ApiError.body): the log keeps them for whoever runs the server
            if (e.status == 500) log.accept("error in " + method + " " + path + ": " + e.getMessage());
        } catch (CancellationException e) {
            status[0] = 499;   // the client went away mid-stream: there is no one to tell
        } catch (RuntimeException | IOException | Error e) {
            status[0] = error(ex, ApiError.server(e));
            if (!(e instanceof IOException)) log.accept("error in " + method + " " + path + ": " + e);
        } finally {
            release(ex);
            ex.close();
            log.accept(String.format(Locale.ROOT, "%s %s %d %.2fs", method, path, status[0], (System.nanoTime() - t0) / 1e9));
        }
    }

    // ---- /v1/chat/completions ----

    private void chat(HttpExchange ex) throws IOException {
        ChatRequest req = ChatRequest.of(Json.object(body(ex)), engines.defaultModel(Use.CHAT));
        String id = "chatcmpl-" + HexFormat.of().formatHex(randomBytes(12));
        long created = System.currentTimeMillis() / 1000;
        if (req.stream()) { stream(ex, req, id, created); return; }

        Stopper stopper = new Stopper();
        Stops stops = new Stops(req.stops(), piece -> {});
        String res;
        generating.add(stopper);
        try {
            res = engines.chat(req.model(), req.engineJson(), req.images(),
                    req.stops().isEmpty() ? null : piece -> { if (stops.accept(piece)) stopper.stop(); }, stopper);
        } finally { generating.remove(stopper); }
        Map<String, Object> r = Json.map(org.liberationdata.fornax.Json.parse(res));
        Map<String, Object> message = Json.map(r.get("message"));
        uniqueToolCallIds(message);
        Object finish = r.get("finish_reason");
        if (stops.stopped()) {
            message.put("content", stops.text());
            finish = "stop";
        }
        json(ex, 200, Json.obj("id", id, "object", "chat.completion", "created", created, "model", req.model(),
                "choices", List.of(Json.obj("index", 0, "message", message, "logprobs", null, "finish_reason", finish)),
                "usage", usage(r.get("usage"))));
    }

    /** {@code stream: true}: the content as it is generated, then the tool calls and the finish reason, then [DONE]. */
    private void stream(HttpExchange ex, ChatRequest req, String id, long created) {
        engines.checkChat(req.model(), req.images());   // a name it won't serve is still an error status, not a stream
        Stopper stopper = new Stopper();
        generating.add(stopper);
        Sse sse = new Sse(ex, stopper::stop);
        boolean[] roleSent = {false};
        Consumer<Map<String, Object>> delta = d -> {
            if (!roleSent[0]) {
                roleSent[0] = true;
                sse.data(chunk(id, created, req.model(), Json.obj("role", "assistant", "content", ""), null));
            }
            sse.data(chunk(id, created, req.model(), d, null));
        };
        try {
            Stops stops = new Stops(req.stops(), piece -> delta.accept(Json.obj("content", piece)));
            String res = engines.chat(req.model(), req.engineJson(), req.images(), piece -> { if (stops.accept(piece)) stopper.stop(); }, stopper);
            stops.finish();
            if (stopper.requested() && !stops.stopped()) throw new CancellationException("the client has gone");
            Map<String, Object> r = Json.map(org.liberationdata.fornax.Json.parse(res));
            Map<String, Object> message = Json.map(r.get("message"));
            uniqueToolCallIds(message);
            if (message.get("reasoning_content") instanceof String s && !s.isEmpty()) delta.accept(Json.obj("reasoning_content", s));
            List<Object> calls = Json.list(message.get("tool_calls"));
            if (calls != null && !calls.isEmpty()) {
                List<Object> indexed = new ArrayList<>();
                for (int i = 0; i < calls.size(); i++) {
                    Map<String, Object> c = new java.util.LinkedHashMap<>();
                    c.put("index", i);
                    c.putAll(Json.map(calls.get(i)));
                    indexed.add(c);
                }
                delta.accept(Json.obj("tool_calls", indexed));
            }
            if (!roleSent[0]) delta.accept(Json.obj("content", ""));
            sse.data(chunk(id, created, req.model(), Json.obj(), stops.stopped() ? "stop" : r.get("finish_reason")));
            if (req.includeUsage())
                sse.data(Json.obj("id", id, "object", "chat.completion.chunk", "created", created, "model", req.model(),
                        "choices", List.of(), "usage", usage(r.get("usage"))));
            release(ex);   // held while it was generated (req holds the body still); free before the client has its end
            sse.done();
        } catch (ApiError | CancellationException e) {
            if (!sse.abandon()) streamError(sse, e);
            else throw e;
        } catch (RuntimeException e) {
            if (!sse.abandon()) streamError(sse, e);
            else throw e;
        } finally {
            generating.remove(stopper);
            release(ex);
            sse.close();
        }
    }

    /** A failure after the stream has started: an error event (if the client is still there), as OpenAI sends one. */
    private void streamError(Sse sse, RuntimeException e) {
        if (e instanceof CancellationException) throw e;
        ApiError a = e instanceof ApiError x ? x : ApiError.server(e);
        try { sse.data(a.body()); } catch (CancellationException gone) { /* nobody to tell */ }
        log.accept("error mid-stream: " + e);
    }

    /**
     * The engine numbers the tool calls a template gives no id ({@code call_0}, {@code call_1}), the same in every reply,
     * so a conversation's second round of calls would repeat its first's ids, which clients that key tool results by id
     * across a conversation mix up. Such calls get a random id, as OpenAI's and llama-server's do.
     */
    static void uniqueToolCallIds(Map<String, Object> message) {
        List<Object> calls = message == null ? null : Json.list(message.get("tool_calls"));
        if (calls == null) return;
        for (Object c : calls) {
            Map<String, Object> call = Json.map(c);
            if (call != null && !(call.get("id") instanceof String id && !id.isEmpty() && !ENGINE_CALL_ID.matcher(id).matches()))
                call.put("id", "call_" + HexFormat.of().formatHex(randomBytes(12)));
        }
    }

    private static final java.util.regex.Pattern ENGINE_CALL_ID = java.util.regex.Pattern.compile("call_\\d+");

    private static Map<String, Object> chunk(String id, long created, String model, Map<String, Object> delta, Object finish) {
        return Json.obj("id", id, "object", "chat.completion.chunk", "created", created, "model", model,
                "choices", List.of(Json.obj("index", 0, "delta", delta, "logprobs", null, "finish_reason", finish)));
    }

    private static Map<String, Object> usage(Object u) {
        long p = org.liberationdata.fornax.Json.get(u, "prompt_tokens") instanceof Number n ? n.longValue() : 0;
        long c = org.liberationdata.fornax.Json.get(u, "completion_tokens") instanceof Number n ? n.longValue() : 0;
        return Json.obj("prompt_tokens", p, "completion_tokens", c, "total_tokens", p + c);
    }

    // ---- /v1/models ----

    private void models(HttpExchange ex) throws IOException {
        List<Object> data = new ArrayList<>();
        for (var m : engines.models()) data.add(modelJson(m));
        json(ex, 200, Json.obj("object", "list", "data", data));
    }

    private void model(HttpExchange ex, String id) throws IOException {
        for (var m : engines.models()) if (m.id().equals(id)) { json(ex, 200, modelJson(m)); return; }
        throw ApiError.noSuchModel("no model " + id + " here (GET /v1/models lists them)");
    }

    private static Map<String, Object> modelJson(Engines.Listed m) {
        return Json.obj("id", m.id(), "object", "model", "created", 0, "owned_by", "fornax", "kind", m.kind(), "loaded", m.loaded());
    }

    // ---- /v1/embeddings ----

    /** The most inputs {@code /v1/embeddings} embeds in one request, as OpenAI's. */
    static final int MAX_EMBEDDING_INPUTS = 2048;

    private void embeddings(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.object(body(ex));
        String model = ChatRequest.model(body, engines.defaultModel(Use.EMBED));
        List<String> texts = new ArrayList<>();
        Object input = body.get("input");
        if (input instanceof String s) texts.add(s);
        else if (Json.list(input) != null) {
            for (Object o : Json.list(input)) {
                if (!(o instanceof String s)) throw ApiError.badRequest("input must be a string or an array of strings (token arrays aren't taken)", "input");
                texts.add(s);
            }
        }
        if (texts.isEmpty()) throw ApiError.badRequest("input must be a string or a non-empty array of strings", "input");
        if (texts.size() > MAX_EMBEDDING_INPUTS)   // as OpenAI: else a body of short strings is a vector each, gigabytes of them
            throw ApiError.badRequest("input can have at most " + MAX_EMBEDDING_INPUTS + " strings; split it into several requests", "input");
        if (texts.contains("")) throw ApiError.badRequest("input can't be an empty string", "input");   // as OpenAI: it has no tokens to embed
        Object format = body.getOrDefault("encoding_format", "float");
        if (!"float".equals(format) && !"base64".equals(format)) throw ApiError.badRequest("encoding_format must be float or base64", "encoding_format");
        Object dims = body.get("dimensions");
        int dimensions = dims == null ? 0 : (int) Math.min(Integer.MAX_VALUE, ChatRequest.whole(dims, "dimensions", 1));

        float[][] vectors = engines.embed(model, texts);
        List<Object> data = new ArrayList<>();
        for (int i = 0; i < vectors.length; i++) {
            float[] v = dimensions == 0 ? vectors[i] : truncate(vectors[i], dimensions);
            data.add(Json.obj("object", "embedding", "index", i, "embedding", "base64".equals(format) ? base64(v) : v));
        }
        json(ex, 200, Json.obj("object", "list", "data", data, "model", model, "usage", Json.obj("prompt_tokens", 0, "total_tokens", 0)));
    }

    /** The first n dimensions, normalised again (Qwen3-Embedding is trained for this: Matryoshka representation). */
    static float[] truncate(float[] v, int n) {
        if (n > v.length) throw ApiError.badRequest("dimensions can be at most " + v.length + " for this model", "dimensions");
        float[] t = java.util.Arrays.copyOf(v, n);
        double norm = 0;
        for (float x : t) norm += x * x;
        norm = Math.sqrt(norm);
        if (norm > 0) for (int i = 0; i < n; i++) t[i] = (float) (t[i] / norm);
        return t;
    }

    /** Little-endian float32s, base64, as OpenAI's {@code encoding_format: "base64"} gives them. */
    static String base64(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) b.putFloat(x);
        return Base64.getEncoder().encodeToString(b.array());
    }

    // ---- /v1/images/generations ----

    private void images(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.object(body(ex));
        ImageJob job = imageJob(body);
        String model = ChatRequest.model(body, engines.defaultModel(Use.IMAGE));
        List<Object> data = new ArrayList<>();
        for (BufferedImage img : engines.images(model, job)) data.add(Json.obj("b64_json", Base64.getEncoder().encodeToString(png(img))));
        json(ex, 200, Json.obj("created", System.currentTimeMillis() / 1000, "data", data, "output_format", "png",
                "size", job.width() + "x" + job.height()));
    }

    /** The most steps an image request may take (Fornax's own parameter; the models need 4 to 50). */
    static final int MAX_STEPS = 100;
    /** The most pixels an image may have (4 megapixels: 2048 x 2048, or 4096 x 1024), the most the image models make well. */
    static final int MAX_IMAGE_PIXELS = 2048 * 2048;
    /** An image size: digits only (no sign, no trailing x), at most 5 a side so it parses as an int. */
    private static final java.util.regex.Pattern SIZE = java.util.regex.Pattern.compile("([0-9]{1,5})x([0-9]{1,5})");

    /**
     * The image request, checked: a prompt, n from 1 to 8, a size in multiples of 16 of at most {@link #MAX_IMAGE_PIXELS},
     * and Fornax's own steps (at most {@link #MAX_STEPS}) and cfg_scale.
     */
    static ImageJob imageJob(Map<String, Object> body) {
        if (!(body.get("prompt") instanceof String prompt) || prompt.isBlank()) throw ApiError.badRequest("prompt is required", "prompt");
        Object format = body.get("response_format");
        if (format != null && !"b64_json".equals(format))
            throw ApiError.badRequest("response_format must be b64_json: the server keeps no files to link to", "response_format");
        Object output = body.get("output_format");
        if (output != null && !"png".equals(output)) throw ApiError.badRequest("output_format must be png", "output_format");
        long n = body.get("n") == null ? 1 : ChatRequest.whole(body.get("n"), "n", 1);   // checked before it is an int: 2^32 + 1 isn't 1
        if (n > 8) throw ApiError.badRequest("n can be at most 8", "n");
        Object size = body.get("size");
        int w = 1024, h = 1024;
        if (size != null && !"auto".equals(size)) {
            var wh = size instanceof String s ? SIZE.matcher(s) : null;
            if (wh == null || !wh.matches()) throw ApiError.badRequest("size must be WIDTHxHEIGHT, such as 1024x1024", "size");
            w = Integer.parseInt(wh.group(1));
            h = Integer.parseInt(wh.group(2));
            if (w < 64 || h < 64 || w > 4096 || h > 4096 || w % 16 != 0 || h % 16 != 0)
                throw ApiError.badRequest("size must be from 64 to 4096 a side, in multiples of 16", "size");
            if ((long) w * h > MAX_IMAGE_PIXELS)
                throw ApiError.badRequest("size can be at most " + MAX_IMAGE_PIXELS / (1 << 20) + " megapixels (2048x2048, or 4096x1024)", "size");
        }
        long steps = body.get("steps") == null ? 20 : ChatRequest.whole(body.get("steps"), "steps", 1);
        if (steps > MAX_STEPS) throw ApiError.badRequest("steps can be at most " + MAX_STEPS, "steps");
        Object cfg = body.get("cfg_scale");
        if (cfg != null && !(cfg instanceof Number c && c.doubleValue() >= 0)) throw ApiError.badRequest("cfg_scale must be 0 or more", "cfg_scale");
        long seed = body.get("seed") == null ? ThreadLocalRandom.current().nextInt(Integer.MAX_VALUE) : ChatRequest.whole(body.get("seed"), "seed", 0);
        return new ImageJob(prompt, w, h, (int) steps, cfg == null ? null : ((Number) cfg).doubleValue(), seed, (int) n);
    }

    private static byte[] png(BufferedImage img) {
        var out = new ByteArrayOutputStream();
        try { ImageIO.write(img, "png", out); } catch (IOException e) { throw new UncheckedIOException(e); }
        return out.toByteArray();
    }

    // ---- /v1/audio ----

    /** The sample rates a WAV to transcribe may have: telephone audio's to studio audio's. */
    static final int MIN_SAMPLE_RATE = 8000, MAX_SAMPLE_RATE = 384_000;

    /** The most characters {@code /v1/audio/speech} speaks in one request, as OpenAI's. */
    static final int MAX_SPEECH_INPUT = 4096;

    private void speech(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.object(body(ex));
        if (!(body.get("input") instanceof String input) || input.isBlank()) throw ApiError.badRequest("input is required", "input");
        if (input.codePointCount(0, input.length()) > MAX_SPEECH_INPUT)   // as OpenAI: speech is one request's text, not a book's
            throw ApiError.badRequest("input can be at most " + MAX_SPEECH_INPUT + " characters; split longer text", "input");
        String model = ChatRequest.model(body, engines.defaultModel(Use.SPEECH));
        Object format = body.get("response_format");
        if (format != null && !"wav".equals(format)) throw ApiError.badRequest("response_format must be wav", "response_format");
        Object voice = body.get("voice");
        if (voice != null && !(voice instanceof String)) throw ApiError.badRequest("voice must be a string", "voice");
        Object speed = body.get("speed");
        if (speed != null && !(speed instanceof Number s && s.doubleValue() >= 0.25 && s.doubleValue() <= 4))
            throw ApiError.badRequest("speed must be from 0.25 to 4", "speed");
        Audio audio = engines.speak(model, input, (String) voice, speed == null ? 0 : ((Number) speed).floatValue());
        send(ex, 200, "audio/wav", audio.wav());
    }

    private void transcriptions(HttpExchange ex) throws IOException {
        var parts = Multipart.parse(ex.getRequestHeaders().getFirst("Content-Type"), bytes(ex));
        var file = parts.get("file");
        if (file == null) throw ApiError.badRequest("file is required (a WAV)", "file");
        String model = parts.containsKey("model") && !parts.get("model").text().isBlank() ? parts.get("model").text().strip()
                : engines.defaultModel(Use.TRANSCRIPTION);
        if (model == null) throw ApiError.badRequest("model is required", "model");
        String format = parts.containsKey("response_format") ? parts.get("response_format").text().strip() : "json";
        if (!Set.of("json", "text", "verbose_json", "srt", "vtt").contains(format))
            throw ApiError.badRequest("response_format must be json, text, verbose_json, srt or vtt", "response_format");
        String language = parts.containsKey("language") && !parts.get("language").text().isBlank() ? parts.get("language").text().strip() : null;
        String prompt = parts.containsKey("prompt") && !parts.get("prompt").text().isBlank() ? parts.get("prompt").text() : null;
        // Decoded, a WAV is floats for two channels: an 8-bit mono file is 8 times its size, more again resampled and
        // mixed down, which {@link #buffering} doesn't count. So one is decoded at a time, and held only while it is
        // transcribed (as each whisper model takes one request at a time anyway); others wait with just their bytes.
        try { transcribing.acquire(); }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ApiError(503, "server_error", "the server is stopping", null, "server_busy");
        }
        Audio audio;
        Transcript t;
        try {
            try { audio = Audio.readWav(file.bytes()); }
            catch (IOException | RuntimeException e) {
                throw ApiError.badRequest("file must be a WAV (" + e.getMessage() + "); convert other formats first, e.g. ffmpeg -i in.mp3 in.wav", "file");
            }
            // it is resampled to whisper's 16 kHz, so a header's rate sizes the work: 1 Hz would be 16000 times the samples
            if (audio.sampleRate() < MIN_SAMPLE_RATE || audio.sampleRate() > MAX_SAMPLE_RATE)
                throw ApiError.badRequest("file's sample rate must be from " + MIN_SAMPLE_RATE + " to " + MAX_SAMPLE_RATE + " Hz, not "
                        + audio.sampleRate(), "file");
            t = engines.transcribe(model, audio, language, prompt);
        } finally { transcribing.release(); }
        String text = t.text().strip();
        switch (format) {
            case "text" -> send(ex, 200, "text/plain; charset=utf-8", (text + "\n").getBytes(StandardCharsets.UTF_8));
            case "srt" -> send(ex, 200, "text/plain; charset=utf-8", t.srt().getBytes(StandardCharsets.UTF_8));
            case "vtt" -> send(ex, 200, "text/vtt; charset=utf-8", t.vtt().getBytes(StandardCharsets.UTF_8));
            case "verbose_json" -> {
                List<Object> segments = new ArrayList<>(), words = new ArrayList<>();
                for (int i = 0; i < t.segments().size(); i++) {
                    var s = t.segments().get(i);
                    segments.add(Json.obj("id", i, "start", s.start(), "end", s.end(), "text", s.text()));
                }
                for (var w : t.words()) words.add(Json.obj("word", w.text().strip(), "start", w.start(), "end", w.end()));
                json(ex, 200, Json.obj("task", "transcribe", "language", t.language(), "duration", audio.seconds(), "text", text,
                        "segments", segments, "words", words));
            }
            default -> json(ex, 200, Json.obj("text", text));
        }
    }

    // ---- requests and replies ----

    private String body(HttpExchange ex) throws IOException { return new String(bytes(ex), StandardCharsets.UTF_8); }

    private byte[] bytes(HttpExchange ex) throws IOException {
        String length = ex.getRequestHeaders().getFirst("Content-Length");
        long declared = -1;
        if (length != null) {
            try { declared = Long.parseLong(length.strip()); }
            catch (NumberFormatException e) { throw ApiError.badRequest("a bad Content-Length", null); }
            if (declared > maxBody) throw tooLarge();
        }
        InputStream in = ex.getRequestBody();   // left open when refused, for error() to read what is left of it
        if (declared >= 0) {   // the room for it all, before a byte is read
            hold(ex, declared);
            try (in) { return in.readNBytes((int) declared); }
        }
        var out = new ByteArrayOutputStream();   // chunked: the room for each piece as it comes
        byte[] buf = new byte[64 << 10];
        for (int n; (n = in.read(buf)) != -1; ) {
            if (out.size() + (long) n > maxBody) throw tooLarge();
            hold(ex, out.size() + (long) n);
            out.write(buf, 0, n);
        }
        in.close();
        return out.toByteArray();
    }

    /** Takes room for {@code total} bytes of this request's body in all, or refuses it (503) when the server holds its fill. */
    private void hold(HttpExchange ex, long total) {
        int more = (int) (kb(total) - held.getOrDefault(ex, 0));
        if (more <= 0) return;
        if (!buffering.tryAcquire(more))
            throw new ApiError(503, "server_error", "the server is busy with other requests; retry shortly", null, "server_busy");
        held.merge(ex, more, Integer::sum);
    }

    private static long kb(long bytes) { return (bytes + 1023) >> 10; }

    /** The request body held now, in KB, across every request in hand: for tests, to wait on rather than on time. */
    int bufferedKb() { return bufferingKb - buffering.availablePermits(); }

    private ApiError tooLarge() {
        return new ApiError(413, "invalid_request_error", "the request is larger than " + maxBody / (1 << 20) + " MB", null, "request_too_large");
    }

    private void json(HttpExchange ex, int status, Object body) throws IOException {
        send(ex, status, "application/json", Json.write(body).getBytes(StandardCharsets.UTF_8));
    }

    private void send(HttpExchange ex, int status, String type, byte[] body) throws IOException {
        release(ex);
        ex.getResponseHeaders().set("Content-Type", type);
        boolean head = ex.getRequestMethod().equals("HEAD");
        ex.sendResponseHeaders(status, head ? -1 : body.length);
        if (!head) try (OutputStream out = ex.getResponseBody()) { out.write(body); }
    }

    /**
     * Gives back the room this request's body held, before its reply goes out: a client that has its reply may send its
     * next request at once, and must find the room free (it was given back only after the reply, and so not yet).
     */
    private void release(HttpExchange ex) {
        Integer kb = held.remove(ex);
        if (kb != null) buffering.release(kb);
    }

    /** The most of an unread request body read (and dropped) before an error reply; the JDK's own drain is the same. */
    static final int DRAIN_BYTES = 64 << 10;

    /**
     * Sends the error, unless a reply has already started; returns the status. A body the request still has unread is
     * read first, up to {@link #DRAIN_BYTES}: a reply sent over unread bytes can be lost to the reset the connection's
     * close then sends, so the client would see a reset instead of the 403 or 413. Past that, the connection closes
     * after the reply ({@code Connection: close}), and a client still sending may see the reset.
     */
    private int error(HttpExchange ex, ApiError e) {
        try {
            if (ex.getResponseCode() == -1) {
                if (!drained(ex)) ex.getResponseHeaders().set("Connection", "close");
                if (e.status == 503) ex.getResponseHeaders().set("Retry-After", "1");
                json(ex, e.status, e.body());
            }
        } catch (IOException ignored) { /* the client has gone */ }
        return e.status;
    }

    /** Reads what is left of the request body, up to {@link #DRAIN_BYTES}; whether it all was. */
    static boolean drained(HttpExchange ex) {
        try {
            InputStream in = ex.getRequestBody();
            byte[] buf = new byte[8192];
            long read = 0;
            for (int n; (n = in.read(buf)) != -1; ) {
                read += n;
                if (read > DRAIN_BYTES) return false;
            }
            return true;
        } catch (IOException e) {
            return true;   // read whole already (and closed), or the client has gone: nothing to wait for
        }
    }

    private static byte[] randomBytes(int n) {
        byte[] b = new byte[n];
        ThreadLocalRandom.current().nextBytes(b);
        return b;
    }

    /** What runs the language and embedding models. Images, speech and transcription always run on their own engines. */
    public enum LanguageEngine {
        /** llama.cpp (fornax-llama): every feature, vision models included. The default. */
        LLAMA_CPP,
        /**
         * jitllm (fornax-jitllm), a Java engine on the GPU through TornadoVM: chat, streaming and embeddings. A request
         * with tools, a JSON schema or images is refused (400); vision models are llama.cpp's; one request at a time
         * per model; {@code min_p} and {@code top_k} are ignored (it samples with temperature and top-p). Needs a
         * TornadoVM SDK ({@code TORNADOVM_HOME}).
         */
        JITLLM
    }

    /** How the server is set up. Every setting has a default: {@code FornaxServer.builder().build().start()} serves on 127.0.0.1:54154. */
    public static final class Builder {
        private String host = "127.0.0.1";
        private int port = DEFAULT_PORT;
        private String apiKey;
        private final Set<String> cors = new LinkedHashSet<>();
        private long budgetBytes = 24L << 30;
        private Duration idleTimeout = Duration.ofMinutes(15);
        private int contextSize = 8192;
        private int parallel = 1;
        private FornaxHub hub;
        private PullListener pulls = PullListener.logging();
        private Consumer<String> log = logger();
        private final List<String> models = new ArrayList<>();
        private long maxBodyBytes = 64L << 20;
        private long maxBufferedBytes = 256L << 20;
        private boolean allowDownloads;
        private LanguageEngine languageEngine = LanguageEngine.LLAMA_CPP;
        private Engines engines;

        private Builder() {}

        /** The address to listen on (default 127.0.0.1). Another than loopback lets other machines call it: set an API key. */
        public Builder host(String host) { this.host = Objects.requireNonNull(host, "host"); return this; }

        /** The port (default {@value #DEFAULT_PORT}; 0 picks a free one, which {@link #port()} gives). */
        public Builder port(int port) {
            if (port < 0 || port > 65535) throw new IllegalArgumentException("port must be from 0 to 65535, not " + port);
            this.port = port;
            return this;
        }

        /** The key every request must carry as {@code Authorization: Bearer <key>}; null or blank for none (the default). */
        public Builder apiKey(String key) { this.apiKey = key; return this; }

        /**
         * An origin a web page may call from ({@code https://app.example}, or {@code http://localhost:5173}), which then
         * gets CORS headers; {@code *} allows any. Without one, a request from a web page is refused.
         */
        public Builder corsOrigin(String origin) {
            String o = Objects.requireNonNull(origin, "origin").strip();
            if (o.endsWith("/")) o = o.substring(0, o.length() - 1);
            cors.add(o);
            return this;
        }

        /** The model pool's budget: loaded models' sizes, in bytes (default 24 GB; more than 0). */
        public Builder budgetBytes(long bytes) {
            if (bytes <= 0) throw new IllegalArgumentException("budgetBytes must be more than 0, not " + bytes);
            this.budgetBytes = bytes;
            return this;
        }

        /** How long an unused model stays loaded (default 15 minutes; null keeps it until the budget needs the room). */
        public Builder idleTimeout(Duration timeout) {
            if (timeout != null && !timeout.isPositive())
                throw new IllegalArgumentException("idleTimeout must be more than 0 (null never unloads an idle model), not " + timeout);
            this.idleTimeout = timeout;
            return this;
        }

        /** Context size in tokens for language and embedding models (default 8192; 0 for the model's own). */
        public Builder contextSize(int tokens) {
            if (tokens < 0) throw new IllegalArgumentException("contextSize must be 0 or more, not " + tokens);
            this.contextSize = tokens;
            return this;
        }

        /**
         * How many chat requests each language model runs at once (default 1). More than 1 loads it with that many
         * slots, which decode together in one batch (continuous batching), so concurrent clients share the GPU instead
         * of queueing; each slot has {@link #contextSize} tokens of its own, so the context takes that many times the
         * memory, and the pool's budget counts it (so {@code contextSize} must not be 0; {@link #build} refuses it).
         * A streaming client that falls a whole buffer behind is stopped rather than waited for, which would stall
         * every slot. Vision models, which a model with slots can't take images in yet, still run one request at a time.
         */
        public Builder parallel(int requests) {
            if (requests < 1 || requests > 256) throw new IllegalArgumentException("parallel must be from 1 to 256, not " + requests);
            this.parallel = requests;
            return this;
        }

        /** Where models are found and downloaded (default {@link FornaxHub#standard()}). */
        public Builder hub(FornaxHub hub) { this.hub = hub; return this; }

        /** Download progress, for a model downloaded at start or on a request's first use (default: logged). */
        public Builder pullListener(PullListener listener) { this.pulls = Objects.requireNonNull(listener); return this; }

        /** A line per request, and models loading (default: {@code System.Logger} {@code org.liberationdata.fornax.server}, INFO). */
        public Builder log(Consumer<String> log) { this.log = Objects.requireNonNull(log); return this; }

        /**
         * A model to download and load at start: a catalogue alias ({@code qwen3-4b}), an {@code org/name[:TAG]}
         * reference or a path. The first chat model answers requests that name none. A reference or path whose name
         * says "embed" is loaded for embeddings; any other one for chat.
         */
        public Builder model(String model) { models.add(Objects.requireNonNull(model, "model").strip()); return this; }

        /**
         * Whether a request may name any model Fornax can load (a catalogue alias, an {@code org/name[:TAG]} reference
         * or a path), downloading it the first time (default false: only the {@link #model} models and catalogue models
         * already downloaded, as anyone who can call the server could otherwise fill its disk).
         */
        public Builder allowDownloads(boolean allow) { this.allowDownloads = allow; return this; }

        /** What runs the language and embedding models (default {@link LanguageEngine#LLAMA_CPP}). */
        public Builder languageEngine(LanguageEngine engine) { this.languageEngine = Objects.requireNonNull(engine, "engine"); return this; }

        /** The largest request body taken (default 64 MB: an image or a WAV file in it). */
        public Builder maxBodyBytes(long bytes) {
            if (bytes <= 0) throw new IllegalArgumentException("maxBodyBytes must be more than 0, not " + bytes);
            this.maxBodyBytes = bytes;
            return this;
        }

        /**
         * The most request body held in memory at once, across every request in hand (default 256 MB; at least
         * {@link #maxBodyBytes}). A request's body is read whole before it waits for its model, and kept until it is
         * answered, so many large ones at once could fill the heap: past this, a request is answered
         * {@code 503 server_busy} with {@code Retry-After} rather than queued (with a {@code Content-Length},
         * before any of its body is read).
         */
        public Builder maxBufferedBytes(long bytes) {
            if (bytes <= 0) throw new IllegalArgumentException("maxBufferedBytes must be more than 0, not " + bytes);
            this.maxBufferedBytes = bytes;
            return this;
        }

        /** A test's engines instead of Fornax's. */
        Builder engines(Engines engines) { this.engines = engines; return this; }

        /** Binds the port (failing here if it is in use); {@link FornaxServer#start} downloads, loads and serves. */
        public FornaxServer build() throws IOException {
            // here, before the server makes its model pool, whose reaper thread a refused build would leave running
            if (engines == null && parallel > 1 && contextSize == 0)
                throw new IllegalArgumentException("parallel needs each slot's context size: set contextSize (not 0)");
            if (engines == null && parallel > 1 && languageEngine == LanguageEngine.JITLLM)
                throw new IllegalArgumentException("parallel is for llama.cpp; jitllm answers one request at a time per model");
            // jitllm's TornadoVM SDK is checked before any model is downloaded
            if (engines == null && languageEngine == LanguageEngine.JITLLM)
                org.liberationdata.fornax.jitllm.JitLlmModel.checkRuntime();
            return new FornaxServer(this);
        }

        private static Consumer<String> logger() {
            System.Logger l = System.getLogger("org.liberationdata.fornax.server");
            return s -> l.log(System.Logger.Level.INFO, s);
        }
    }
}
