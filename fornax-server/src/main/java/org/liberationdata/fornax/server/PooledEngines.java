package org.liberationdata.fornax.server;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelRef;
import org.liberationdata.fornax.hub.ModelReferences;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.PullListener;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.sd.DiffusionModel;
import org.liberationdata.fornax.tts.TtsModel;
import org.liberationdata.fornax.whisper.Transcript;
import org.liberationdata.fornax.whisper.WhisperModel;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Flow;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The server's engines: a model named by a request is found with {@link FornaxHub} (downloaded the first time, when
 * downloads are allowed; otherwise it must be one the server was started with, or a catalogue model already
 * downloaded), and kept loaded in a {@link ModelPool}, which unloads the least recently used to stay within its budget, and idle
 * ones after its timeout. A model has one request at a time; others wait for it. With {@code parallel} above 1, a language
 * model is loaded with that many slots and has that many requests at once ({@link #chatInSlot}), its KV cache counted
 * in its size.
 *
 * <p>A name is a catalogue alias ({@code qwen3-4b}, {@code flux2-klein}, {@code kokoro}, {@code whisper-base.en}), an
 * {@code org/name[:TAG]} reference to one GGUF (or whisper.cpp) file, or a path to one. Embeddings take
 * {@link #EMBEDDING} by name, or a reference or path to an embedding GGUF.
 */
final class PooledEngines implements Engines {
    /** The embedding model known by name. */
    static final String EMBEDDING = "qwen3-embedding-0.6b";
    private static final LlamaModel.Weights EMBEDDING_WEIGHTS = LlamaModel.Weights.QWEN3_EMBEDDING_0_6B;
    /** OpenAI's voice names, which clients send by default: each gets the model's own default voice. */
    static final Set<String> OPENAI_VOICES = Set.of("alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx",
            "sage", "shimmer", "verse", "marin", "cedar");
    private static final Pattern RC = Pattern.compile("\\(rc=(\\d+)\\)");

    private final FornaxHub hub;
    private final ModelPool pool;
    private final List<String> configured;
    private final int contextSize;
    private final int parallel;   // chat requests at once per language model: its slots
    /** Each model with slots' KV cache, by GGUF: read from its header once, not on every request. */
    private final Map<String, Long> kvBytes = new java.util.concurrent.ConcurrentHashMap<>();
    private final PullListener pulls;
    private final Consumer<String> log;
    private final boolean allowDownloads;
    private final Map<String, Found> found = new java.util.concurrent.ConcurrentHashMap<>();

    /** A model name, found: its key in the pool, its files by role, and its catalogue set (null for a file or reference). */
    record Found(String key, String id, Use use, Map<String, Path> files, ModelSet set) {
        Path file(String role) {
            Path p = files.get(role);
            if (p == null) throw ApiError.server(id + " has no " + role + " among " + files.keySet());
            return p;
        }
    }

    /** allowDownloads: whether a request may name a model the server wasn't started with that isn't downloaded. */
    PooledEngines(FornaxHub hub, ModelPool pool, List<String> configured, int contextSize, int parallel, PullListener pulls,
                  Consumer<String> log, boolean allowDownloads) {
        if (parallel > 1 && contextSize <= 0)
            throw new IllegalArgumentException("parallel needs each slot's context size: set contextSize (not 0)");
        this.parallel = parallel;
        this.allowDownloads = allowDownloads;
        this.hub = hub;
        this.pool = pool;
        this.configured = List.copyOf(configured);
        this.contextSize = contextSize;
        this.pulls = pulls;
        this.log = log;
    }

    /** Loads (downloading first if need be) each model the server was started with, for what its name says it is. */
    void warm() {
        for (String name : configured) {
            Found f = find(name, use(name), true);
            long t0 = System.nanoTime();
            lease(f).close();   // loaded, and left in the pool
            log.accept(String.format(Locale.ROOT, "loaded %s in %.1fs", f.id(), (System.nanoTime() - t0) / 1e9));
        }
    }

    /** What a model the server is started with is for: its catalogue kind, else embeddings for a name that says so, else chat. */
    static Use use(String name) {
        var set = ModelCatalog.byName(name);
        if (set.isPresent()) return switch (set.get().kind()) {
            case IMAGE -> Use.IMAGE;
            case SPEECH -> Use.SPEECH;
            case TRANSCRIPTION -> Use.TRANSCRIPTION;
            default -> Use.CHAT;
        };
        String lower = name.toLowerCase(Locale.ROOT);
        if (lower.contains("embed")) return Use.EMBED;
        if (lower.contains("whisper") && !lower.endsWith(".gguf")) return Use.TRANSCRIPTION;
        return Use.CHAT;
    }

    // ---- chat ----

    @Override public void checkChat(String model, boolean images) { check(find(model, Use.CHAT, false), images); }

    private static void check(Found f, boolean images) {
        if (images && (f.set() == null || f.set().kind() != Kind.VISION))
            throw ApiError.badRequest(f.id() + " takes no images; use a vision model (" + ids(Kind.VISION) + ")", "messages");
    }

    @Override public String chat(String model, String requestJson, boolean images, Consumer<String> onContent, Stopper stop) {
        Found f = find(model, Use.CHAT, true);
        check(f, images);
        try (ModelPool.Lease<LlamaModel> lease = lease(f)) {
            LlamaModel m = lease.model();
            if (m.slots() > 1) return chatInSlot(sub -> m.chatJsonAsync(requestJson, List.of(), sub), onContent, stop);
            stop.attach(m::cancel);
            try { return m.chatJson(requestJson, onContent); }
            catch (IllegalStateException e) { throw engineError(e); }
            finally { stop.detach(); }
        }
    }

    /**
     * A chat on a model with slots, where cancel() would stop every slot's request. This one is stopped through its
     * subscription instead (a stop sequence, a client that has gone): the engine then ends just this request with the
     * text so far, as {@code cancel()} ends a serial chat, so the callers see what they would there. The content
     * pieces arrive through a buffer on another thread, so this returns only once the last of them has been
     * delivered: the stream's final chunk must not overtake its content.
     * @param start runs the chat, streaming its content to the subscriber ({@code LlamaModel.chatJsonAsync})
     */
    static String chatInSlot(Function<Flow.Subscriber<String>, CompletableFuture<String>> start, Consumer<String> onContent,
                             Stopper stop) {
        CompletableFuture<Flow.Subscription> subscribed = new CompletableFuture<>();
        CompletableFuture<Void> delivered = new CompletableFuture<>();   // no more pieces will reach onContent
        // attached first, so a stop asked for already cancels the subscription in onSubscribe, before any piece
        stop.attach(() -> subscribed.thenAccept(s -> { s.cancel(); delivered.complete(null); }));
        try {
            var f = start.apply(new Flow.Subscriber<String>() {
                @Override public void onSubscribe(Flow.Subscription s) {
                    subscribed.complete(s);
                    s.request(Long.MAX_VALUE);
                }
                @Override public void onNext(String piece) {
                    if (onContent == null) return;
                    try { onContent.accept(piece); }
                    catch (RuntimeException | Error e) { delivered.complete(null); throw e; }   // cancels the subscription
                }
                @Override public void onError(Throwable t) { delivered.complete(null); }
                @Override public void onComplete() { delivered.complete(null); }
            });
            String reply = f.join();
            delivered.join();
            return reply;
        } catch (CompletionException e) {
            if (e.getCause() instanceof IllegalStateException ise) throw engineError(ise);
            if (e.getCause() instanceof RuntimeException re) throw re;
            throw e;
        } finally { stop.detach(); }
    }

    /**
     * A failure the request caused (rc 2: the prompt is longer than the context; 6: a bad schema; 7: bad input) as a
     * 400; a prompt too long has OpenAI's {@code context_length_exceeded} code, which clients check to trim a conversation.
     */
    static RuntimeException engineError(IllegalStateException e) {
        Matcher m = RC.matcher(String.valueOf(e.getMessage()));
        if (m.find() && Set.of("2", "6", "7").contains(m.group(1))) {
            String msg = e.getMessage().substring(e.getMessage().indexOf("): ") + 3);
            return m.group(1).equals("2") ? new ApiError(400, "invalid_request_error", msg, "messages", "context_length_exceeded")
                    : ApiError.badRequest(msg, null);
        }
        return e;
    }

    // ---- embeddings ----

    @Override public LlamaModel.Embeddings embed(String model, List<String> texts) {
        Found f = find(model, Use.EMBED, true);
        try (ModelPool.Lease<LlamaModel> lease = lease(f)) {
            try { return lease.model().embeddings(texts, LlamaModel.EmbedOptions.DEFAULT); }
            catch (IllegalStateException e) { throw ApiError.badRequest(e.getMessage(), "input"); }   // a text longer than the context
        }
    }

    // ---- images ----

    @Override public List<BufferedImage> images(String model, ImageJob job) {
        Found f = find(model, Use.IMAGE, true);
        if (f.set().id().equals("qwen-image-edit"))
            throw ApiError.badRequest("qwen-image-edit edits an image, which /v1/images/generations doesn't take; use qwen-image, flux2-klein, z-image-turbo or ernie-image-turbo", "model");
        double cfg = job.cfg() != null ? job.cfg() : ModelCatalog.imageCfg(f.set());
        int steps = job.steps() != null ? job.steps() : ModelCatalog.imageSteps(f.set());
        try (ModelPool.Lease<DiffusionModel> lease = lease(f)) {
            List<BufferedImage> out = new ArrayList<>();
            for (int i = 0; i < job.n(); i++)
                out.addAll(lease.model().generate(new DiffusionModel.Request(job.prompt(), job.width(), job.height(), steps,
                        cfg, job.seed() + i)));
            return out;
        }
    }

    // ---- speech ----

    @Override public Audio speak(String model, String text, String voice, float speed) {
        Found f = find(model, Use.SPEECH, true);
        try (ModelPool.Lease<TtsModel> lease = lease(f)) {
            TtsModel tts = lease.model();
            List<String> voices = tts.voices();
            String v = voice == null || OPENAI_VOICES.contains(voice.toLowerCase(Locale.ROOT)) && !voices.contains(voice)
                    ? (voices.isEmpty() ? null : voices.get(0)) : voice;
            if (v != null && !voices.isEmpty() && !voices.contains(v))
                throw ApiError.badRequest(f.id() + " has no voice " + v + "; its voices: " + String.join(", ", voices), "voice");
            TtsModel.Options o = speed > 0 ? TtsModel.Options.DEFAULT.withSpeed(speed) : TtsModel.Options.DEFAULT;
            return tts.speak(text, v, o, null);
        }
    }

    // ---- transcription ----

    @Override public Transcript transcribe(String model, Audio audio, String language, String prompt) {
        Found f = find(model, Use.TRANSCRIPTION, true);
        var o = WhisperModel.Options.DEFAULT.withLanguage(language);
        if (prompt != null) o = o.withInitialPrompt(prompt);
        try (ModelPool.Lease<WhisperModel> lease = lease(f)) {
            return lease.model().transcribe(audio, o);
        }
    }

    // ---- models ----

    @Override public List<Listed> models() {
        Map<String, Listed> out = new LinkedHashMap<>();
        Set<String> loaded = new java.util.HashSet<>();
        for (var e : pool.entries()) if (e.loaded()) loaded.add(e.key());
        for (String name : configured) {
            Use use = use(name);
            String id = canonical(name, use);
            out.putIfAbsent(id, new Listed(id, kind(id, use), loaded.contains(key(use, id))));
        }
        for (String key : loaded) {
            int colon = key.indexOf(':');
            String id = key.substring(colon + 1);
            out.putIfAbsent(id, new Listed(id, kind(id, Use.valueOf(key.substring(0, colon))), true));
        }
        for (ModelSet m : ModelCatalog.all()) {
            if (m.kind() == Kind.SONG || m.kind() == Kind.VIDEO || out.containsKey(m.id())) continue;
            if (hub.status(m).available()) out.put(m.id(), new Listed(m.id(), kind(m.id(), use(m.id())), false));
        }
        if (!out.containsKey(EMBEDDING) && hub.find(EMBEDDING_WEIGHTS.repo, EMBEDDING_WEIGHTS.modelFile).isPresent())
            out.put(EMBEDDING, new Listed(EMBEDDING, "embedding", false));
        return List.copyOf(out.values());
    }

    /** The first model the server was started with for that use, else Fornax's usual one (none for chat or images). */
    @Override public String defaultModel(Use use) {
        return configured.stream().filter(n -> use(n) == use).findFirst().orElse(switch (use) {
            case EMBED -> EMBEDDING;
            case SPEECH -> "kokoro";
            case TRANSCRIPTION -> "whisper-base.en";
            default -> null;
        });
    }

    private static String kind(String name, Use use) {
        if (use == Use.EMBED) return "embedding";
        var set = ModelCatalog.byName(name);
        return set.map(s -> s.kind().name().toLowerCase(Locale.ROOT)).orElse(use.name().toLowerCase(Locale.ROOT));
    }

    @Override public void close() { pool.close(); }

    // ---- finding and loading ----

    private static String canonical(String name) {
        return ModelCatalog.byName(name).map(ModelSet::id).orElse(name.strip());
    }

    /**
     * The name a model is kept by: a catalogue alias as its id, and the embedding model's name as {@link #EMBEDDING},
     * in whatever case they were given (else {@code QWEN3-4B} and {@code qwen3-4b} would be loaded twice); a reference
     * or a path as given, as the Hub and the filesystem may care about its case.
     */
    static String canonical(String name, Use use) {
        if (ModelReferences.isPath(name)) return name;
        if (use == Use.EMBED) return name.equalsIgnoreCase(EMBEDDING) ? EMBEDDING : name;   // aliases aren't embeddings
        return canonical(name);
    }

    private static String key(Use use, String id) { return use + ":" + id; }

    /**
     * The model a request names, for use. With {@code fetch}, its files too, downloaded the first time; without, only
     * checked (no network, nothing loaded), so a stream can still get an error status.
     */
    Found find(String model, Use use, boolean fetch) {
        String name = canonical(model, use);
        Found known = found.get(key(use, name));   // a name found once needs no hub (nor its progress line) again
        if (known != null) return known;
        Found f = locate(name, use, fetch);
        if (fetch) found.put(key(use, name), f);
        return f;
    }

    private Found locate(String name, Use use, boolean fetch) {
        String transcriptionRole = "model";
        if (ModelReferences.isPath(name)) {
            if (use == Use.IMAGE || use == Use.SPEECH)
                throw ApiError.badRequest(name + ": " + what(use) + " needs several files; name one by its alias: " + ids(kindOf(use)), "model");
            if (!servable(name)) throw notServed(name, "isn't one this server was started with");
            return new Found(key(use, name), name, use, Map.of(use == Use.TRANSCRIPTION ? transcriptionRole : "gguf", Path.of(name)), null);
        }
        String ref = name;
        boolean embeddingByName = use == Use.EMBED && name.equalsIgnoreCase(EMBEDDING);
        if (embeddingByName) ref = EMBEDDING_WEIGHTS.repo + ":" + EMBEDDING_WEIGHTS.modelFile;
        var alias = ModelCatalog.byName(name);
        if (alias.isPresent() && use != Use.EMBED) {
            ModelSet m = alias.get();
            if (!fits(m.kind(), use))
                throw ApiError.badRequest(m.id() + " is " + aKind(m.kind()) + "; " + what(use) + " takes " + ids(kindOf(use)), "model");
            if (!servable(name) && !hub.status(m).available()) throw notServed(m.id(), "isn't downloaded");
            return new Found(key(use, m.id()), m.id(), use, fetch ? pull(m) : Map.of(), m);
        }
        if (alias.isPresent())
            throw ApiError.badRequest(alias.get().id() + " is " + aKind(alias.get().kind()) + "; embeddings take " + EMBEDDING
                    + ", or an embedding GGUF by org/name:TAG or path", "model");
        if (ModelRef.tryParse(ref).isEmpty())
            throw ApiError.noSuchModel("no model " + name + ": name a catalogue alias, an org/name:TAG reference or a path (GET /v1/models lists what is here)");
        if (use == Use.IMAGE || use == Use.SPEECH)
            throw ApiError.badRequest(name + ": " + what(use) + " loads a model set from the catalogue (" + ids(kindOf(use)) + "), not a reference", "model");
        if (!servable(name) && !(embeddingByName && hub.find(EMBEDDING_WEIGHTS.repo, EMBEDDING_WEIGHTS.modelFile).isPresent()))
            throw notServed(name, embeddingByName ? "isn't downloaded" : "isn't one this server was started with");
        if (!fetch) return new Found(key(use, name), name, use, Map.of(), null);
        Map<String, Path> files;
        try { files = hub.fetch(ref, pulls); }
        catch (HubException e) {
            if (e.reason() == HubException.Reason.NOT_FOUND) throw ApiError.noSuchModel(name + ": " + e.getMessage());
            throw ApiError.server(name + ": " + e.getMessage());
        } catch (IllegalArgumentException e) { throw ApiError.badRequest(e.getMessage(), "model"); }
        catch (IOException e) { throw ApiError.server(name + ": " + e); }
        String role = use == Use.TRANSCRIPTION ? (files.containsKey("gguf") ? "gguf" : transcriptionRole) : "gguf";
        if (!files.containsKey(role)) throw ApiError.badRequest(name + " isn't a GGUF model (it names " + files.keySet() + ")", "model");
        return new Found(key(use, name), name, use, Map.of(use == Use.TRANSCRIPTION ? transcriptionRole : "gguf", files.get(role)), null);
    }

    /** Whether a request may name this model whether or not it is downloaded: downloads are allowed, or it was started with it. */
    private boolean servable(String name) {
        if (allowDownloads) return true;
        String c = canonical(name);
        for (String n : configured) if (canonical(n).equalsIgnoreCase(c)) return true;
        return false;
    }

    private static ApiError notServed(String name, String why) {
        return ApiError.noSuchModel("no model " + name + " here: it " + why + ", and this server only serves the models it was started "
                + "with and catalogue models already downloaded. Start it with -m " + name + ", or with --allow-downloads to let "
                + "requests name any model (downloaded on first use). GET /v1/models lists what is here");
    }

    private Map<String, Path> pull(ModelSet m) {
        try {
            var status = hub.status(m);
            if (status.available()) return status.paths();
            log.accept("downloading " + m.id() + ": " + m.title());
            return hub.pull(m, pulls);
        } catch (IOException e) { throw ApiError.server(m.id() + ": " + e.getMessage()); }
    }

    /** The model, from the pool: loaded now unless it already is. */
    @SuppressWarnings("unchecked")
    <M extends AutoCloseable> ModelPool.Lease<M> lease(Found f) {
        long size = ModelPool.sizeOf(f.files().values().toArray(Path[]::new));
        int slots = slots(f);
        if (slots > 1) size += kvBytes.computeIfAbsent(f.file("gguf").toString(), g -> LlamaModel.kvCacheBytes(g, contextSize * slots));
        Supplier<AutoCloseable> loader = () -> {
            log.accept("loading " + f.id() + " for " + what(f.use()));
            return load(f);
        };
        try { return (ModelPool.Lease<M>) (ModelPool.Lease<?>) pool.acquire(f.key(), size, slots, loader); }
        catch (UncheckedIOException e) { throw ApiError.server(f.id() + ": " + e.getMessage()); }
    }

    /** A language model's slots: {@link #parallel}, except for a vision model, whose images a model with slots can't take. */
    int slots(Found f) {
        return f.use() == Use.CHAT && !(f.set() != null && f.set().kind() == Kind.VISION) ? parallel : 1;
    }

    private AutoCloseable load(Found f) {
        int slots = slots(f);
        if (slots > 1) return new LlamaModel(f.file("gguf").toString(), -1, contextSize, slots);
        return switch (f.use()) {
            case CHAT -> new LlamaModel(f.file("gguf").toString(),
                    f.set() != null && f.set().kind() == Kind.VISION ? f.file("mmproj").toString() : null, -1, contextSize);
            case EMBED -> new LlamaModel(f.file("gguf").toString(), -1, contextSize, LlamaModel.Mode.EMBED);
            case IMAGE -> new DiffusionModel(DiffusionModel.Config.builder().diffusionModel(f.file("diffusion_model").toString())
                    .vae(f.file("vae").toString()).llm(f.file("llm").toString()).build());
            case SPEECH -> new TtsModel(f.file("gguf").toString(), f.file("mmproj").toString());
            case TRANSCRIPTION -> new WhisperModel(f.file("model").toString(), alignment(f.file("model")), true);
        };
    }

    /** The DTW alignment heads for one of whisper.cpp's own files, else none (as fornax transcribe does). */
    private static WhisperModel.Alignment alignment(Path model) {
        String name = model.getFileName().toString();
        for (WhisperModel.Weights w : WhisperModel.Weights.values()) if (w.file.equals(name)) return w.alignment;
        return WhisperModel.Alignment.NONE;
    }

    private static boolean fits(Kind kind, Use use) {
        return switch (use) {
            case CHAT -> kind == Kind.CHAT || kind == Kind.VISION;
            case IMAGE -> kind == Kind.IMAGE;
            case SPEECH -> kind == Kind.SPEECH;
            case TRANSCRIPTION -> kind == Kind.TRANSCRIPTION;
            case EMBED -> false;
        };
    }

    private static Kind kindOf(Use use) {
        return switch (use) {
            case CHAT, EMBED -> Kind.CHAT;
            case IMAGE -> Kind.IMAGE;
            case SPEECH -> Kind.SPEECH;
            case TRANSCRIPTION -> Kind.TRANSCRIPTION;
        };
    }

    private static String what(Use use) {
        return switch (use) {
            case CHAT -> "chat";
            case EMBED -> "embeddings";
            case IMAGE -> "image generation";
            case SPEECH -> "speech";
            case TRANSCRIPTION -> "transcription";
        };
    }

    /** "a chat model", "an image model". */
    private static String aKind(Kind kind) {
        String k = kind == Kind.TRANSCRIPTION ? "speech-to-text" : kind.name().toLowerCase(Locale.ROOT);
        return ("aeiou".indexOf(k.charAt(0)) >= 0 ? "an " : "a ") + k + " model";
    }

    /** The catalogue's ids of a kind (chat's with vision's): "flux2-klein, qwen-image, qwen-image-edit". */
    private static String ids(Kind kind) {
        return String.join(", ", ModelCatalog.all().stream()
                .filter(m -> m.kind() == kind || kind == Kind.CHAT && m.kind() == Kind.VISION).map(ModelSet::id).toList());
    }
}
