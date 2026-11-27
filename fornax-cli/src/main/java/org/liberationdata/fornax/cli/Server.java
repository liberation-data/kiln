package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.server.FornaxServer;

import java.io.IOException;
import java.io.PrintStream;
import java.net.BindException;
import java.net.InetAddress;
import java.time.Duration;
import java.util.Map;
import java.util.Set;

/** {@code fornax server}: Fornax's engines behind an OpenAI-compatible HTTP API ({@link FornaxServer}). */
final class Server {
    static final Set<String> OPTIONS = Set.of("model", "port", "host", "api-key", "cors", "budget-gb", "ctx", "idle-minutes", "parallel", "engine");
    static final Set<String> FLAGS = Set.of("allow-downloads");
    static final Map<String, String> SHORT = Map.of("m", "model");

    static final String USAGE = """
            usage: fornax server [-m MODEL]... [options]

            Serves models over an OpenAI-compatible HTTP API, for any OpenAI client, agent framework or tool:
              POST /v1/chat/completions (streaming, tools, JSON schemas, images for a vision model)
              GET  /v1/models
              POST /v1/embeddings          (%s, or an embedding GGUF)
              POST /v1/images/generations  (b64_json PNGs)
              POST /v1/audio/speech        (WAV)
              POST /v1/audio/transcriptions (a WAV file)

            The -m models are downloaded if need be, and loaded, before it starts; the first chat one answers
            requests that name no model. A request may also name a catalogue model already downloaded, which is
            loaded on first use and kept loaded within --budget-gb; any other is refused (404) unless
            --allow-downloads lets a request name any model (an alias, org/name:TAG or a path), downloaded on first use.

              -m, --model M        a model to load at start (repeat for more): an alias, org/name:TAG or a path
              --port N             default %d
              --host ADDR          default 127.0.0.1 (this machine only); 0.0.0.0 for every interface, with an API key
              --api-key K          require Authorization: Bearer K (default FORNAX_API_KEY, if set)
              --cors ORIGIN        let web pages from ORIGIN call it (repeat for more; * for any)
              --budget-gb N        memory for loaded models, by their files' size (default 24)
              --idle-minutes N     unload a model unused this long (default 15; 0 = never)
              --ctx N              context size for language and embedding models (default 8192; 0 = the model's own)
              --parallel N         chat requests each language model answers at once, decoded together in one batch
                                   (default 1); each gets its own --ctx, which --budget-gb counts. Vision models: 1
              --engine E           what runs language and embedding models: llama.cpp (default) or jitllm, a Java
                                   engine on the GPU through TornadoVM (needs TORNADOVM_HOME; no tools, JSON
                                   schemas, images or --parallel)
              --allow-downloads    let a request name any model, downloading it on first use (default: only the -m
                                   models and catalogue models already downloaded)

            curl -s localhost:%d/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'"""
            .formatted(FornaxServer.EMBEDDING_MODEL, FornaxServer.DEFAULT_PORT, FornaxServer.DEFAULT_PORT);

    /**
     * The JDK HTTP server's limits on slow clients, which it reads once per JVM, when its classes load, so they are set
     * as the {@code fornax} command starts ({@link Main#main}), not by the library: a client gets 120 seconds to send its
     * request (headers and body), and the server takes 512 connections at once, a dribbling client otherwise holding a
     * thread and a connection forever. No limit on answering ({@code sun.net.httpserver.maxRspTime}): that clock runs
     * from the request read to the reply's end, so it would cut off a long stream, or a request waiting for a model to
     * load. A {@code -D} in {@code FORNAX_OPTS} or {@code JAVA_OPTS} wins.
     */
    static final Map<String, String> HTTP_LIMITS = Map.of("sun.net.httpserver.maxReqTime", "120", "jdk.httpserver.maxConnections", "512");

    static void httpLimits() {
        HTTP_LIMITS.forEach((k, v) -> { if (System.getProperty(k) == null) System.setProperty(k, v); });
    }

    static int run(Args a, FornaxHub hub, PrintStream out, PrintStream err) throws IOException {
        if (!a.positional().isEmpty()) throw new UsageException("give models with -m (fornax server -m qwen3-4b), not " + a.positional());
        int port = a.intOption("port", FornaxServer.DEFAULT_PORT);
        if (port < 0 || port > 65535) throw new UsageException("--port must be from 0 to 65535");
        String host = a.option("host", "127.0.0.1");
        String key = a.option("api-key", System.getenv("FORNAX_API_KEY"));
        float budget = a.floatOption("budget-gb", 24);
        if (!(budget > 0)) throw new UsageException("--budget-gb must be more than 0");
        int idle = a.intOption("idle-minutes", 15);
        if (idle < 0) throw new UsageException("--idle-minutes must be 0 or more");
        int ctx = a.intOption("ctx", 8192);
        if (ctx < 0) throw new UsageException("--ctx must be 0 or more");
        int parallel = a.intOption("parallel", 1);
        if (parallel < 1 || parallel > 256) throw new UsageException("--parallel must be from 1 to 256");
        if (parallel > 1 && ctx == 0) throw new UsageException("--parallel needs a context size for each request: --ctx N, not 0");
        var engine = Chat.engine(a) == Chat.Engine.JITLLM ? FornaxServer.LanguageEngine.JITLLM : FornaxServer.LanguageEngine.LLAMA_CPP;
        if (parallel > 1 && engine == FornaxServer.LanguageEngine.JITLLM)
            throw new UsageException("--parallel is for llama.cpp; jitllm answers one request at a time per model");
        boolean terminal = Main.terminal();

        var b = FornaxServer.builder().host(host).port(port).apiKey(key).hub(hub).budgetBytes((long) (budget * (1L << 30)))
                .idleTimeout(idle == 0 ? null : Duration.ofMinutes(idle)).contextSize(ctx).parallel(parallel)
                .languageEngine(engine)
                .allowDownloads(a.has("allow-downloads"))
                .pullListener(new PullLine(err, terminal)).log(s -> err.println("[fornax] " + s));
        for (String origin : a.all("cors")) b.corsOrigin(origin);
        for (String spec : a.all("model")) {
            var set = ModelCatalog.byName(spec);
            if (set.isPresent() && (set.get().kind() == Kind.SONG || set.get().kind() == Kind.VIDEO))
                throw new Main.Failure(set.get().id() + " is " + Models.aKind(set.get().kind()) + "; fornax server serves chat, "
                        + "vision, embedding, image, speech and speech-to-text models");
            b.model(spec);
        }
        boolean loopback;
        try { loopback = InetAddress.getByName(host).isLoopbackAddress(); }
        catch (java.net.UnknownHostException e) { throw new Main.Failure("--host " + host + ": no such address"); }
        if (!loopback && (key == null || key.isBlank()))
            err.println("[fornax] warning: listening on " + host + " without an API key: anyone who can reach it can use it (--api-key)");

        // every mistake above, and a port in use, is said before any model is downloaded
        FornaxServer server;
        try { server = b.build(); }
        catch (BindException e) { throw new Main.Failure("can't listen on " + host + ":" + port + " (" + e.getMessage() + "); try another --port"); }
        try {
            for (String spec : a.all("model")) {
                var set = ModelCatalog.byName(spec);   // downloaded first, with its notes; anything else as the server loads it
                if (set.isPresent())
                    Models.fetch(spec, set.get().kind(), "server", set.get().kind() != Kind.IMAGE, hub, err, terminal);
            }
        } catch (IOException | RuntimeException e) { server.close(); throw e; }
        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread(server::close, "fornax-server-shutdown"));
        err.println("[fornax] serving an OpenAI-compatible API at " + server.url() + (key != null && !key.isBlank() ? " (API key required)" : "")
                + "; Ctrl-C stops");
        out.println(server.url());
        out.flush();
        try { server.await(); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        return 0;
    }
}
