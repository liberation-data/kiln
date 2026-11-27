package org.liberationdata.fornax.jitllm;

import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.Wire;
import org.liberationdata.fornax.WorkerService;
import org.liberationdata.fornax.jitllm.JitLlmModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * The worker side of an isolated {@link JitLlmModel}: an in-process JitLlmModel in a worker JVM started with
 * TornadoVM's options, and the encoding of its calls. Not for direct use; the worker creates it by name.
 */
public final class JitLlmService implements WorkerService {
    private final JitLlmModel model;

    /** config: {@link #config}. */
    public JitLlmService(byte[] config) {
        Wire.In in = new Wire.In(config);
        String gguf = in.string();
        int contextSize = in.integer();
        model = JitLlmModel.inProcess(gguf, contextSize, Mode.valueOf(in.string()));
    }

    @Override public byte[] call(String method, byte[] args, Call call, Consumer<byte[]> events) {
        Wire.In in = new Wire.In(args == null ? new byte[0] : args);
        return switch (method) {
            case "info" -> new Wire.Out().integer(model.mode() == Mode.EMBED ? model.embeddingSize() : 0).toBytes();
            case "chat" -> {
                List<Message> messages = messages(in);
                Options o = options(in);
                yield result(model.chatOn(call, messages, o, in.bool() ? pieces(events) : null));
            }
            case "chatJson" -> {
                String request = in.string();
                yield utf8(model.chatJsonOn(call, request, in.bool() ? pieces(events) : null));
            }
            case "embed" -> embeddings(model.embedOn(call, strings(in)));
            default -> throw new UnsupportedOperationException("JitLlmService has no method " + method);
        };
    }

    /** The calls run on the worker's thread, not the model's: the engine is stopped directly, mid-prompt too. */
    @Override public void cancel() { model.cancelEngine(); }

    @Override public void close() { model.close(); }

    private static Consumer<String> pieces(Consumer<byte[]> events) { return s -> events.accept(utf8(s)); }

    // ---- the encoding, shared with JitLlmModel's isolated side ----

    static byte[] config(String gguf, int contextSize, Mode mode) {
        return new Wire.Out().string(gguf).integer(contextSize).string(mode.name()).toBytes();
    }

    static byte[] chatArgs(List<Message> messages, Options o, boolean stream) {
        Wire.Out out = new Wire.Out().integer(messages.size());
        for (Message m : messages) out.string(m.role()).string(m.content());
        return out.integer(o.maxTokens()).floatValue(o.temperature()).floatValue(o.topP()).floatValue(o.minP())
                .integer(o.seed()).bool(stream).toBytes();
    }

    static byte[] chatJsonArgs(String request, boolean stream) {
        return new Wire.Out().string(request).bool(stream).toBytes();
    }

    static byte[] embedArgs(List<String> texts) {
        Wire.Out out = new Wire.Out().integer(texts.size());
        for (String s : texts) out.string(s);
        return out.toBytes();
    }

    static byte[] embeddings(float[][] v) {
        int dim = v.length == 0 ? 0 : v[0].length;
        ByteBuffer floats = ByteBuffer.allocate(Math.multiplyExact(v.length, dim) * Float.BYTES);
        for (float[] e : v) for (float x : e) floats.putFloat(x);
        return new Wire.Out().integer(v.length).integer(dim).bytes(floats.array()).toBytes();
    }

    static float[][] decodeEmbeddings(byte[] b) {
        Wire.In in = new Wire.In(b);
        int n = in.integer(), dim = in.integer();
        ByteBuffer floats = ByteBuffer.wrap(in.bytes());
        float[][] v = new float[n][dim];
        for (float[] e : v) for (int i = 0; i < dim; i++) e[i] = floats.getFloat();
        return v;
    }

    static byte[] result(Result r) {
        return new Wire.Out().string(r.text()).integer(r.promptTokens()).integer(r.generatedTokens()).toBytes();
    }

    static Result decodeResult(byte[] b) {
        Wire.In in = new Wire.In(b);
        return new Result(in.string(), in.integer(), in.integer());
    }

    static int decodeInt(byte[] b) { return new Wire.In(b).integer(); }

    static String decodeString(byte[] b) { return new String(b, StandardCharsets.UTF_8); }

    private static byte[] utf8(String s) { return s == null ? new byte[0] : s.getBytes(StandardCharsets.UTF_8); }

    private static List<String> strings(Wire.In in) {
        int n = in.integer();
        List<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(in.string());
        return out;
    }

    private static List<Message> messages(Wire.In in) {
        int n = in.integer();
        List<Message> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(new Message(in.string(), in.string()));
        return out;
    }

    private static Options options(Wire.In in) {
        return new Options(in.integer(), in.floatValue(), in.floatValue(), in.floatValue(), in.integer());
    }
}
