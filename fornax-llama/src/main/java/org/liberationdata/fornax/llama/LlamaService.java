package org.liberationdata.fornax.llama;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.Scored;
import org.liberationdata.fornax.Wire;
import org.liberationdata.fornax.WorkerService;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Embeddings;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * The worker side of an isolated {@link LlamaModel} ({@link LlamaModel#isolated}): an in-process LlamaModel in the
 * worker process, and the encoding of its calls. Not for direct use; the worker creates it by name.
 *
 * <p>Calls go straight to the model's own methods with the worker's {@link Call}, which the worker's ModelThread runs
 * one at a time: its token upcalls poll that Call, so a cancel from the parent stops the generation even if it arrives
 * before the generation has started. A model with slots runs that many calls at once (see {@link #lanes()}).
 */
public final class LlamaService implements WorkerService {
    private final LlamaModel model;

    /** config: {@link #config}. */
    public LlamaService(byte[] config) {
        Wire.In in = new Wire.In(config);
        String gguf = in.string(), mmproj = in.string();
        Mode mode = Mode.valueOf(in.string());
        int gpuLayers = in.integer(), contextSize = in.integer();
        model = LlamaModel.inProcess(gguf, mmproj, mode, gpuLayers, contextSize, in.integer());
    }

    /** A model with slots batches its calls: as many run at once, each stopping through its own Call. */
    @Override public int lanes() { return model.slots(); }

    @Override public byte[] call(String method, byte[] args, Call call, Consumer<byte[]> events) {
        Wire.In in = new Wire.In(args == null ? new byte[0] : args);
        return switch (method) {
            case "info" -> new Wire.Out().bool(model.supportsVision())
                    .integer(model.mode() == Mode.EMBED ? model.embeddingSize() : 0)
                    .bool(model.supportsAudio()).integer(model.audioSampleRate()).toBytes();
            case "chat" -> {
                List<Message> messages = messages(in);
                Options o = options(in);
                yield result(model.chatOn(call, messages, o, in.bool() ? pieces(events) : null));
            }
            case "chatJson" -> {
                String request = in.string();
                List<Image> images = images(in);
                List<Audio> audio = audio(in);
                yield utf8(model.chatJsonOn(call, request, images, audio, in.bool() ? pieces(events) : null));
            }
            case "complete" -> {
                String prompt = in.string();
                List<Image> images = images(in);
                Options o = options(in);
                Consumer<String> p = in.bool() ? pieces(events) : null;
                yield result(images.isEmpty() ? model.completeOn(call, prompt, o, p) : model.completeImages(call, prompt, images, o, p));
            }
            case "embed" -> {
                List<String> texts = strings(in);
                String prefix = in.string();
                yield embeddings(model.embedOn(call, texts, new EmbedOptions(prefix, in.bool())));
            }
            case "rerank" -> {
                String query = in.string();
                yield scores(model.rerankOn(call, query, strings(in)));
            }
            default -> throw new UnsupportedOperationException("LlamaService has no method " + method);
        };
    }

    @Override public void close() { model.close(); }

    private static Consumer<String> pieces(Consumer<byte[]> events) { return s -> events.accept(utf8(s)); }

    // ---- the encoding, shared with LlamaModel's isolated side ----

    static byte[] config(String gguf, String mmproj, Mode mode, int gpuLayers, int contextSize, int slots) {
        return new Wire.Out().string(gguf).string(mmproj).string(mode.name()).integer(gpuLayers).integer(contextSize)
                .integer(slots).toBytes();
    }

    /** A message's audio isn't sent: a chat with audio goes to the worker as chatJson (LlamaModel.chatOn). */
    static byte[] chatArgs(List<Message> messages, Options o, boolean stream) {
        Wire.Out out = new Wire.Out().integer(messages.size());
        for (Message m : messages) {
            out.string(m.role()).string(m.content());
            images(out, m.images());
        }
        return options(out, o).bool(stream).toBytes();
    }

    /** audio goes as mono: the parent has already mixed it down and resampled it to the model's rate. */
    static byte[] chatJsonArgs(String request, List<Image> images, List<Audio> audio, boolean stream) {
        Wire.Out out = new Wire.Out().string(request);
        return audio(images(out, images), audio).bool(stream).toBytes();
    }

    static byte[] completeArgs(String prompt, List<Image> images, Options o, boolean stream) {
        Wire.Out out = new Wire.Out().string(prompt);
        return options(images(out, images), o).bool(stream).toBytes();
    }

    static byte[] embedArgs(List<String> texts, EmbedOptions o) {
        return strings(new Wire.Out(), texts).string(o.prefix()).bool(o.normalize()).toBytes();
    }

    static byte[] rerankArgs(String query, List<String> documents) {
        return strings(new Wire.Out().string(query), documents).toBytes();
    }

    /** Embeddings as a count, a length and the floats in one block (a big batch goes through shared memory), then their tokens. */
    static byte[] embeddings(Embeddings e) {
        float[][] v = e.vectors();
        int dim = v.length == 0 ? 0 : v[0].length;
        ByteBuffer floats = ByteBuffer.allocate(Math.multiplyExact(v.length, dim) * Float.BYTES);
        for (float[] x : v) for (float f : x) floats.putFloat(f);
        return new Wire.Out().integer(v.length).integer(dim).bytes(floats.array()).integer(e.tokens()).toBytes();
    }

    static Embeddings decodeEmbeddings(byte[] b) {
        Wire.In in = new Wire.In(b);
        int n = in.integer(), dim = in.integer();
        ByteBuffer floats = ByteBuffer.wrap(in.bytes());
        float[][] v = new float[n][dim];
        for (float[] e : v) for (int i = 0; i < dim; i++) e[i] = floats.getFloat();
        return new Embeddings(v, in.integer());
    }

    static byte[] scores(List<Scored> scores) {
        Wire.Out out = new Wire.Out().integer(scores.size());
        for (Scored s : scores) out.integer(s.index()).floatValue(s.score());
        return out.toBytes();
    }

    static List<Scored> decodeScores(byte[] b) {
        Wire.In in = new Wire.In(b);
        int n = in.integer();
        List<Scored> scores = new ArrayList<>(n);
        for (int i = 0; i < n; i++) scores.add(new Scored(in.integer(), in.floatValue()));
        return scores;
    }

    static Result decodeResult(byte[] b) {
        Wire.In in = new Wire.In(b);
        return new Result(in.string(), in.integer(), in.integer());
    }

    static String decodeString(byte[] b) { return new String(b, StandardCharsets.UTF_8); }

    static byte[] result(Result r) {
        return new Wire.Out().string(r.text()).integer(r.promptTokens()).integer(r.generatedTokens()).toBytes();
    }

    private static byte[] utf8(String s) { return s == null ? new byte[0] : s.getBytes(StandardCharsets.UTF_8); }

    private static Wire.Out images(Wire.Out out, List<Image> images) {
        out.integer(images.size());
        for (Image img : images) out.integer(img.width()).integer(img.height()).bytes(img.rgb());
        return out;
    }

    /** Each as its sample rate and its mono samples, one block of floats (a long clip goes through shared memory). */
    private static Wire.Out audio(Wire.Out out, List<Audio> audio) {
        out.integer(audio.size());
        for (Audio a : audio) {
            float[] mono = a.mono();
            ByteBuffer b = ByteBuffer.allocate(Math.multiplyExact(mono.length, Float.BYTES));
            b.asFloatBuffer().put(mono);
            out.integer(a.sampleRate()).bytes(b.array());
        }
        return out;
    }

    static List<Audio> audio(Wire.In in) {
        int n = in.integer();
        List<Audio> audio = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int rate = in.integer();
            FloatBuffer f = ByteBuffer.wrap(in.bytes()).asFloatBuffer();
            float[] mono = new float[f.remaining()];
            f.get(mono);
            audio.add(Audio.ofMono(mono, rate));
        }
        return audio;
    }

    private static Wire.Out strings(Wire.Out out, List<String> strings) {
        out.integer(strings.size());
        for (String s : strings) out.string(s);
        return out;
    }

    static List<String> strings(Wire.In in) {
        int n = in.integer();
        List<String> strings = new ArrayList<>(n);
        for (int i = 0; i < n; i++) strings.add(in.string());
        return strings;
    }

    static List<Image> images(Wire.In in) {
        int n = in.integer();
        List<Image> images = new ArrayList<>(n);
        for (int i = 0; i < n; i++) images.add(new Image(in.integer(), in.integer(), in.bytes()));
        return images;
    }

    static List<Message> messages(Wire.In in) {
        int n = in.integer();
        List<Message> messages = new ArrayList<>(n);
        for (int i = 0; i < n; i++) messages.add(new Message(in.string(), in.string(), images(in)));
        return messages;
    }

    // Options.thinking isn't sent: a chat with it off goes to the worker as chatJson (LlamaModel.chatOn), and
    // complete has no template to turn it off in.
    private static Wire.Out options(Wire.Out out, Options o) {
        return out.integer(o.maxTokens()).floatValue(o.temperature()).floatValue(o.topP()).floatValue(o.minP()).integer(o.seed());
    }

    static Options options(Wire.In in) {
        return new Options(in.integer(), in.floatValue(), in.floatValue(), in.floatValue(), in.integer());
    }
}
