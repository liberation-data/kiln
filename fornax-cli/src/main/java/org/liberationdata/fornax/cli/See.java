package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;

import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Stream;

/** {@code fornax see}: asks a vision language model about an image, the reply streamed as in {@code fornax chat}. */
final class See {
    static final Set<String> OPTIONS = Set.copyOf(Stream.concat(Chat.OPTIONS.stream(), Stream.of("mmproj")).toList());
    static final Set<String> FLAGS = Chat.FLAGS;
    static final String QUESTION = "Describe this image.";

    static final String USAGE = """
            usage: fornax see <model> <image> ["question"] [options]

            Asks a vision language model about an image (PNG, JPEG, BMP or GIF) and streams its answer.
            The question defaults to "%s" ("OCR:" for paddleocr-vl).
            <model> is a vision alias (%s), or a GGUF by org/name:TAG or path
            with its --mmproj. A model that isn't downloaded yet is pulled first.

              --mmproj M           the model's vision encoder, by org/name:file.gguf or path (an alias brings its own)
              --system TEXT        system prompt
              --max-tokens N       default 1024; 0 = until the model stops or the context fills
              --temperature T      default 0.7; 0 is greedy
              --seed N             default random
              --ctx N              context size in tokens, the image's included (default 8192; 0 = the model's own)
              --gpu-layers N       layers to offload (default -1 = all)
              --think              show the model's reasoning (<think> blocks)""".formatted(QUESTION, Models.ids(Kind.VISION));

    static int run(Args a, FornaxHub hub, PrintStream out, PrintStream err) throws IOException {
        String spec = a.positional(0), image = a.positional(1);
        if (spec == null) throw new UsageException("which model? " + Models.kinds(Kind.VISION));
        if (image == null) throw new UsageException("which image?");
        String question = String.join(" ", a.rest(2)).strip();
        LlamaModel.Image picture = read(Path.of(image));
        var o = Chat.options(a);
        String mmprojSpec = a.option("mmproj", null);
        if (mmprojSpec == null && Models.withoutMmproj(spec))
            throw new UsageException(spec + " needs its mmproj (the vision encoder): --mmproj org/name:mmproj-file.gguf or a path");
        var m = Models.fetch(spec, Kind.VISION, "see", true, hub, err, Main.terminal());
        Path mmproj = mmprojSpec != null ? Models.fetch(mmprojSpec, Kind.VISION, "see --mmproj", true, hub, err, Main.terminal()).file("gguf")
                : m.file("mmproj");

        err.println("[fornax] loading " + m.file("gguf"));
        long t0 = System.nanoTime();
        try (var model = new LlamaModel(m.file("gguf").toString(), mmproj.toString(), a.intOption("gpu-layers", -1), a.intOption("ctx", 8192))) {
            err.printf(Locale.ROOT, "[fornax] loaded in %s%n", Fmt.seconds((System.nanoTime() - t0) / 1e9));
            List<Message> history = new ArrayList<>();
            if (a.option("system", null) != null) history.add(Message.system(a.option("system", null)));
            history.add(Message.user(question.isEmpty() ? question(m.set() == null ? null : m.set().id()) : question, picture));
            Chat.reply(Chat.Model.of(model), history, o, a.has("think"), false, out, err);
        }
        return 0;
    }

    /** The question when none is given: PaddleOCR-VL is trained to read text out ("OCR:"), the others to describe. */
    static String question(String alias) { return alias != null && alias.startsWith("paddleocr") ? "OCR:" : QUESTION; }

    private static LlamaModel.Image read(Path file) {
        if (!Files.isRegularFile(file)) throw new Main.Failure("no such image: " + file);
        try { return LlamaModel.Image.read(file); }
        catch (UncheckedIOException e) { throw new Main.Failure(file + ": " + e.getCause().getMessage()); }
        catch (IllegalArgumentException e) { throw new Main.Failure(file + " isn't an image Java can read (PNG, JPEG, BMP or GIF)"); }
    }
}
