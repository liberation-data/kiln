package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

/** {@code fornax image}: one image from a prompt, with an image model set (stable-diffusion.cpp). */
final class Image {
    static final Set<String> OPTIONS = Set.of("out", "width", "height", "steps", "cfg", "seed", "ref");

    static final String USAGE = """
            usage: fornax image <model> "prompt" [options]

            Draws an image from the prompt and writes it as a PNG.
            <model> is an image alias: %s.
            A model that isn't downloaded yet is pulled first (see fornax list for sizes).

              --out FILE           where to write it (default image.png here, numbered rather than overwrite one;
                                   a folder, or a name ending in /, puts image.png in it)
              --width N            default 1024, or the --ref image's; a multiple of 16
              --height N           default 1024, or the --ref image's; a multiple of 16
              --steps N            default 20 (8 for z-image-turbo and ernie-image-turbo)
              --cfg X              guidance scale (default 4.0 for flux2-klein, 2.5 for the Qwen-Image models, 1.0 for the Turbos)
              --seed N             default random; the seed is printed, to make the same image again
              --ref FILE           an image to edit (qwen-image-edit needs one)""".formatted(Models.ids(Kind.IMAGE));

    /** What to draw, with every default filled in. */
    record Job(String prompt, int width, int height, int steps, double cfg, long seed) {}

    static int run(Args a, FornaxHub hub, PrintStream out, PrintStream err) throws IOException {
        String spec = a.positional(0);
        if (spec == null) throw new UsageException("which model? " + Models.kinds(Kind.IMAGE));
        String prompt = String.join(" ", a.rest(1)).strip();
        if (prompt.isEmpty()) throw new UsageException("what should it draw? Give a prompt");
        BufferedImage ref = a.option("ref", null) == null ? null : read(Path.of(a.option("ref", null)));
        var set = ModelCatalog.byName(spec);   // its defaults, and a missing --ref, before anything is downloaded
        if (set.isPresent() && set.get().kind() == Kind.IMAGE && needsRef(set.get()) && ref == null)
            throw new UsageException(set.get().id() + " edits an image: give it with --ref FILE");
        Job job = job(a, set.filter(m -> m.kind() == Kind.IMAGE).orElse(null), prompt, ref);
        Path file = OutFile.choose(a.option("out", null), Path.of(""), "image.png", ".png");
        var m = Models.fetch(spec, Kind.IMAGE, "image", false, hub, err, Main.terminal());

        boolean terminal = Main.terminal();
        err.println("[fornax] loading " + m.set().id() + ": " + m.set().title());
        long t0 = System.nanoTime();
        try (var model = new DiffusionModel(config(m.files()))) {
            err.printf(Locale.ROOT, "[fornax] loaded in %s%n", Fmt.seconds(secs(t0)));
            if (terminal) DiffusionModel.onProgress((s, n, t) -> {
                err.printf(Locale.ROOT, "\r\033[K[fornax] step %d/%d (%.2fs/step)", s, n, t);
                err.flush();
            });
            long g0 = System.nanoTime();
            var img = model.generate(new DiffusionModel.Request(job.prompt(), job.width(), job.height(), job.steps(), job.cfg(),
                    job.seed(), ref == null ? List.of() : List.of(ref), Map.of())).get(0);
            double s = secs(g0);
            if (terminal) err.print("\r\033[K");
            ImageIO.write(img, "png", file.toFile());
            err.printf(Locale.ROOT, "[fornax] %dx%d, %d steps in %s (%.2fs/step), seed %d%n", job.width(), job.height(),
                    job.steps(), Fmt.seconds(s), s / job.steps(), job.seed());
        } finally {
            DiffusionModel.onProgress((s, n, t) -> {});
        }
        out.println(file);
        return 0;
    }

    /** The options, checked, with the defaults for m (null: a name {@link Models#fetch} will refuse) and the reference. */
    static Job job(Args a, ModelSet m, String prompt, BufferedImage ref) {
        int width = side(a, "width", ref == null ? 1024 : ref.getWidth() / 16 * 16);
        int height = side(a, "height", ref == null ? 1024 : ref.getHeight() / 16 * 16);
        int steps = a.intOption("steps", m == null ? 20 : steps(m));
        if (steps < 1) throw new UsageException("--steps must be at least 1");
        float cfg = a.floatOption("cfg", m == null ? 4.0f : (float) cfg(m));
        if (!(cfg >= 0)) throw new UsageException("--cfg must be 0 or more");
        long seed = a.intOption("seed", ThreadLocalRandom.current().nextInt(Integer.MAX_VALUE));
        return new Job(prompt, width, height, steps, cfg, seed);
    }

    private static int side(Args a, String name, int fallback) {
        int v = a.intOption(name, fallback);
        if (v < 16 || v % 16 != 0) throw new UsageException("--" + name + " must be a multiple of 16, not " + v);
        return v;
    }

    /** The guidance scale the model is tuned for ({@link ModelCatalog#imageCfg}). */
    static double cfg(ModelSet m) { return ModelCatalog.imageCfg(m); }

    /** The steps the model takes ({@link ModelCatalog#imageSteps}). */
    static int steps(ModelSet m) { return ModelCatalog.imageSteps(m); }

    /** Whether the model edits an image rather than drawing one from nothing. */
    static boolean needsRef(ModelSet m) { return m.id().equals("qwen-image-edit"); }

    /** An image set's files as the engine takes them. */
    static DiffusionModel.Config config(Map<String, Path> files) {
        return DiffusionModel.Config.builder().diffusionModel(files.get("diffusion_model").toString())
                .vae(files.get("vae").toString()).llm(files.get("llm").toString()).build();
    }

    private static BufferedImage read(Path file) throws IOException {
        if (!Files.isRegularFile(file)) throw new Main.Failure("no such image: " + file);
        BufferedImage img = ImageIO.read(file.toFile());
        if (img == null) throw new Main.Failure(file + " isn't an image Java can read (PNG, JPEG, BMP or GIF)");
        return img;
    }

    private static double secs(long t) { return (System.nanoTime() - t) / 1e9; }
}
