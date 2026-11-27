package org.liberationdata.fornax.jitllm;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.List;

/**
 * The JVM options a worker needs to run jitllm: the TornadoVM SDK's {@code tornado-argfile} (its module path, native
 * library path and module options), the Vector API, and jitllm's own switches. A release SDK that has only
 * {@code tornado-argfile.template} gets one written from it, with the SDK's path in place of {@code ${TORNADOVM_HOME}},
 * into {@code <cache>/tornadovm}: {@code <cache>} is {@code -Dfornax.cache.dir} or {@code FORNAX_CACHE_DIR}, else the
 * user's cache directory ({@code ~/Library/Caches/fornax} on macOS, {@code $XDG_CACHE_HOME/fornax} or
 * {@code ~/.cache/fornax} elsewhere), never a directory other users can write to. The SDK is {@code -Dfornax.tornadovm.home},
 * else {@code TORNADOVM_HOME}; use one built for this JDK and your GPU's backend (CUDA, OpenCL or Metal).
 * Every {@code -Djitllm.*} and {@code -Dtornado.*} property of this JVM goes to the worker too. TornadoVM's device
 * memory budget is {@value #DEVICE_MEMORY}, as the {@code jitllm} launcher's, unless {@code -Dtornado.device.memory}
 * says otherwise.
 */
public final class TornadoJvm {
    /** The worker's {@code tornado.device.memory} when this JVM sets none: the jitllm launcher's default. */
    static final String DEVICE_MEMORY = "14GB";

    private TornadoJvm() {}

    /** The TornadoVM SDK, or an IllegalStateException that says how to set one. */
    public static Path home() {
        String h = System.getProperty("fornax.tornadovm.home");
        if (h == null || h.isBlank()) h = System.getenv("TORNADOVM_HOME");
        if (h == null || h.isBlank())
            throw new IllegalStateException("jitllm needs a TornadoVM SDK: set TORNADOVM_HOME (or -Dfornax.tornadovm.home) "
                    + "to one for this JDK and your GPU (https://github.com/beehive-lab/TornadoVM/releases)");
        Path home = Path.of(h).toAbsolutePath().normalize();
        if (!Files.isRegularFile(home.resolve("tornado-argfile")) && !Files.isRegularFile(home.resolve("tornado-argfile.template")))
            throw new IllegalStateException(home + " has no tornado-argfile: is it a TornadoVM SDK (8.0 or newer)?");
        return home;
    }

    /**
     * The worker's argfile: the SDK's own {@code tornado-argfile}, else one written from
     * {@code tornado-argfile.template} (as the SDK's {@code tornado} launcher does), without the working directory
     * TornadoVM puts first on the module path (see {@link #withoutWorkingDirectory}). It is written into this user's
     * cache directory, as the SDK may not be writable, and created readable by this user only, as the worker's JVM
     * options come from it.
     */
    static Path argfile(Path home) {
        Path own = home.resolve("tornado-argfile");
        try {
            String text = withoutWorkingDirectory(Files.isRegularFile(own) ? Files.readString(own, StandardCharsets.UTF_8)
                    : expand(Files.readString(home.resolve("tornado-argfile.template"), StandardCharsets.UTF_8), home));
            Path dir = privateDirectory(cacheRoot().resolve("tornadovm"));
            Path out = dir.resolve("tornado-argfile-" + Integer.toHexString(home.toString().hashCode()));
            Path tmp = Files.createTempFile(dir, "argfile", ".tmp");
            Files.writeString(tmp, text, StandardCharsets.UTF_8);
            Files.move(tmp, out, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException("can't write a tornado-argfile for " + home + ": " + e.getMessage(), e);
        }
    }

    /**
     * The template with the SDK's path for {@code ${TORNADOVM_HOME}}. A token that then holds whitespace (an SDK under
     * "Tornado SDKs") is quoted as a whole, as an argfile needs, with its backslashes and quotes escaped.
     */
    static String expand(String template, Path home) {
        StringBuilder out = new StringBuilder();
        String[] lines = template.split("\\R", -1);
        for (int n = 0; n < lines.length; n++) {
            String line = lines[n];
            if (n > 0) out.append('\n');
            if (line.stripLeading().startsWith("#") || !line.contains("${TORNADOVM_HOME}")) { out.append(line); continue; }
            String[] tokens = line.strip().split("\\s+");
            for (int i = 0; i < tokens.length; i++) {
                String t = tokens[i].replace("${TORNADOVM_HOME}", home.toString());
                if (i > 0) out.append(' ');
                out.append(t.chars().anyMatch(Character::isWhitespace)
                        ? '"' + t.replace("\\", "\\\\").replace("\"", "\\\"") + '"' : t);
            }
        }
        return out.toString();
    }

    private static final java.util.regex.Pattern WORKING_DIRECTORY =
            java.util.regex.Pattern.compile("(?m)((?:--module-path|-p)(?:[ \\t]+|=)\"?)\\.[:;]");

    /**
     * The argfile without the {@code .} TornadoVM puts first on {@code --module-path}: the worker starts in the
     * application's working directory, and every jar there would be read as a module as it boots, so a stray or
     * duplicate one (an app run from its lib directory) would stop the worker starting.
     */
    static String withoutWorkingDirectory(String argfile) {
        return WORKING_DIRECTORY.matcher(argfile).replaceAll("$1");
    }

    /** The user's Fornax cache directory, as fornax-core's natives cache finds it. */
    static Path cacheRoot() {
        String dir = setting("fornax.cache.dir", "FORNAX_CACHE_DIR");
        if (dir != null) return Path.of(dir);
        String home = System.getProperty("user.home");
        if (System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("mac"))
            return Path.of(home, "Library", "Caches", "fornax");
        String xdg = System.getenv("XDG_CACHE_HOME");
        return xdg != null && !xdg.isBlank() ? Path.of(xdg, "fornax") : Path.of(home, ".cache", "fornax");
    }

    /** dir, created if need be; on a POSIX file system, owner-only. */
    private static Path privateDirectory(Path dir) throws IOException {
        Files.createDirectories(dir);
        try { Files.setPosixFilePermissions(dir, PosixFilePermissions.fromString("rwx------")); }
        catch (UnsupportedOperationException notPosix) { /* Windows: the user's profile is private already */ }
        return dir;
    }

    private static String setting(String property, String env) {
        String v = System.getProperty(property);
        if (v == null || v.isBlank()) v = System.getenv(env);
        return v == null || v.isBlank() ? null : v;
    }

    /** The options for a worker JVM, after Fornax's own. */
    public static List<String> workerArgs() {
        List<String> args = new ArrayList<>();
        args.add("@" + argfile(home()));
        args.add("--add-modules");
        args.add("jdk.incubator.vector");
        args.add("-Dtornado.tvm.maxbytecodesize=" + JitLlmModel.MAX_BYTECODE);
        args.add("-Duse.tornadovm=true");
        if (System.getProperty("tornado.device.memory") == null) args.add("-Dtornado.device.memory=" + DEVICE_MEMORY);
        for (String key : System.getProperties().stringPropertyNames().stream().sorted().toList())
            if (key.startsWith("jitllm.") || key.startsWith("tornado.") && !key.startsWith("tornado.load."))
                args.add("-D" + key + "=" + System.getProperty(key));
        return args;
    }
}
