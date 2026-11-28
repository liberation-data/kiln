package org.liberationdata.fornax.studio;

import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.hub.GgufFiles;
import org.liberationdata.fornax.llama.LlamaModel;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * What in the Hugging Face cache Fornax can run, as the studio offers it:
 * <ul>
 *   <li><b>chat</b>: every GGUF language model in the cache (llama.cpp), found by scanning it, but the speech-only models
 *       (whose mmproj makes or transcribes audio, and doesn't also chat);</li>
 *   <li><b>image</b>, <b>song</b> and <b>video</b>: the presets the Fornax examples use (stable-diffusion.cpp, ACE-Step,
 *       YuE2, LTX-2.5), each a set of files, available when all of them are downloaded;</li>
 *   <li><b>vision</b>: the vision language models Fornax knows by name ({@link LlamaModel.Weights}), each a GGUF and its
 *       mmproj (llama.cpp's mtmd), for the Inspect window.</li>
 * </ul>
 * Read afresh on each call, so a model downloaded while the studio runs shows up on the next refresh.
 */
final class Catalog {

    enum Kind { chat, image, song, video, vision }

    /** One file a model needs: which part it is, where on the Hub it comes from, and where it is now (or null). */
    record Part(String role, String repo, String file, Path path) {
        boolean present() { return path != null; }
    }

    /**
     * A model the studio can load. {@code defaults} are the generation settings the preset suggests (steps, guidance,
     * size, ...), which the page fills its form with. {@code license} is said where the weights' license limits what
     * may be done with what they make (null otherwise).
     */
    record Model(String id, Kind kind, String name, String engine, List<Part> parts, Map<String, Object> defaults, String license) {
        boolean available() { return parts.stream().allMatch(Part::present); }

        Model licensed(String license) { return new Model(id, kind, name, engine, parts, defaults, license); }

        /** The same model loaded without one of its parts (LTX without its audio VAE), as a model of its own in the pool. */
        Model without(String role, String idSuffix, String nameSuffix) {
            return new Model(id + idSuffix, kind, name + nameSuffix, engine,
                    parts.stream().filter(p -> !p.role().equals(role)).toList(), defaults, license);
        }

        long sizeBytes() {
            long n = 0;
            for (Part p : parts) {
                if (!p.present()) continue;
                try { n += Files.size(p.path()); } catch (IOException e) { throw new UncheckedIOException(e); }
            }
            return n;
        }

        Path path(String role) {
            return parts.stream().filter(p -> p.role().equals(role)).findFirst().map(Part::path).orElse(null);
        }
    }

    static final String YUE2 = "Serveurperso/YuE2-GGUF";
    /** YuE2's weights' license, as Fornax's README and website say it. */
    static final String YUE2_LICENSE = "CC BY-NC 4.0: no commercial use of the weights or of what they make";

    /** MedGemma's weights' terms: not research-only; use in diagnosis or treatment may need a regulator's authorization (HAI-DEF terms, 3.1). */
    static final String MEDGEMMA_LICENSE = "Health AI Developer Foundations terms: diagnosing or treating patients with it may need a health regulator's authorization";

    /** The vision language models the Inspect window offers, as Fornax names them; the first is its default. */
    static final List<LlamaModel.Weights> VISION = List.of(
            LlamaModel.Weights.QWEN3_VL_4B, LlamaModel.Weights.PADDLEOCR_VL, LlamaModel.Weights.MEDGEMMA_4B, LlamaModel.Weights.OCRIA_1);

    // GGUFs in the cache that aren't chat models: the song engines' own weights, ERNIE-Image's diffusion model, embedders,
    // VAEs and vision projectors, and the vision models, which the Inspect window loads with their projectors.
    private static final Set<String> NOT_CHAT_REPOS = Set.of("Serveurperso/ACE-Step-1.5-GGUF", YUE2, "unsloth/ERNIE-Image-Turbo-GGUF",
            LlamaModel.Weights.QWEN3_VL_4B.repo, LlamaModel.Weights.PADDLEOCR_VL.repo, LlamaModel.Weights.MEDGEMMA_4B.repo,
            LlamaModel.Weights.OCRIA_1.repo);
    private static final Pattern NOT_CHAT_FILE = Pattern.compile("(?i).*(embed|vae|mmproj|reranker).*");
    // Of a split GGUF, the first shard names the model (llama.cpp loads the rest beside it).
    private static final Pattern SHARD = Pattern.compile("(?i)(.*)-(\\d{5})-of-(\\d{5})\\.gguf");

    private final Path root;
    private final List<Path> folders;

    Catalog(Path root) { this(root, List.of()); }

    /**
     * Model folders are searched after the Hugging Face cache, each for a part's file at its path in the repo (as
     * ComfyUI's models folder keeps them: diffusion_models/…, text_encoders/…, vae/…).
     */
    Catalog(Path root, List<Path> folders) { this.root = root; this.folders = List.copyOf(folders); }

    /** The Hugging Face cache, then the folders in fornax.studio.modelDirs or FORNAX_STUDIO_MODEL_DIRS (path-separated). */
    Catalog() { this(HfCache.root(), modelFolders()); }

    static List<Path> modelFolders() {
        String v = System.getProperty("fornax.studio.modelDirs", System.getenv("FORNAX_STUDIO_MODEL_DIRS"));
        if (v == null || v.isBlank()) return List.of();
        return java.util.Arrays.stream(v.split(java.io.File.pathSeparator)).filter(d -> !d.isBlank()).map(Path::of).toList();
    }

    Path root() { return root; }

    /** The model folders searched after the cache. */
    List<Path> folders() { return folders; }

    List<Model> models() {
        List<Model> all = new ArrayList<>(chatModels());
        all.addAll(imagePresets());
        all.addAll(songPresets());
        all.addAll(videoPresets());
        all.addAll(visionPresets());
        return all;
    }

    Optional<Model> find(String id) { return models().stream().filter(m -> m.id().equals(id)).findFirst(); }

    // ---- chat: scanned ----

    List<Model> chatModels() {
        List<Model> out = new ArrayList<>();
        if (!Files.isDirectory(root)) return out;
        try (Stream<Path> repos = Files.list(root)) {
            for (Path repoDir : repos.filter(p -> p.getFileName().toString().startsWith("models--")).sorted().toList()) {
                String repo = repoDir.getFileName().toString().substring("models--".length()).replace("--", "/");
                if (NOT_CHAT_REPOS.contains(repo)) continue;
                Path snapshots = repoDir.resolve("snapshots");
                if (!Files.isDirectory(snapshots)) continue;
                // The same file in several snapshots is one model: the first snapshot to have it.
                Map<String, Path> files = new LinkedHashMap<>();
                try (Stream<Path> snaps = Files.list(snapshots)) {
                    for (Path snap : snaps.sorted().toList()) {
                        try (Stream<Path> walk = Files.walk(snap)) {
                            walk.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".gguf"))
                                .filter(Files::isRegularFile)  // follows the cache's symlinks into blobs
                                .sorted(Comparator.comparing(Path::toString))
                                .forEach(p -> files.putIfAbsent(snap.relativize(p).toString(), p));
                        }
                    }
                }
                // A speech model (text to speech, speech recognition, voice cloning) is a GGUF beside a projector that
                // makes or transcribes audio; Chat can't talk to it, though its architecture may be a chat model's.
                // An audio chat model, which hears or speaks and also chats, stays.
                if (files.entrySet().stream().anyMatch(e -> GgufFiles.isAuxiliary(e.getKey()) && GgufHeader.onlySpeech(e.getValue()))) continue;
                for (var e : files.entrySet()) {
                    String file = e.getKey();
                    String base = Path.of(file).getFileName().toString();
                    if (NOT_CHAT_FILE.matcher(base).matches()) continue;
                    var shard = SHARD.matcher(base);
                    List<Part> parts = new ArrayList<>();
                    if (shard.matches()) {
                        if (!shard.group(2).equals("00001")) continue;
                        int n = Integer.parseInt(shard.group(3));
                        String dir = file.contains("/") ? file.substring(0, file.lastIndexOf('/') + 1) : "";
                        for (int k = 1; k <= n; k++) {
                            String f = dir + String.format("%s-%05d-of-%05d.gguf", shard.group(1), k, n);
                            parts.add(new Part(k == 1 ? "gguf" : "shard", repo, f, files.get(f)));
                        }
                    } else {
                        parts.add(new Part("gguf", repo, file, e.getValue()));
                    }
                    String name = base.replaceAll("(?i)(-\\d{5}-of-\\d{5})?\\.gguf$", "");
                    out.add(new Model("chat:" + repo + "/" + file, Kind.chat, name, "llama.cpp", parts,
                            Json.obj("maxTokens", 1024L, "temperature", 0.7, "contextSize", 8192L), null));
                }
            }
        } catch (IOException e) { throw new UncheckedIOException(e); }
        return out;
    }

    // ---- image, song and video: presets, as the Fornax examples have them ----

    List<Model> imagePresets() {
        String qwenVl = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
        return List.of(
                preset("image:flux2-klein", Kind.image, "FLUX.2 [klein] base 4B", "stable-diffusion.cpp",
                        Json.obj("width", 768L, "height", 768L, "steps", 12L, "cfg", 4.0),
                        part("diffusion_model", "black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors"),
                        part("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
                        part("llm", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors")),
                preset("image:qwen-image-2512", Kind.image, "Qwen-Image 2512", "stable-diffusion.cpp",
                        Json.obj("width", 1024L, "height", 1024L, "steps", 20L, "cfg", 2.5),
                        part("diffusion_model", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"),
                        part("vae", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
                        anyPart("llm", "Comfy-Org/Qwen-Image_ComfyUI", qwenVl, "Comfy-Org/HunyuanVideo_1.5_repackaged", qwenVl)),
                // distilled: 8 steps, no guidance
                preset("image:z-image-turbo", Kind.image, "Z-Image Turbo", "stable-diffusion.cpp",
                        Json.obj("width", 1024L, "height", 1024L, "steps", 8L, "cfg", 1.0),
                        part("diffusion_model", "Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
                        part("vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
                        part("llm", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors")),
                preset("image:ernie-image-turbo", Kind.image, "ERNIE-Image-Turbo", "stable-diffusion.cpp",
                        Json.obj("width", 1024L, "height", 1024L, "steps", 8L, "cfg", 1.0),
                        part("diffusion_model", "unsloth/ERNIE-Image-Turbo-GGUF", "ernie-image-turbo-Q8_0.gguf"),
                        part("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
                        part("llm", "unsloth/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q8_0.gguf")));
    }

    List<Model> songPresets() {
        String ace = "Serveurperso/ACE-Step-1.5-GGUF";
        Map<String, Object> song = Json.obj("duration", 60L, "bpm", 96L, "keyscale", "G major", "steps", 8L, "cfg", 1.0, "shift", 3.0);
        return List.of(
                preset("song:acestep-turbo", Kind.song, "ACE-Step 1.5 turbo", "acestep.cpp", song,
                        part("text_encoder", ace, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                        part("dit", ace, "acestep-v15-turbo-Q8_0.gguf"),
                        part("vae", ace, "vae-BF16.gguf")),
                preset("song:acestep-turbo-lm4b", Kind.song, "ACE-Step 1.5 turbo + 4B LM", "acestep.cpp", song,
                        part("lm", ace, "acestep-5Hz-lm-4B-Q8_0.gguf"),
                        part("text_encoder", ace, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                        part("dit", ace, "acestep-v15-turbo-Q8_0.gguf"),
                        part("vae", ace, "vae-BF16.gguf")),
                preset("song:acestep-xl-turbo", Kind.song, "ACE-Step 1.5 XL turbo", "acestep.cpp", song,
                        part("text_encoder", ace, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                        part("dit", ace, "acestep-v15-xl-turbo-Q8_0.gguf"),
                        part("vae", ace, "vae-BF16.gguf")),
                // Yue2Song's files. YuE2 takes no BPM or key: its style says them, in words.
                preset("song:yue2-3b", Kind.song, "YuE2 3B", "yue2.cpp", Json.obj("duration", 60L),
                        part("model", YUE2, "YuE2-3B-Q8_0.gguf"),
                        part("vae", YUE2, "YuE2-Vae-F32.gguf")).licensed(YUE2_LICENSE));
    }

    /**
     * LTX-2.5 through stable-diffusion.cpp, with VideoClip's files: the transformer, Gemma text encoder, video and audio
     * VAEs, and the x2 latent upscaler its two-stage recipe refines with. Sizes are the draft's; two-stage doubles them.
     */
    List<Model> videoPresets() {
        String ltx = "Lightricks/LTX-2.5";
        List<Model> out = new ArrayList<>();
        for (String variant : List.of("distilled", "dev")) {
            boolean distilled = variant.equals("distilled");
            out.add(preset("video:ltx-2.5-" + variant, Kind.video, "LTX-2.5 22B " + variant, "stable-diffusion.cpp",
                    Json.obj("width", 672L, "height", 448L, "frames", 49L, "fps", 24L, "steps", distilled ? 8L : 20L,
                            "cfg", distilled ? 1.0 : 3.0, "strength", 0.7),
                    part("diffusion_model", ltx, "diffusion_models/ltx-2.5-22b-" + variant + "-transformer-comfy-int8-convrot.safetensors"),
                    part("llm", ltx, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
                    part("vae", ltx, "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
                    part("audio_vae", ltx, "vae/ltx-2.5-audio-vae-bf16.safetensors"),
                    part("upscaler", ltx, "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")));
        }
        return out;
    }

    /**
     * The vision models, each a GGUF and its mmproj. Greedy by default, as reading a document wants. PaddleOCR-VL takes
     * a task prompt ("OCR:"), not a question, and doesn't fill a JSON schema well: its {@code ocr} default says so, and
     * the backend then has another vision model fill the schema from the text it reads.
     */
    List<Model> visionPresets() {
        List<Model> out = new ArrayList<>();
        for (var w : VISION) {
            boolean ocr = w == LlamaModel.Weights.PADDLEOCR_VL;
            var defaults = Json.obj("maxTokens", 1024L, "temperature", 0.0, "contextSize", 8192L);
            if (ocr) defaults.put("ocr", true);
            // OCRia-1 writes the page twice with thinking on (inside its think block, then again); a page takes more context
            if (w == LlamaModel.Weights.OCRIA_1) {
                defaults.put("thinking", false);
                defaults.put("prompt", "Transcribe this page in markdown.");
                defaults.put("maxTokens", 4096L);
                defaults.put("contextSize", 16384L);
            }
            var m = preset(visionId(w), Kind.vision, switch (w) {
                        case QWEN3_VL_4B -> "Qwen3-VL 4B";
                        case PADDLEOCR_VL -> "PaddleOCR-VL 1.6";
                        case MEDGEMMA_4B -> "MedGemma 1.5 4B";
                        case OCRIA_1 -> "OCRia-1 4B";
                        default -> w.name();
                    }, "llama.cpp", defaults, part("gguf", w.repo, w.modelFile), part("mmproj", w.repo, w.mmprojFile));
            out.add(w == LlamaModel.Weights.MEDGEMMA_4B ? m.licensed(MEDGEMMA_LICENSE)
                    : w == LlamaModel.Weights.OCRIA_1 ? m.licensed(org.liberationdata.fornax.hub.ModelCatalog.OCRIA_LICENSE) : m);
        }
        return out;
    }

    /** "vision:qwen3-vl-4b", from QWEN3_VL_4B. */
    static String visionId(LlamaModel.Weights w) { return "vision:" + w.name().toLowerCase().replace('_', '-'); }

    private static Model preset(String id, Kind kind, String name, String engine, Map<String, Object> defaults, Part... parts) {
        return new Model(id, kind, name, engine, List.of(parts), defaults, null);
    }

    private Part part(String role, String repo, String file) {
        Path p = find(repo, file);
        if (p == null) p = inFolders(file);
        return new Part(role, repo, file, p);
    }

    private Path inFolders(String file) {
        for (Path d : folders) {
            Path p = d.resolve(file);
            if (Files.isRegularFile(p)) return p;
        }
        return null;
    }

    /** The first of two places a file may be in, as HfCache.requireAny has it. */
    private Part anyPart(String role, String repo1, String file1, String repo2, String file2) {
        Path p = find(repo1, file1);
        if (p != null) return new Part(role, repo1, file1, p);
        Path q = find(repo2, file2);
        return q != null ? new Part(role, repo2, file2, q) : new Part(role, repo1, file1, null);
    }

    // HfCache.find, on this catalogue's root (HfCache reads its root from the environment, which a test can't set).
    private Path find(String repo, String file) {
        Path snapshots = root.resolve("models--" + repo.replace("/", "--")).resolve("snapshots");
        if (!Files.isDirectory(snapshots)) return null;
        try (Stream<Path> s = Files.list(snapshots)) {
            return s.sorted().map(snap -> snap.resolve(file)).filter(Files::exists).findFirst().orElse(null);
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }
}
