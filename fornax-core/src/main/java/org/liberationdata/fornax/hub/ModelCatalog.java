package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;
import org.liberationdata.fornax.hub.ModelSet.Source;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * The models Fornax's examples, CLI and Studio use, as {@link ModelSet}s. One list, so that what {@code fornax pull} fetches
 * is what Studio offers and what the examples load. The repos and file names match the {@code Weights} enums in the
 * engine modules and the files the examples read.
 *
 * <p>Ids are short names ({@code "qwen3-4b"}, {@code "flux2-klein"}); Fornax Studio's ids ({@code "image:flux2-klein"})
 * are aliases, so either finds the set.
 */
public final class ModelCatalog {
    private ModelCatalog() {}

    /** YuE2's weights' license, as Fornax's README and website say it. */
    public static final String YUE2_LICENSE = "CC BY-NC 4.0: no commercial use of the weights or of what they make";

    /** MedGemma's terms: not research-only, but use in diagnosis or treatment may need a regulator's authorization (HAI-DEF terms, 3.1). */
    public static final String MEDGEMMA_LICENSE =
            "Health AI Developer Foundations terms: diagnosing or treating patients with it may need a health regulator's authorization";

    /** OCRia-1's license (Attachment A): not open source, and closed to larger organisations and to competing services. */
    public static final String OCRIA_LICENSE = "OCRia-1 license, a modified OpenRAIL-M (not open source): only personal or research use "
            + "if your organisation had more than US$2M revenue last year or has raised more than US$2M, and no use in a service "
            + "that competes with LlamaParse or LlamaCloud";

    private static final String ACE = "Serveurperso/ACE-Step-1.5-GGUF", YUE2 = "Serveurperso/YuE2-GGUF", LTX = "Lightricks/LTX-2.5";
    private static final String QWEN_VL_FILE = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
    private static final Part QWEN_VL = Part.anyOf("llm",
            new Source("Comfy-Org/Qwen-Image_ComfyUI", QWEN_VL_FILE),
            new Source("Comfy-Org/HunyuanVideo_1.5_repackaged", QWEN_VL_FILE));
    private static final Part QWEN_IMAGE_VAE = Part.of("vae", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors");

    private static final List<ModelSet> ALL = build();

    /** Every set, in a stable order: by kind, then as listed here. */
    public static List<ModelSet> all() { return ALL; }

    /** The sets of one kind. */
    public static List<ModelSet> of(Kind kind) { return ALL.stream().filter(m -> m.kind() == kind).toList(); }

    /** The set with this id or alias, ignoring case. */
    public static Optional<ModelSet> byName(String name) { return ALL.stream().filter(m -> m.isNamed(name)).findFirst(); }

    /** Like {@link #byName}, but fails with the names there are. */
    public static ModelSet require(String name) {
        return byName(name).orElseThrow(() -> new IllegalArgumentException("No model set named " + name + ". Known: "
                + ALL.stream().map(ModelSet::id).collect(Collectors.joining(", "))));
    }

    /**
     * The guidance scale an image set is tuned for, where a request doesn't say: Qwen-Image 2.5, FLUX.2 [klein] 4.0,
     * Z-Image Turbo and ERNIE-Image-Turbo 1.0 (they are distilled to run without guidance).
     */
    public static double imageCfg(ModelSet m) { return distilled(m) ? 1.0 : m.id().startsWith("qwen") ? 2.5 : 4.0; }

    /** The steps an image set takes, where a request doesn't say: 20, or 8 for the distilled Turbos. */
    public static int imageSteps(ModelSet m) { return distilled(m) ? 8 : 20; }

    private static boolean distilled(ModelSet m) { return m.id().equals("z-image-turbo") || m.id().equals("ernie-image-turbo"); }

    private static List<ModelSet> build() {
        List<ModelSet> l = new ArrayList<>();

        // chat (llama.cpp)
        l.add(ModelSet.of("qwen3-0.6b", Kind.CHAT, "Qwen3 0.6B", "llama.cpp",
                Part.of("gguf", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")));
        l.add(ModelSet.of("qwen3-4b", Kind.CHAT, "Qwen3 4B Instruct 2507", "llama.cpp",
                Part.of("gguf", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")));

        // vision (llama.cpp with mtmd): LlamaModel.Weights
        l.add(ModelSet.of("qwen3-vl-4b", Kind.VISION, "Qwen3-VL 4B", "llama.cpp",
                Part.of("gguf", "Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
                Part.of("mmproj", "Qwen/Qwen3-VL-4B-Instruct-GGUF", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"))
                .withAliases("vision:qwen3-vl-4b"));
        l.add(ModelSet.of("paddleocr-vl", Kind.VISION, "PaddleOCR-VL 1.6", "llama.cpp",
                Part.of("gguf", "PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "PaddleOCR-VL-1.6-GGUF.gguf"),
                Part.of("mmproj", "PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"))
                .withAliases("vision:paddleocr-vl"));
        l.add(ModelSet.of("medgemma-4b", Kind.VISION, "MedGemma 1.5 4B", "llama.cpp",
                Part.of("gguf", "unsloth/medgemma-1.5-4b-it-GGUF", "medgemma-1.5-4b-it-Q4_K_M.gguf"),
                Part.of("mmproj", "unsloth/medgemma-1.5-4b-it-GGUF", "mmproj-F16.gguf"))
                .withLicense(MEDGEMMA_LICENSE).withAliases("vision:medgemma-4b"));
        l.add(ModelSet.of("ocria-1", Kind.VISION, "OCRia-1 4B", "llama.cpp",
                Part.of("gguf", "llamaindex/OCRia-1-GGUF", "OCRia-1-Q8_0.gguf"),
                Part.of("mmproj", "llamaindex/OCRia-1-GGUF", "mmproj-OCRia-1-f16.gguf"))
                .withLicense(OCRIA_LICENSE).withAliases("vision:ocria-1"));

        // speech (llama.cpp text to speech): TtsModel.Weights.KOKORO, the default voice model
        l.add(ModelSet.of("kokoro", Kind.SPEECH, "Kokoro 82M", "llama.cpp",
                Part.of("gguf", "ggml-org/Kokoro-82M-GGUF", "Kokoro-82M-F32.gguf"),
                Part.of("mmproj", "ggml-org/Kokoro-82M-GGUF", "mmproj-Kokoro-82M-F16.gguf")));
        // ZONOS3-Mini (TtsModel.Weights.ZONOS3): 27 voices, the British and Irish English ones among them; 44.1 kHz
        l.add(ModelSet.of("zonos3-mini", Kind.SPEECH, "ZONOS3-Mini", "llama.cpp",
                Part.of("gguf", "ggml-org/ZONOS3-Mini-GGUF", "ZONOS3-Mini-Q8_0.gguf"),
                Part.of("mmproj", "ggml-org/ZONOS3-Mini-GGUF", "mmproj-ZONOS3-Mini-F32.gguf")));
        // VoxCPM2 (TtsModel.Weights.VOXCPM2): no fixed voices; a description in parentheses before the text designs one
        l.add(ModelSet.of("voxcpm2", Kind.SPEECH, "VoxCPM2 2B", "llama.cpp",
                Part.of("gguf", "ggml-org/VoxCPM2-GGUF", "VoxCPM2-Q8_0.gguf"),
                Part.of("mmproj", "ggml-org/VoxCPM2-GGUF", "mmproj-VoxCPM2-BF16.gguf")));

        // speech to text (whisper.cpp): WhisperModel.Weights
        l.add(ModelSet.of("whisper-large-v3-turbo", Kind.TRANSCRIPTION, "Whisper large-v3-turbo (5-bit)", "whisper.cpp",
                Part.of("model", "ggerganov/whisper.cpp", "ggml-large-v3-turbo-q5_0.bin")));
        l.add(ModelSet.of("whisper-base.en", Kind.TRANSCRIPTION, "Whisper base.en", "whisper.cpp",
                Part.of("model", "ggerganov/whisper.cpp", "ggml-base.en.bin")));

        // image (stable-diffusion.cpp)
        l.add(ModelSet.of("flux2-klein", Kind.IMAGE, "FLUX.2 [klein] base 4B", "stable-diffusion.cpp",
                Part.of("diffusion_model", "black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors"),
                Part.of("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
                Part.of("llm", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"))
                .withAliases("image:flux2-klein"));
        l.add(ModelSet.of("qwen-image", Kind.IMAGE, "Qwen-Image 2512", "stable-diffusion.cpp",
                Part.of("diffusion_model", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"),
                QWEN_IMAGE_VAE, QWEN_VL)
                .withAliases("image:qwen-image-2512"));
        l.add(ModelSet.of("qwen-image-edit", Kind.IMAGE, "Qwen-Image-Edit 2511", "stable-diffusion.cpp",
                Part.of("diffusion_model", "Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"),
                QWEN_IMAGE_VAE, QWEN_VL));
        // Z-Image Turbo: distilled, 8 steps at CFG 1; the FLUX.1 VAE, and the Qwen3 4B text encoder klein uses too
        l.add(ModelSet.of("z-image-turbo", Kind.IMAGE, "Z-Image Turbo", "stable-diffusion.cpp",
                Part.of("diffusion_model", "Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
                Part.of("vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
                Part.of("llm", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"))
                .withAliases("image:z-image-turbo"));
        // ERNIE-Image-Turbo (Baidu): distilled, 8 steps at CFG 1; the FLUX.2 VAE klein uses, and a Ministral 3 3B encoder
        l.add(ModelSet.of("ernie-image-turbo", Kind.IMAGE, "ERNIE-Image-Turbo", "stable-diffusion.cpp",
                Part.of("diffusion_model", "unsloth/ERNIE-Image-Turbo-GGUF", "ernie-image-turbo-Q8_0.gguf"),
                Part.of("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
                Part.of("llm", "unsloth/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q8_0.gguf"))
                .withAliases("image:ernie-image-turbo"));

        // songs (acestep.cpp, yue2.cpp). "ace-step" is the CLI's: turbo with the 4B lyric LM.
        Part aceEncoder = Part.of("text_encoder", ACE, "Qwen3-Embedding-0.6B-Q8_0.gguf");
        Part aceVae = Part.of("vae", ACE, "vae-BF16.gguf");
        Part aceLm = Part.of("lm", ACE, "acestep-5Hz-lm-4B-Q8_0.gguf");
        Part turbo = Part.of("dit", ACE, "acestep-v15-turbo-Q8_0.gguf"), xlTurbo = Part.of("dit", ACE, "acestep-v15-xl-turbo-Q8_0.gguf");
        l.add(ModelSet.of("ace-step-turbo", Kind.SONG, "ACE-Step 1.5 turbo", "acestep.cpp", aceEncoder, turbo, aceVae)
                .withAliases("song:acestep-turbo"));
        l.add(ModelSet.of("ace-step", Kind.SONG, "ACE-Step 1.5 turbo + 4B LM", "acestep.cpp", aceLm, aceEncoder, turbo, aceVae)
                .withAliases("song:acestep-turbo-lm4b"));
        l.add(ModelSet.of("ace-step-xl-turbo", Kind.SONG, "ACE-Step 1.5 XL turbo", "acestep.cpp", aceEncoder, xlTurbo, aceVae)
                .withAliases("song:acestep-xl-turbo"));
        l.add(ModelSet.of("ace-step-xl", Kind.SONG, "ACE-Step 1.5 XL turbo + 4B LM", "acestep.cpp", aceLm, aceEncoder, xlTurbo, aceVae));
        l.add(ModelSet.of("yue2", Kind.SONG, "YuE2 3B", "yue2.cpp",
                Part.of("model", YUE2, "YuE2-3B-Q8_0.gguf"),
                Part.of("vae", YUE2, "YuE2-Vae-F32.gguf"))
                .withLicense(YUE2_LICENSE).withAliases("song:yue2-3b"));

        // video (stable-diffusion.cpp): VideoClip's files. Lightricks gates the repo: accept its terms and set HF_TOKEN.
        for (String variant : List.of("distilled", "dev")) {
            l.add(ModelSet.of("ltx-2.5-" + variant, Kind.VIDEO, "LTX-2.5 22B " + variant, "stable-diffusion.cpp",
                    Part.of("diffusion_model", LTX, "diffusion_models/ltx-2.5-22b-" + variant + "-transformer-comfy-int8-convrot.safetensors"),
                    Part.of("llm", LTX, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
                    Part.of("vae", LTX, "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
                    Part.of("audio_vae", LTX, "vae/ltx-2.5-audio-vae-bf16.safetensors"),
                    Part.of("upscaler", LTX, "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"))
                    .withGated(true).withAliases("video:ltx-2.5-" + variant));
        }
        return List.copyOf(l);
    }
}
