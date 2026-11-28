package org.liberationdata.fornax.studio;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.llama.LlamaModel;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class CatalogTest {
    @TempDir Path root;

    private void file(String repo, String snapshot, String path, int bytes) throws Exception {
        Path p = root.resolve("models--" + repo.replace("/", "--")).resolve("snapshots").resolve(snapshot).resolve(path);
        Files.createDirectories(p.getParent());
        Files.write(p, new byte[bytes]);
    }

    @Test void findsChatModelsButNotTheSongEnginesOrEmbedders() throws Exception {
        file("Qwen/Qwen3-0.6B-GGUF", "a", "Qwen3-0.6B-Q8_0.gguf", 10);
        file("Qwen/Qwen3-0.6B-GGUF", "b", "Qwen3-0.6B-Q8_0.gguf", 10);  // the same file in a second snapshot
        file("big/Model-GGUF", "a", "q4/Model-Q4-00001-of-00002.gguf", 5);
        file("big/Model-GGUF", "a", "q4/Model-Q4-00002-of-00002.gguf", 7);
        file("Serveurperso/ACE-Step-1.5-GGUF", "a", "acestep-v15-turbo-Q8_0.gguf", 1);
        file("some/Embedder-GGUF", "a", "Qwen3-Embedding-0.6B-Q8_0.gguf", 1);
        file("some/safetensors", "a", "model.safetensors", 1);

        List<Catalog.Model> chat = new Catalog(root).chatModels();
        assertEquals(List.of("chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", "chat:big/Model-GGUF/q4/Model-Q4-00001-of-00002.gguf"),
                chat.stream().map(Catalog.Model::id).sorted().toList());
        var sharded = chat.stream().filter(m -> m.id().contains("big")).findFirst().orElseThrow();
        assertEquals(12, sharded.sizeBytes());
        assertEquals("Model-Q4", sharded.name());
        assertTrue(sharded.available());
    }

    @Test void speechModelsAreNotChatModelsThoughTheirArchitectureIs() throws Exception {
        // Soprano (text to speech) is a qwen3 GGUF; its projector makes audio. Qwen3-ASR's takes audio.
        file("Qwen/Qwen3-0.6B-GGUF", "a", "Qwen3-0.6B-Q8_0.gguf", 10);
        bytes("ggml-org/Soprano-1.1-80M-GGUF", "a", "Soprano-1.1-80M-F32.gguf", GgufHeaderTest.gguf("general.architecture", "qwen3"));
        bytes("ggml-org/Soprano-1.1-80M-GGUF", "a", "mmproj-Soprano-1.1-80M-F16.gguf",
                GgufHeaderTest.gguf("clip.has_gen_audio_encoder", true, "clip.gen.audio.projector_type", "soprano"));
        bytes("ggml-org/Qwen3-ASR-1.7B-GGUF", "a", "Qwen3-ASR-1.7B-Q8_0.gguf", GgufHeaderTest.gguf("general.architecture", "qwen3vl"));
        bytes("ggml-org/Qwen3-ASR-1.7B-GGUF", "a", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
                GgufHeaderTest.gguf("clip.has_audio_encoder", true, "clip.audio.projector_type", "qwen3a"));
        // An audio chat model hears (and here speaks) but chats too: it's in Chat as well.
        bytes("LiquidAI/LFM3-Audio-2B-GGUF", "a", "LFM3-Audio-2B-Q8_0.gguf", GgufHeaderTest.gguf("general.architecture", "lfm3"));
        bytes("LiquidAI/LFM3-Audio-2B-GGUF", "a", "mmproj-LFM3-Audio-2B-F16.gguf", GgufHeaderTest.gguf("clip.has_audio_encoder", true,
                "clip.audio.projector_type", "lfm3a", "clip.has_gen_audio_encoder", true, "clip.gen.audio.projector_type", "lfm3a_gen"));
        // A vision model kept outside the vision presets still chats.
        bytes("some/Vision-GGUF", "a", "Vision-Q4_K_M.gguf", GgufHeaderTest.gguf("general.architecture", "gemma3"));
        bytes("some/Vision-GGUF", "a", "mmproj-F16.gguf", GgufHeaderTest.gguf("clip.has_vision_encoder", true));

        assertEquals(List.of("chat:LiquidAI/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf", "chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
                        "chat:some/Vision-GGUF/Vision-Q4_K_M.gguf"),
                new Catalog(root).chatModels().stream().map(Catalog.Model::id).sorted().toList());
    }

    private void bytes(String repo, String snapshot, String path, byte[] content) throws Exception {
        Path p = root.resolve("models--" + repo.replace("/", "--")).resolve("snapshots").resolve(snapshot).resolve(path);
        Files.createDirectories(p.getParent());
        Files.write(p, content);
    }

    @Test void aPresetIsAvailableOnlyWithAllItsFiles() throws Exception {
        String ace = "Serveurperso/ACE-Step-1.5-GGUF";
        file(ace, "a", "Qwen3-Embedding-0.6B-Q8_0.gguf", 1);
        file(ace, "a", "acestep-v15-turbo-Q8_0.gguf", 2);
        var catalog = new Catalog(root);
        assertFalse(catalog.find("song:acestep-turbo").orElseThrow().available());
        file(ace, "a", "vae-BF16.gguf", 3);
        var turbo = catalog.find("song:acestep-turbo").orElseThrow();
        assertTrue(turbo.available());
        assertEquals(6, turbo.sizeBytes());
        assertFalse(catalog.find("song:acestep-turbo-lm4b").orElseThrow().available());
        assertFalse(catalog.find("image:flux2-klein").orElseThrow().available());
    }

    @Test void yue2IsASongModelWithItsLicenseAndNotAChatModel() throws Exception {
        file("Serveurperso/YuE2-GGUF", "a", "YuE2-3B-Q8_0.gguf", 4);
        var catalog = new Catalog(root);
        assertFalse(catalog.find("song:yue2-3b").orElseThrow().available());
        file("Serveurperso/YuE2-GGUF", "a", "YuE2-Vae-F32.gguf", 5);
        var yue2 = catalog.find("song:yue2-3b").orElseThrow();
        assertTrue(yue2.available());
        assertEquals(9, yue2.sizeBytes());
        assertEquals("yue2.cpp", yue2.engine());
        assertTrue(yue2.license().startsWith("CC BY-NC 4.0"));
        assertTrue(catalog.chatModels().isEmpty());
    }

    @Test void ernieImagesDiffusionGgufIsNotAChatModelButItsMinistralEncoderIs() throws Exception {
        file("unsloth/ERNIE-Image-Turbo-GGUF", "a", "ernie-image-turbo-Q8_0.gguf", 4);
        file("unsloth/Ministral-3-3B-Instruct-2512-GGUF", "a", "Ministral-3-3B-Instruct-2512-Q8_0.gguf", 2);
        file("Comfy-Org/flux2-dev", "a", "split_files/vae/flux2-vae.safetensors", 1);
        var catalog = new Catalog(root);
        // Ministral 3 3B Instruct chats in its own right, as well as encoding ERNIE's prompts.
        assertEquals(List.of("chat:unsloth/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q8_0.gguf"),
                catalog.chatModels().stream().map(Catalog.Model::id).toList());
        assertTrue(catalog.find("image:ernie-image-turbo").orElseThrow().available());
    }

    @Test void ocria1IsOfferedInInspectWithThinkingOffAndItsLicenseNotAsAChatModel() throws Exception {
        file(LlamaModel.Weights.OCRIA_1.repo, "a", LlamaModel.Weights.OCRIA_1.modelFile, 4);
        file(LlamaModel.Weights.OCRIA_1.repo, "a", LlamaModel.Weights.OCRIA_1.mmprojFile, 1);
        var catalog = new Catalog(root);
        assertTrue(catalog.chatModels().isEmpty());
        var ocria = catalog.visionPresets().stream().filter(m -> m.id().equals("vision:ocria-1")).findFirst().orElseThrow();
        assertEquals(false, ocria.defaults().get("thinking"));
        assertEquals("Transcribe this page in markdown.", ocria.defaults().get("prompt"));
        assertTrue(ocria.license().contains("not open source"));
        assertEquals("vision:qwen3-vl-4b", catalog.visionPresets().getFirst().id());   // still the default
    }

    @Test void ltxWithoutItsAudioVaeIsAnotherModel() throws Exception {
        var ltx = new Catalog(root).find("video:ltx-2.5-distilled").orElseThrow();
        assertEquals(Catalog.Kind.video, ltx.kind());
        assertEquals(List.of("diffusion_model", "llm", "vae", "audio_vae", "upscaler"), ltx.parts().stream().map(Catalog.Part::role).toList());
        var silent = ltx.without("audio_vae", "+silent", " (no audio)");
        assertEquals("video:ltx-2.5-distilled+silent", silent.id());
        assertNull(silent.path("audio_vae"));
        assertEquals(4, silent.parts().size());
    }

    @Test void findsPartsInAModelFolderAfterTheCache() throws Exception {
        // The video VAE in the cache, the rest in a ComfyUI-style models folder, as on a machine that ran ComfyUI.
        file("Lightricks/LTX-2.5", "s1", "vae/ltx-2.5-video-vae-conv-bf16.safetensors", 4);
        Path folder = root.resolve("comfy-models");
        for (String f : List.of("diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                "vae/ltx-2.5-audio-vae-bf16.safetensors",
                "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")) {
            Path p = folder.resolve(f);
            Files.createDirectories(p.getParent());
            Files.write(p, new byte[4]);
        }
        assertFalse(new Catalog(root).find("video:ltx-2.5-distilled").orElseThrow().available());
        var ltx = new Catalog(root, List.of(folder)).find("video:ltx-2.5-distilled").orElseThrow();
        assertTrue(ltx.available());
        assertEquals(folder.resolve("text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"), ltx.path("llm"));
        assertTrue(ltx.path("vae").startsWith(root.resolve("models--Lightricks--LTX-2.5")));
    }

    @Test void anEmptyCacheHasNoChatModels() {
        assertTrue(new Catalog(root.resolve("missing")).chatModels().isEmpty());
    }
}
