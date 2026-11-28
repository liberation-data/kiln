package org.liberationdata.fornax.hub;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;

import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ModelCatalogTest {
    @Test void namesAreUnique() {
        Set<String> seen = new HashSet<>();
        for (ModelSet m : ModelCatalog.all()) {
            assertTrue(seen.add(m.id().toLowerCase(Locale.ROOT)), m.id());
            for (String a : m.aliases()) assertTrue(seen.add(a.toLowerCase(Locale.ROOT)), a);
        }
    }

    @Test void fornaxStudiosIdsAndTheClisNamesBothFindASet() {
        for (String studio : List.of("image:flux2-klein", "image:qwen-image-2512", "song:acestep-turbo", "song:acestep-turbo-lm4b",
                "song:acestep-xl-turbo", "song:yue2-3b", "video:ltx-2.5-distilled", "video:ltx-2.5-dev", "vision:qwen3-vl-4b",
                "vision:paddleocr-vl", "vision:medgemma-4b", "vision:ocria-1"))
            assertTrue(ModelCatalog.byName(studio).isPresent(), studio);
        for (String cli : List.of("qwen3-0.6b", "qwen3-4b", "ace-step", "ace-step-xl", "yue2", "flux2-klein", "qwen-image", "qwen-image-edit"))
            assertTrue(ModelCatalog.byName(cli).isPresent(), cli);
        assertEquals("flux2-klein", ModelCatalog.require("IMAGE:FLUX2-KLEIN").id());
        var e = assertThrows(IllegalArgumentException.class, () -> ModelCatalog.require("nope"));
        assertTrue(e.getMessage().contains("qwen3-4b"));
    }

    @Test void theCliAceStepHasTheLyricLmAndStudiosTurboDoesNot() {
        assertEquals(List.of("lm", "text_encoder", "dit", "vae"), roles("ace-step"));
        assertEquals(List.of("text_encoder", "dit", "vae"), roles("song:acestep-turbo"));
        assertEquals("acestep-v15-xl-turbo-Q8_0.gguf", ModelCatalog.require("ace-step-xl").part("dit").orElseThrow().primary().file());
    }

    @Test void ltxIsFiveGatedPartsAndCanDropItsAudioVae() {
        ModelSet ltx = ModelCatalog.require("ltx-2.5-distilled");
        assertEquals(Kind.VIDEO, ltx.kind());
        assertTrue(ltx.gated());
        assertEquals(List.of("diffusion_model", "llm", "vae", "audio_vae", "upscaler"), roles("ltx-2.5-distilled"));
        ModelSet silent = ltx.without("audio_vae", "+silent", " (no audio)");
        assertEquals("ltx-2.5-distilled+silent", silent.id());
        assertEquals(4, silent.parts().size());
        assertTrue(silent.part("audio_vae").isEmpty());
        assertTrue(silent.gated());
        assertThrows(IllegalArgumentException.class, () -> silent.without("audio_vae", "", ""));
    }

    @Test void imageSetsHaveTheStepsAndGuidanceTheyAreTunedFor() {
        assertEquals(20, ModelCatalog.imageSteps(ModelCatalog.require("flux2-klein")));
        assertEquals(4.0, ModelCatalog.imageCfg(ModelCatalog.require("flux2-klein")));
        assertEquals(2.5, ModelCatalog.imageCfg(ModelCatalog.require("qwen-image-edit")));
        assertEquals(8, ModelCatalog.imageSteps(ModelCatalog.require("image:z-image-turbo")));
        assertEquals(1.0, ModelCatalog.imageCfg(ModelCatalog.require("z-image-turbo")));
        assertEquals(List.of("diffusion_model", "vae", "llm"), roles("z-image-turbo"));
        assertEquals(8, ModelCatalog.imageSteps(ModelCatalog.require("image:ernie-image-turbo")));
        assertEquals(1.0, ModelCatalog.imageCfg(ModelCatalog.require("ernie-image-turbo")));
        assertEquals(List.of("diffusion_model", "vae", "llm"), roles("ernie-image-turbo"));
        // the FLUX.2 VAE is klein's own file, so pulling one brings it for the other
        assertEquals(ModelCatalog.require("flux2-klein").part("vae").orElseThrow().primary(),
                ModelCatalog.require("ernie-image-turbo").part("vae").orElseThrow().primary());
    }

    @Test void licensesThatLimitUseAreSaid() {
        assertTrue(ModelCatalog.require("yue2").license().startsWith("CC BY-NC 4.0"));
        assertTrue(ModelCatalog.require("medgemma-4b").license().startsWith("Health AI Developer Foundations"));
        assertTrue(ModelCatalog.require("ocria-1").license().contains("not open source"));
        assertTrue(ModelCatalog.require("ocria-1").license().contains("LlamaParse"));
        assertNull(ModelCatalog.require("qwen3-4b").license());
        assertTrue(ModelCatalog.all().stream().filter(ModelSet::gated).allMatch(m -> m.kind() == Kind.VIDEO));
    }

    @Test void everyKindHasASet() {
        for (Kind k : Kind.values()) assertFalse(ModelCatalog.of(k).isEmpty(), k.toString());
    }

    @Test void qwenImageAcceptsItsTextEncoderFromEitherRepo() {
        Part llm = ModelCatalog.require("qwen-image").part("llm").orElseThrow();
        assertEquals(2, llm.sources().size());
        assertEquals("Comfy-Org/Qwen-Image_ComfyUI", llm.primary().repo());
        assertTrue(llm.uses("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"));
    }

    @Test void aSetRejectsNonsense() {
        assertThrows(IllegalArgumentException.class, () -> Part.of("x", "not-a-repo", "f"));
        assertThrows(IllegalArgumentException.class, () -> Part.of("x", "a/b", "../../etc/passwd"));
        assertThrows(IllegalArgumentException.class, () -> ModelSet.of("x", Kind.CHAT, "x", "e", Part.of("a", "a/b", "f"), Part.of("a", "a/b", "g")));
        assertEquals("main", Part.of("x", "a/b", "f").primary().revision());
    }

    private static List<String> roles(String name) { return ModelCatalog.require(name).parts().stream().map(Part::role).toList(); }
}
