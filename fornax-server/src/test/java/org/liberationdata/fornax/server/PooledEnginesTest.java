package org.liberationdata.fornax.server;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelFolders;
import org.liberationdata.fornax.hub.PullListener;
import org.liberationdata.fornax.server.Engines.Use;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Which models a request may name: by default those the server was started with and catalogue models already
 * downloaded; with allowDownloads, any. The hub is offline over an empty cache, so nothing is downloaded or loaded.
 */
class PooledEnginesTest {
    static final String REF = "someone/some-model-GGUF:Q4_K_M";
    @TempDir Path dir;

    FornaxHub hub() {
        return FornaxHub.builder().cacheRoot(dir.resolve("hf")).modelFolders(new ModelFolders(List.of(dir.resolve("models"))))
                .offline(true).build();
    }

    PooledEngines engines(boolean allowDownloads, String... models) {
        return new PooledEngines(hub(), new ModelPool(1L << 30, null), List.of(models), 0, 1, PullListener.NONE, s -> {}, allowDownloads);
    }

    static ApiError notServed(org.junit.jupiter.api.function.Executable call) {
        ApiError e = assertThrows(ApiError.class, call);
        assertEquals(404, e.status, e.getMessage());
        assertEquals("model_not_found", e.code);
        assertTrue(e.getMessage().contains("--allow-downloads"), e.getMessage());
        return e;
    }

    @Test void byDefaultARequestCantNameAModelToDownload() throws Exception {
        Path gguf = Files.write(dir.resolve("model.gguf"), new byte[]{0});
        try (var e = engines(false)) {
            notServed(() -> e.checkChat("qwen3-4b", false));
            notServed(() -> e.chat("qwen3-4b", "{}", false, null, new Engines.Stopper()));
            notServed(() -> e.checkChat(REF, false));
            notServed(() -> e.checkChat(gguf.toString(), false));
            notServed(() -> e.embed(e.defaultModel(Use.EMBED), List.of("a")));
            notServed(() -> e.embed(REF, List.of("a")));
            notServed(() -> e.images("flux2-klein", new Engines.ImageJob("p", 64, 64, 1, null, 0, 1)));
            notServed(() -> e.speak(e.defaultModel(Use.SPEECH), "Hi", null, 0));
            assertThrows(ApiError.class, () -> e.checkChat("flux2-klein", false), "the wrong kind is still a 400");
            assertEquals(400, assertThrows(ApiError.class, () -> e.checkChat("flux2-klein", false)).status);
        }
    }

    @Test void theModelsItWasStartedWithAndThoseDownloadedAreServed() throws Exception {
        try (var e = engines(false, "qwen3-4b", REF)) {
            e.checkChat("qwen3-4b", false);
            e.checkChat("Qwen3-4B", false);
            e.checkChat(REF, false);
        }
        var set = ModelCatalog.byName("qwen3-0.6b").orElseThrow();
        for (var p : set.parts()) {
            Path f = dir.resolve("models").resolve(p.primary().file());
            Files.createDirectories(f.getParent());
            Files.write(f, new byte[]{0});
        }
        try (var e = engines(false)) {
            e.checkChat("qwen3-0.6b", false);
            assertTrue(e.models().stream().anyMatch(m -> m.id().equals("qwen3-0.6b")));
            notServed(() -> e.checkChat("qwen3-4b", false));
        }
    }

    @Test void allowDownloadsLetsARequestNameAnyModel() {
        try (var e = engines(true)) {
            e.checkChat("qwen3-4b", false);
            e.checkChat(REF, false);
            // and it is downloaded: here the offline hub says it can't be
            ApiError a = assertThrows(ApiError.class, () -> e.chat("qwen3-4b", "{}", false, null, new Engines.Stopper()));
            assertEquals(500, a.status);
            assertTrue(a.getMessage().contains("HF_HUB_OFFLINE"), a.getMessage());
            ApiError b = assertThrows(ApiError.class, () -> e.embed(REF, List.of("a")));
            assertTrue(b.getMessage().contains("HF_HUB_OFFLINE"), b.getMessage());
        }
    }

    @Test void aModelThatWontDownloadAtStartClosesTheServer() throws Exception {
        var b = FornaxServer.builder().port(0).hub(hub()).model(REF).log(s -> {});
        var server = b.build();
        int port = server.port();
        var e = assertThrows(IllegalStateException.class, server::start);
        assertTrue(e.getMessage().contains("HF_HUB_OFFLINE"), e.getMessage());
        FornaxServer.builder().port(port).hub(hub()).log(s -> {}).build().close();   // the port is free again
    }

    @Test void aPromptLongerThanTheContextIsOpenAIsContextLengthError() {
        var e = (ApiError) PooledEngines.engineError(new IllegalStateException("llama chat failed (rc=2): prompt (9000 tokens) exceeds context 8192"));
        assertEquals(400, e.status);
        assertEquals("context_length_exceeded", e.code);
        assertEquals("messages", e.param);
        assertEquals("prompt (9000 tokens) exceeds context 8192", e.getMessage());
        var schema = (ApiError) PooledEngines.engineError(new IllegalStateException("llama chat failed (rc=6): bad schema"));
        assertEquals(400, schema.status);
        assertNull(schema.code);
        var other = new IllegalStateException("llama chat failed (rc=3): llama_process failed: 1");
        assertSame(other, PooledEngines.engineError(other));
    }

    /** A catalogue alias or the embedding model's name in another case is the same model, not a second one loaded. */
    @Test void aNameInAnotherCaseIsTheSameModel() {
        try (var e = engines(true, "QWEN3-EMBEDDING-0.6B", "Qwen3-4B")) {
            assertEquals("EMBED:qwen3-embedding-0.6b", e.find("QWEN3-EMBEDDING-0.6B", Use.EMBED, false).key());
            assertEquals("EMBED:qwen3-embedding-0.6b", e.find("qwen3-embedding-0.6b", Use.EMBED, false).key());
            assertEquals("CHAT:qwen3-4b", e.find("QWEN3-4B", Use.CHAT, false).key());
            assertEquals("CHAT:" + REF.toUpperCase(java.util.Locale.ROOT), e.find(REF.toUpperCase(java.util.Locale.ROOT), Use.CHAT, false).key(),
                    "a reference keeps its case");
            assertEquals(List.of("qwen3-embedding-0.6b", "qwen3-4b"), e.models().stream().map(Engines.Listed::id).limit(2).toList(),
                    "listed by their ids");
        }
        assertEquals("/Models/My-Embed.gguf", PooledEngines.canonical("/Models/My-Embed.gguf", Use.EMBED));
    }

    @Test void withParallelAChatModelHasSlotsButAVisionModelAndEmbeddingsHaveOne() {
        try (var e = new PooledEngines(hub(), new ModelPool(1L << 30, null), List.of("qwen3-4b", "qwen3-vl-4b", REF), 2048, 4,
                PullListener.NONE, s -> {}, false)) {
            assertEquals(4, e.slots(e.find("qwen3-4b", Use.CHAT, false)));
            assertEquals(4, e.slots(e.find(REF, Use.CHAT, false)));
            assertEquals(1, e.slots(e.find("qwen3-vl-4b", Use.CHAT, false)), "a model with slots takes no images");
            assertEquals(1, e.slots(e.find(REF, Use.EMBED, false)));
        }
        assertThrows(IllegalArgumentException.class, () -> new PooledEngines(hub(), new ModelPool(1L << 30, null), List.of(), 0, 4,
                PullListener.NONE, s -> {}, false), "slots need a context size each");
        assertThrows(IllegalArgumentException.class, () -> FornaxServer.builder().parallel(0));
        assertThrows(IllegalArgumentException.class, () -> FornaxServer.builder().parallel(257));
    }

    @Test void buildRefusesParallelWithoutAContextSizeBeforeStartingThePoolsReaper() {
        // a refused build is not closed by anyone: a model pool made first would leave its reaper thread running
        java.util.function.LongSupplier reapers = () -> Thread.getAllStackTraces().keySet().stream()
                .filter(t -> t.getName().equals("fornax-model-pool-reaper")).count();
        long before = reapers.getAsLong();
        var e = assertThrows(IllegalArgumentException.class, () -> FornaxServer.builder().port(0).hub(hub()).log(s -> {})
                .parallel(4).contextSize(0).build());
        assertTrue(e.getMessage().contains("contextSize"), e.getMessage());
        assertEquals(before, reapers.getAsLong(), "no model pool left behind");
    }

    PooledEngines jitllm(boolean allowDownloads, String... models) {
        return new PooledEngines(hub(), new ModelPool(1L << 30, null), List.of(models), 0, 1, PullListener.NONE, s -> {},
                allowDownloads, FornaxServer.LanguageEngine.JITLLM);
    }

    @Test void jitllmRefusesWhatItCantRunBeforeLoadingAnything() {
        try (var e = jitllm(true)) {
            e.checkChat("qwen3-4b", false);
            ApiError vision = assertThrows(ApiError.class, () -> e.checkChat("qwen3-vl-4b", false));
            assertEquals(400, vision.status);
            assertTrue(vision.getMessage().contains("vision model"), vision.getMessage());
            ApiError images = assertThrows(ApiError.class, () -> e.checkChat("qwen3-4b", true));
            assertEquals(400, images.status);
            assertTrue(images.getMessage().contains("jitllm"), images.getMessage());
            assertEquals(1, e.slots(e.find("qwen3-4b", Use.CHAT, false)));
        }
        try (var e = jitllm(false, "qwen3-vl-4b")) {   // offline: a download would fail differently
            ApiError startup = assertThrows(ApiError.class, e::warm);
            assertTrue(startup.getMessage().contains("vision model"), startup.getMessage());
        }
        assertThrows(IllegalArgumentException.class, () -> new PooledEngines(hub(), new ModelPool(1L << 30, null), List.of(),
                2048, 4, PullListener.NONE, s -> {}, false, FornaxServer.LanguageEngine.JITLLM));
    }
}
