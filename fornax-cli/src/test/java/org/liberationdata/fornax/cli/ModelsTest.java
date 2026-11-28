package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.ModelFolders;
import org.liberationdata.fornax.hub.ModelSet.Kind;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ModelsTest {
    @TempDir Path root;

    FornaxHub hub() throws Exception {
        Path cache = root.resolve("hub");
        put(cache, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf");
        put(cache, "some/Multi-GGUF", "m-Q4_K_M.gguf");
        put(cache, "some/Multi-GGUF", "m-Q8_0.gguf");
        put(cache, "some/Multi-GGUF", "mmproj-m-F16.gguf");
        put(cache, "Serveurperso/ACE-Step-1.5-GGUF", "vae-BF16.gguf");   // one part of the ACE-Step sets
        put(cache, "ggerganov/whisper.cpp", "ggml-base.en.bin");
        return FornaxHub.builder().cacheRoot(cache).modelFolders(ModelFolders.none()).offline(true).build();
    }

    /** A file in the hub layout: a blob, and a snapshot link to it. */
    static void put(Path cache, String repo, String file) throws Exception {
        Path repoDir = cache.resolve("models--" + repo.replace("/", "--"));
        Path blob = repoDir.resolve("blobs/" + file + ".blob");
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, file);
        Path snap = repoDir.resolve("snapshots/abc/" + file);
        Files.createDirectories(snap.getParent());
        Files.createSymbolicLink(snap, snap.getParent().relativize(blob));
    }

    static final java.util.Set<Kind> CHAT = java.util.EnumSet.of(Kind.CHAT, Kind.VISION);

    @Test
    void chatFindsPathsAliasesAndReposInTheCacheWithoutDownloading() throws Exception {
        var hub = hub();
        var err = new ByteArrayOutputStream();
        var ps = new PrintStream(err, true, StandardCharsets.UTF_8);
        Path file = Files.writeString(root.resolve("local.gguf"), "x");
        assertEquals(file, Models.fetch(file.toString(), CHAT, "chat", true, hub, ps, false).file("gguf"));
        Path alias = Models.fetch("qwen3-0.6b", CHAT, "chat", true, hub, ps, false).file("gguf");
        assertEquals("Qwen3-0.6B-Q8_0.gguf", alias.getFileName().toString());
        assertEquals(alias, Models.fetch("Qwen/Qwen3-0.6B-GGUF", CHAT, "chat", true, hub, ps, false).file("gguf"));   // one model in the repo
        assertEquals("m-Q8_0.gguf", Models.fetch("some/Multi-GGUF:q8_0", CHAT, "chat", true, hub, ps, false).file("gguf").getFileName().toString());
        assertEquals("m-Q4_K_M.gguf", Models.fetch("some/Multi-GGUF:m-Q4_K_M.gguf", CHAT, "chat", true, hub, ps, false).file("gguf").getFileName().toString());
        assertEquals("", err.toString(StandardCharsets.UTF_8), "all in the cache: nothing said");
    }

    @Test
    void chatDownloadsWhatIsMissingAndRefusesOtherKindsFirst() throws Exception {
        var hub = hub();
        var err = new ByteArrayOutputStream();
        var ps = new PrintStream(err, true, StandardCharsets.UTF_8);
        // missing: pulled (this hub is offline, so the pull stops at the first file, after saying what it fetches)
        var offline = assertThrows(HubException.class, () -> Models.fetch("qwen3-4b", CHAT, "chat", true, hub, ps, false));
        assertEquals(HubException.Reason.OFFLINE, offline.reason());
        assertTrue(err.toString(StandardCharsets.UTF_8).contains("[fornax] downloading qwen3-4b"), err.toString());
        // a split model missing a part is finished, not used (offline: the missing part stops it)
        put(hub.cacheRoot(), "some/Split-GGUF", "s-Q4_K_M-00001-of-00002.gguf");
        assertThrows(HubException.class, () -> Models.fetch("some/Split-GGUF:Q4_K_M", CHAT, "chat", true, hub, ps, false));
        err.reset();
        var image = assertThrows(Main.Failure.class, () -> Models.fetch("flux2-klein", CHAT, "chat", true, hub, ps, false));
        assertTrue(image.getMessage().startsWith("flux2-klein is an image model; chat takes a chat model: qwen3-0.6b"), image.getMessage());
        assertTrue(image.getMessage().contains(", or a vision model: qwen3-vl-4b"), image.getMessage());
        assertEquals("", err.toString(StandardCharsets.UTF_8), "refused before any download");
        var unknown = assertThrows(Main.Failure.class, () -> Models.fetch("what", CHAT, "chat", true, hub, ps, false));
        assertTrue(unknown.getMessage().startsWith("No such file or model: what"), unknown.getMessage());
    }

    @Test
    void offlineARepoWithSeveralModelsInTheCacheSaysToNameOne() throws Exception {
        var hub = hub();
        var ps = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
        // the cache alone can't pick one, and the Hub can't be asked: name one, rather than "needs the Hub"
        var several = assertThrows(IllegalArgumentException.class, () -> Models.fetch("some/Multi-GGUF", CHAT, "chat", true, hub, ps, false));
        assertTrue(several.getMessage().contains("name one") && several.getMessage().contains("some/Multi-GGUF:m-Q8_0.gguf"),
                several.getMessage());
        assertFalse(several.getMessage().contains("mmproj"), several.getMessage());
        // nothing in the cache matches: the Hub would be asked, so offline says so
        var none = assertThrows(HubException.class, () -> Models.fetch("some/Multi-GGUF:IQ2", CHAT, "chat", true, hub, ps, false));
        assertEquals(HubException.Reason.OFFLINE, none.reason());
        // a named revision is fetched as pull fetches it, not taken from main's copy in the cache
        var revision = assertThrows(HubException.class, () -> Models.fetch("some/Multi-GGUF@dev:Q8_0", CHAT, "chat", true, hub, ps, false));
        assertEquals(HubException.Reason.OFFLINE, revision.reason());
    }

    /** Not offline, but nothing answers at the Hub's address: the same as offline, by the cache. */
    @Test
    void anUnreachableHubWithSeveralModelsInTheCacheSaysToNameOne() throws Exception {
        int port;
        try (var s = new java.net.ServerSocket(0, 1, java.net.InetAddress.getLoopbackAddress())) { port = s.getLocalPort(); }   // closed: refused
        Path cache = hub().cacheRoot();
        var hub = FornaxHub.builder().cacheRoot(cache).modelFolders(ModelFolders.none()).token(null).offline(false)
                .endpoint(java.net.URI.create("http://127.0.0.1:" + port)).reserveBytes(0).build();
        var ps = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
        var several = assertThrows(IllegalArgumentException.class, () -> Models.fetch("some/Multi-GGUF", CHAT, "chat", true, hub, ps, false));
        assertTrue(several.getMessage().contains("name one"), several.getMessage());
        // nothing in the cache to name: the connection's own failure, not one made up for it
        var none = assertThrows(java.io.IOException.class, () -> Models.fetch("some/Multi-GGUF:IQ2", CHAT, "chat", true, hub, ps, false));
        assertFalse(none instanceof HubException, none.toString());
    }

    @Test
    void testRunsChatImageAndSongModelsButNotOthers() throws Exception {
        var hub = hub();
        var ps = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
        assertEquals("Qwen3-0.6B-Q8_0.gguf", Models.fetch("qwen3-0.6b", SmokeTest.KINDS, "test", true, hub, ps, false).file("gguf").getFileName().toString());
        var speech = assertThrows(Main.Failure.class, () -> Models.fetch("kokoro", SmokeTest.KINDS, "test", true, hub, ps, false));
        assertTrue(speech.getMessage().startsWith("kokoro is a speech model; test takes a chat model"), speech.getMessage());
        assertTrue(speech.getMessage().contains("an image model: flux2-klein") && speech.getMessage().contains("a song model: ace-step-turbo"),
                speech.getMessage());
    }

    /** A real download for chat, from a Hub on localhost: the file lands in the cache and its path comes back. */
    @Test
    void chatDownloadsFromTheHub() throws Exception {
        String commit = "0123456789abcdef0123456789abcdef01234567";
        byte[] content = "not really a gguf".getBytes(StandardCharsets.UTF_8);
        String etag = java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(content));
        var server = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress(java.net.InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/", ex -> {
            String path = ex.getRequestURI().getPath();
            byte[] body = new byte[0];
            int code = 404;
            if (path.equals("/api/models/fresh/model-GGUF/revision/main")) {
                code = 200;
                body = ("{\"sha\": \"" + commit + "\", \"siblings\": [{\"rfilename\": \"model-Q8_0.gguf\", \"size\": " + content.length + "}]}")
                        .getBytes(StandardCharsets.UTF_8);
            } else if (path.matches("/fresh/model-GGUF/resolve/(main|" + commit + ")/model-Q8_0.gguf")) {
                code = 200;
                ex.getResponseHeaders().set("X-Repo-Commit", commit);
                ex.getResponseHeaders().set("ETag", "\"" + etag + "\"");
                if (ex.getRequestMethod().equals("HEAD")) ex.getResponseHeaders().set("Content-Length", String.valueOf(content.length));
                else body = content;
            }
            boolean head = ex.getRequestMethod().equals("HEAD");
            ex.sendResponseHeaders(code, head || body.length == 0 ? -1 : body.length);
            if (!head) ex.getResponseBody().write(body);
            ex.close();
        });
        server.start();
        try {
            var hub = FornaxHub.builder().cacheRoot(root.resolve("empty")).modelFolders(ModelFolders.none()).token(null).offline(false)
                    .endpoint(java.net.URI.create("http://127.0.0.1:" + server.getAddress().getPort())).reserveBytes(0).build();
            var err = new ByteArrayOutputStream();
            Path gguf = Models.fetch("fresh/model-GGUF:Q8_0", CHAT, "chat", true, hub, new PrintStream(err, true, StandardCharsets.UTF_8), false)
                    .file("gguf");
            assertArrayEquals(content, Files.readAllBytes(gguf));
            assertTrue(gguf.startsWith(root.resolve("empty")), gguf.toString());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void fetchTakesOneKindAndPullsWhatIsMissing() throws Exception {
        var hub = hub();
        var err = new ByteArrayOutputStream();
        var ps = new PrintStream(err, true, StandardCharsets.UTF_8);
        var whisper = Models.fetch("whisper-base.en", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false);
        assertEquals("ggml-base.en.bin", whisper.file("model").getFileName().toString());
        assertEquals("whisper-base.en", whisper.set().id());
        assertEquals("", err.toString(StandardCharsets.UTF_8), "nothing to download, so nothing said");

        Path file = Files.writeString(root.resolve("my.bin"), "x");   // a path, for a command that loads one file
        assertEquals(file, Models.fetch(file.toString(), Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false).file("model"));
        assertEquals(Path.of("Qwen3-0.6B-Q8_0.gguf"), Models.fetch("Qwen/Qwen3-0.6B-GGUF:Q8_0", Kind.VISION, "see", true, hub, ps, false)
                .file("gguf").getFileName());

        // missing: pulled, which offline stops at the first file, after saying what it is fetching
        var offline = assertThrows(HubException.class, () -> Models.fetch("whisper-large-v3-turbo", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertEquals(HubException.Reason.OFFLINE, offline.reason());
        assertTrue(err.toString(StandardCharsets.UTF_8).contains("[fornax] downloading whisper-large-v3-turbo: Whisper large-v3-turbo"), err.toString());
    }

    @Test
    void fetchRefusesTheWrongKindAndUnknownNamesBeforeDownloading() throws Exception {
        var hub = hub();
        var ps = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
        var chat = assertThrows(Main.Failure.class, () -> Models.fetch("qwen3-4b", Kind.IMAGE, "image", false, hub, ps, false));
        assertEquals("qwen3-4b is a chat model; image takes an image model: flux2-klein, qwen-image, qwen-image-edit, z-image-turbo, ernie-image-turbo", chat.getMessage());
        var song = assertThrows(Main.Failure.class, () -> Models.fetch("flux2-klein", Kind.SONG, "song", false, hub, ps, false));
        assertTrue(song.getMessage().startsWith("flux2-klein is an image model; song takes a song model: ace-step-turbo"), song.getMessage());
        var stt = assertThrows(Main.Failure.class, () -> Models.fetch("kokoro", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertTrue(stt.getMessage().contains("takes a speech-to-text model: whisper-large-v3-turbo, whisper-base.en"), stt.getMessage());
        var ref = assertThrows(Main.Failure.class, () -> Models.fetch("some/Multi-GGUF:Q8_0", Kind.IMAGE, "image", false, hub, ps, false));
        assertTrue(ref.getMessage().contains("not a reference"), ref.getMessage());
        var unknown = assertThrows(Main.Failure.class, () -> Models.fetch("what", Kind.SONG, "song", false, hub, ps, false));
        assertTrue(unknown.getMessage().startsWith("No such model: what") && unknown.getMessage().contains("ace-step-turbo"), unknown.getMessage());
        var unknownFile = assertThrows(Main.Failure.class, () -> Models.fetch("what.bin", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertTrue(unknownFile.getMessage().startsWith("No such file or model: what.bin"), unknownFile.getMessage());
        var notWhisper = assertThrows(Main.Failure.class, () -> Models.fetch("Qwen/Qwen3-0.6B-GGUF:Q8_0", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertTrue(notWhisper.getMessage().contains("not a whisper.cpp model"), notWhisper.getMessage());
    }

    @Test
    void aNameThatCannotBeAPathIsStillAName() {
        // On Windows a reference's ':' is no path at all (InvalidPathException); a NUL is the same on every OS.
        String notAPath = "acme/x\0y";
        assertFalse(Models.withoutMmproj(notAPath));
        var ps = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
        var e = assertThrows(Main.Failure.class, () -> Models.fetch(notAPath, Kind.SPEECH, "speak", true, hub(), ps, false));
        assertTrue(e.getMessage().startsWith("No such file or model"), e.getMessage());
    }

    @Test
    void aSplitModelMissingAPartIsntListedAsRunnable() throws Exception {
        var hub = hub();
        put(hub.cacheRoot(), "some/Split-GGUF", "s-Q4_K_M-00001-of-00002.gguf");   // part 2 never arrived
        assertEquals(java.util.List.of(), Listing.otherLlms(hub.cacheRoot()).stream()
                .filter(x -> x.repo().equals("some/Split-GGUF")).toList());
    }

    @Test
    void listShowsTheCatalogueAndOtherModels() throws Exception {
        var out = new ByteArrayOutputStream();
        assertEquals(0, Listing.run(hub(), new PrintStream(out, true, StandardCharsets.UTF_8)));
        String s = out.toString(StandardCharsets.UTF_8);
        assertTrue(s.contains("Chat (fornax chat <alias>):\n  qwen3-0.6b"), s);
        assertTrue(s.contains("ace-step-turbo") && s.contains("partly downloaded"), s);   // the VAE alone
        assertTrue(s.contains("some/Multi-GGUF:m-Q8_0.gguf"), s);
        assertFalse(s.contains("Qwen/Qwen3-0.6B-GGUF:"), s);   // a catalogue model shows by its alias
        assertFalse(s.contains("mmproj"), s);
        assertTrue(s.contains("Not downloaded (fornax pull <alias>): ") && s.contains("flux2-klein"), s);
    }
}
