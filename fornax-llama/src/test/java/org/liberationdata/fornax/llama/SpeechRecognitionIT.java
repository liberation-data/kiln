package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Qwen3-ASR 0.6B transcribes the first clip of the explainer's narration (videos/explainer/narration/1.mp3, 14 s of
 * synthesized speech) through chat with an {@link Audio}, through chatJson with an MP3 input_audio part, and in a
 * worker process, each within a word error rate of its script. Skipped unless the model is in the Hugging Face cache
 * and the native library is built.
 */
class SpeechRecognitionIT {
    static final Weights W = Weights.QWEN3_ASR_0_6B;
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withMaxTokens(256);
    static final String ASK = "Transcribe audio to text";
    /**
     * The worst word error rate accepted. Qwen3-ASR 0.6B gets 3 of the clip's 21 words wrong (0.14), as llama-mtmd-cli
     * does on the same file: "Fornax" and "your JVM" ("Fortis", "U GPM") and a dropped "a".
     */
    static final double MAX_WER = 0.25;
    static LlamaModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = new LlamaModel(W, -1, 4096);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    /** The narration's folder: the tests run in the module's directory, the eval script in the repository's. */
    static Path narration() {
        Path p = Stream.of("videos/explainer/narration", "../videos/explainer/narration").map(Path::of)
                .filter(Files::isDirectory).findFirst().orElse(null);
        assumeTrue(p != null, "the explainer's narration isn't in this checkout");
        return p;
    }

    /** What clip i says: its script, without the voice's name on the first line or the [pause] marks. */
    static String script(int i) throws Exception {
        List<String> lines = Files.readAllLines(narration().resolve(i + ".txt"));
        return String.join(" ", lines.subList(1, lines.size())).replaceAll("\\[pause[^]]*]", " ");
    }

    /** Lower case, punctuation as spaces (so "X-ray" is two words either way), apostrophes dropped. */
    static List<String> words(String s) {
        String n = s.toLowerCase().replaceAll("['’]", "").replaceAll("[^\\p{L}\\p{N}]+", " ").strip();
        return n.isEmpty() ? List.of() : List.of(n.split(" "));
    }

    /** Word error rate: the word edit distance from the reference, over the reference's length. */
    static double wer(String reference, String hypothesis) {
        List<String> r = words(reference), h = words(hypothesis);
        int[] prev = new int[h.size() + 1], cur = new int[h.size() + 1];
        for (int j = 0; j <= h.size(); j++) prev[j] = j;
        for (int i = 1; i <= r.size(); i++) {
            cur[0] = i;
            for (int j = 1; j <= h.size(); j++)
                cur[j] = Math.min(Math.min(cur[j - 1], prev[j]) + 1, prev[j - 1] + (r.get(i - 1).equals(h.get(j - 1)) ? 0 : 1));
            int[] t = prev; prev = cur; cur = t;
        }
        return prev[h.size()] / (double) r.size();
    }

    static void assertTranscribes(String text) throws Exception {
        double wer = wer(script(1), text);
        System.out.printf("WER %.3f: %s%n", wer, text);
        assertTrue(wer <= MAX_WER, "WER " + wer + ": " + text);
    }

    /** Clip 1 as a WAV file (ffmpeg decodes the MP3), read into an Audio: 24 kHz, so the model resamples it. */
    static Audio clip(Path dir) throws Exception {
        return Audio.readWav(wav(dir).toFile());
    }

    /** Clip 1 as a WAV file, with ffmpeg's options (-ar 16000 resamples it, say). */
    static Path wav(Path dir, String... options) throws Exception {
        Path mp3 = narration().resolve("1.mp3"), wav = dir.resolve("1-" + options.length + ".wav");
        List<String> cmd = new java.util.ArrayList<>(List.of("ffmpeg", "-v", "error", "-i", mp3.toString()));
        cmd.addAll(List.of(options));
        cmd.addAll(List.of("-y", wav.toString()));
        Process p;
        try {
            p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        } catch (java.io.IOException e) {
            assumeTrue(false, "no ffmpeg to decode the MP3");
            throw e;
        }
        String out = new String(p.getInputStream().readAllBytes());
        assertEquals(0, p.waitFor(), out);
        return wav;
    }

    @Test
    void takesAudioAndNoImages() {
        assertTrue(model.supportsAudio());
        assertFalse(model.supportsVision());
        assertEquals(16000, model.audioSampleRate());
        var e = assertThrows(IllegalStateException.class,
                () -> model.chat(List.of(Message.user("Colour?", new Image(1, 1, new byte[3]))), GREEDY));
        assertTrue(e.getMessage().contains("takes no images: its mmproj has no vision encoder"), e.getMessage());
        e = assertThrows(IllegalStateException.class,
                () -> model.complete(LlamaModel.mediaMarker(), List.of(new Image(1, 1, new byte[3])), GREEDY, null));
        assertTrue(e.getMessage().contains("takes no images: its mmproj has no vision encoder"), e.getMessage());
    }

    @Test
    void chatTranscribesAnAudio(@TempDir Path dir) throws Exception {
        Audio audio = clip(dir);
        assertEquals(24000, audio.sampleRate());
        var pieces = new StringBuilder();
        var r = model.chat(List.of(Message.user(ASK, audio)), GREEDY, pieces::append);
        assertTranscribes(r.text());
        assertEquals(r.text(), pieces.toString());
        // 14 s of audio at Qwen3-ASR's 13 tokens a second, and the prompt's own few
        assertTrue(r.promptTokens() > 150, "audio tokens should count: " + r.promptTokens());
    }

    @Test
    void chatJsonTranscribesAnMp3Part() throws Exception {
        String mp3 = Base64.getEncoder().encodeToString(Files.readAllBytes(narration().resolve("1.mp3")));
        String req = """
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "%s", "format": "mp3"}},
                                                     {"type": "text", "text": "%s"}]}],
             "temperature": 0, "max_tokens": 256}""".formatted(mp3, ASK);
        Object response = Json.parse(model.chatJson(req, null));
        assertTranscribes((String) Json.get(response, "message", "content"));
    }

    /**
     * The same 16 kHz WAV as an Audio (its samples, which need no resampling) and as an input_audio part (decoded by
     * llama.cpp, as llama-mtmd-cli reads a file): the samples reach the encoder unchanged, so the transcripts match.
     */
    @Test
    void anAudioTranscribesAsLlamaCppsOwnDecodingOfTheFile(@TempDir Path dir) throws Exception {
        Path wav = wav(dir, "-ar", "16000", "-ac", "1");
        String viaAudio = model.chat(List.of(Message.user(ASK, Audio.readWav(wav.toFile()))), GREEDY).text();
        String req = """
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "%s", "format": "wav"}},
                                                     {"type": "text", "text": "%s"}]}],
             "temperature": 0, "max_tokens": 256}""".formatted(Base64.getEncoder().encodeToString(Files.readAllBytes(wav)), ASK);
        Object viaFile = Json.parse(model.chatJson(req, null));
        assertEquals(Json.get(viaFile, "message", "content"), viaAudio);
        // the template's language tag comes back as reasoning, so the content (and chat's text) is the transcript alone
        assertEquals("language English", Json.get(viaFile, "message", "reasoning_content"));
        assertTranscribes(viaAudio);
    }

    @Test
    void anIsolatedModelTranscribesAnAudio(@TempDir Path dir) throws Exception {
        Audio audio = clip(dir);
        try (var isolated = LlamaModel.isolated(W)) {
            assertTrue(isolated.isIsolated());
            assertTrue(isolated.supportsAudio());
            assertEquals(16000, isolated.audioSampleRate());
            assertTranscribes(isolated.chat(List.of(Message.user(ASK, audio)), GREEDY).text());
            String req = "{\"messages\": " + ChatRequests.messages(List.of(Message.user(ASK, audio)), new java.util.ArrayList<>(),
                    new java.util.ArrayList<>()) + ", \"temperature\": 0}";
            assertTranscribes((String) Json.get(Json.parse(isolated.chatJson(req, List.of(), List.of(audio), null)), "message", "content"));
        }
    }

    /** The same GGUF without its mmproj, on the CPU (its weights are mmapped, so this costs little): no audio. */
    @Test
    void withoutItsMmprojAModelRefusesAudio() {
        Audio beep = Audio.ofMono(new float[1600], 16000);
        try (var text = new LlamaModel(W.model(), null, 0, 512)) {
            assertFalse(text.supportsAudio());
            assertEquals(-1, text.audioSampleRate());
            var e = assertThrows(IllegalStateException.class, () -> text.chat(List.of(Message.user(ASK, beep)), GREEDY));
            assertTrue(e.getMessage().contains("takes no audio: load it with an mmproj that has an audio encoder"), e.getMessage());
            String req = """
                {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]}]}""";
            e = assertThrows(IllegalStateException.class, () -> text.chatJson(req, null));
            assertTrue(e.getMessage().contains("takes no audio (load it with its mmproj)"), e.getMessage());
        }
        try (var slots = new LlamaModel(W.model(), 0, 512, 2)) {
            var e = assertThrows(IllegalStateException.class, () -> slots.chat(List.of(Message.user(ASK, beep)), GREEDY));
            assertTrue(e.getMessage().contains("audio isn't batched yet"), e.getMessage());
        }
    }

    /** Each malformed audio part is refused with what's wrong with it, and the model goes on working. */
    @Test
    void malformedAudioPartsAreRefused() {
        String part = """
            {"messages": [{"role": "user", "content": [%s, {"type": "text", "text": "Transcribe audio to text"}]}]}""";
        var e = assertThrows(IllegalStateException.class, () -> model.chatJson(part.formatted(
                "{\"type\": \"input_audio\", \"input_audio\": {\"data\": \"not*base64\", \"format\": \"wav\"}}"), null));
        assertTrue(e.getMessage().contains("invalid base64 in a media part"), e.getMessage());
        String noise = Base64.getEncoder().encodeToString("an ogg file, say: no format mtmd reads".getBytes());
        e = assertThrows(IllegalStateException.class, () -> model.chatJson(part.formatted(
                "{\"type\": \"input_audio\", \"input_audio\": {\"data\": \"" + noise + "\", \"format\": \"ogg\"}}"), null));
        assertTrue(e.getMessage().contains("could not decode an input_audio audio file"), e.getMessage());
        Audio beep = Audio.ofMono(new float[1600], 16000);
        e = assertThrows(IllegalStateException.class, () -> model.chatJson(part.formatted(
                "{\"type\": \"audio\", \"index\": 1}"), List.of(), List.of(beep), null));
        assertTrue(e.getMessage().contains("audio index 1 out of range (1 given)"), e.getMessage());
        e = assertThrows(IllegalStateException.class,
                () -> model.chat(List.of(Message.user(ASK, Audio.ofMono(new float[0], 16000))), GREEDY));
        assertTrue(e.getMessage().contains("audio 0 is empty"), e.getMessage());
        assertNotNull(model.chat(List.of(Message.user(ASK, beep)), GREEDY.withMaxTokens(8)).text());
    }

    @Test
    void anInputAudioPartThatIsAnImageIsRefused() throws Exception {
        var png = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(new java.awt.image.BufferedImage(4, 4, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", png);
        String req = """
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "%s", "format": "wav"}}]}]}"""
                .formatted(Base64.getEncoder().encodeToString(png.toByteArray()));
        var e = assertThrows(IllegalStateException.class, () -> model.chatJson(req, null));
        assertTrue(e.getMessage().contains("an input_audio part holds an image"), e.getMessage());
    }
}
