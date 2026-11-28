package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs Qwen3-VL 4B on small generated images when it's in the Hugging Face cache and the native library is built;
 * skipped otherwise.
 */
class VisionIT {
    static final Weights W = Weights.QWEN3_VL_4B;
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withMaxTokens(32);
    static LlamaModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = new LlamaModel(W, -1, 4096);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    /** A red square on white. */
    static BufferedImage redSquare() {
        var img = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 256, 256);
        g.setColor(Color.RED);
        g.fillRect(64, 64, 128, 128);
        g.dispose();
        return img;
    }

    @Test
    void supportsVision() { assertTrue(model.supportsVision()); }

    /** Qwen3-VL's mmproj has a vision encoder and no audio one: audio is refused in Java and by the shim alike. */
    @Test
    void refusesAudio() {
        assertFalse(model.supportsAudio());
        assertEquals(-1, model.audioSampleRate());
        var beep = Audio.ofMono(new float[1600], 16000);
        var e = assertThrows(IllegalStateException.class, () -> model.chat(List.of(Message.user("Transcribe audio to text", beep)), GREEDY));
        assertTrue(e.getMessage().contains("takes no audio: its mmproj has no audio encoder"), e.getMessage());
        String req = """
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]}]}""";
        e = assertThrows(IllegalStateException.class, () -> model.chatJson(req, null));
        assertTrue(e.getMessage().contains("takes no audio (its mmproj has no audio encoder)"), e.getMessage());
    }

    @Test
    void chatNamesTheColourOfASquare() {
        var pieces = new StringBuilder();
        var r = model.chat(List.of(Message.user("What colour is the square? Answer in one word.", Image.of(redSquare()))),
                GREEDY, pieces::append);
        System.out.println(r);
        assertTrue(r.text().toLowerCase().contains("red"), r.text());
        assertEquals(r.text(), pieces.toString());
        assertTrue(r.promptTokens() > 20, "image tokens should count: " + r.promptTokens());
    }

    @Test
    void completeWithAMarker() {
        String prompt = "<|im_start|>user\n<|vision_start|>" + LlamaModel.mediaMarker() + "<|vision_end|>"
                + "What colour is the square? Answer in one word.<|im_end|>\n<|im_start|>assistant\n";
        var r = model.complete(prompt, List.of(Image.of(redSquare())), GREEDY, null);
        assertTrue(r.text().toLowerCase().contains("red"), r.text());
    }

    @Test
    void extractFollowsTheSchema() {
        String schema = """
            {"type": "object", "properties": {"shape": {"type": "string", "enum": ["circle", "square", "triangle"]},
             "colour": {"type": "string"}}, "required": ["shape", "colour"]}""";
        String json = model.extract(List.of(Message.user("Describe the shape.", Image.of(redSquare()))), schema,
                Options.DEFAULT.withTemperature(0).withMaxTokens(64));
        System.out.println(json);
        var m = (Map<?, ?>) Json.parse(json);
        assertEquals("square", m.get("shape"));
        assertTrue(m.get("colour").toString().toLowerCase().contains("red"), json);
    }

    @Test
    void extractThatRunsOutOfTokensIsAnError() {
        String schema = """
            {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}""";
        var e = assertThrows(IllegalStateException.class, () -> model.extract(
                List.of(Message.user("Describe the image in detail.", Image.of(redSquare()))), schema,
                Options.DEFAULT.withTemperature(0).withMaxTokens(3)));
        assertTrue(e.getMessage().contains("cut off (length)"), e.getMessage());
    }

    static String imageUrlRequest() throws Exception {
        var png = new ByteArrayOutputStream();
        ImageIO.write(redSquare(), "png", png);
        String url = "data:image/png;base64," + Base64.getEncoder().encodeToString(png.toByteArray());
        return """
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "%s"}},
               {"type": "text", "text": "What colour is the square? Answer in one word."}]}],
             "temperature": 0, "max_tokens": 16}""".formatted(url);
    }

    @Test
    void chatJsonTakesAnImageUrl() throws Exception {
        String res = model.chatJson(imageUrlRequest(), List.of(), null);
        assertTrue(res.toLowerCase().contains("red"), res);
    }

    @Test
    void aPromptWithoutItsMarkerIsAnError() {
        var e = assertThrows(IllegalStateException.class,
                () -> model.complete("What colour is the square?", List.of(Image.of(redSquare())), GREEDY, null));
        assertTrue(e.getMessage().contains("image markers"), e.getMessage());
    }

    /** A conversation with text-only turns between image turns: each image goes with its own message, in order. */
    @Test
    void chatKeepsImagesWithTheirMessages() {
        var blue = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = blue.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 256, 256);
        g.setColor(Color.BLUE);
        g.fillRect(64, 64, 128, 128);
        g.dispose();
        var r = model.chat(List.of(Message.system("Answer in one word."),
                Message.user("What colour is the square?", Image.of(redSquare())), Message.assistant("Red."),
                Message.user("And the square in this picture?", Image.of(blue))), GREEDY);
        assertTrue(r.text().toLowerCase().contains("blue"), r.text());
    }

    @Test
    void aMissingMmprojIsAnError() {
        var e = assertThrows(IllegalStateException.class, () -> new LlamaModel(W.model(), "no-such-mmproj.gguf", 0, 512));
        assertTrue(e.getMessage().contains("fornax_llama_create_mm: unable to load mmproj: no-such-mmproj.gguf"), e.getMessage());
    }

    /** The same GGUF without its mmproj, on the CPU (its weights are mmapped, so this costs little): no images. */
    @Test
    void withoutItsMmprojAModelRejectsImages() throws Exception {
        try (var text = new LlamaModel(W.model(), null, 0, 512)) {
            assertFalse(text.supportsVision());
            var e = assertThrows(IllegalStateException.class,
                    () -> text.chat(List.of(Message.user("Colour?", Image.of(redSquare()))), GREEDY));
            assertTrue(e.getMessage().contains("takes no images"), e.getMessage());
            String req = imageUrlRequest();
            e = assertThrows(IllegalStateException.class, () -> text.chatJson(req, List.of(), null));
            assertTrue(e.getMessage().contains("takes no images"), e.getMessage());
        }
    }

    /**
     * Qwen3-VL's M-RoPE gives a ~1200-token image only ~40 positions, so generation must stop when the context's
     * cells are full, not when the position reaches the context size (which would overrun the KV cache).
     */
    @Test
    void aFullContextAfterAnImageStopsGeneration() {
        var page = new BufferedImage(1000, 1300, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = page.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 1000, 1300);
        g.dispose();
        Message ask = Message.user("Ignore the image. Count from 1 to 2000, one number per line.", Image.of(page));
        try (var small = new LlamaModel(W, -1, 1536)) {
            var r = small.chat(List.of(ask), Options.DEFAULT.withTemperature(0).withMaxTokens(0));
            assertTrue(r.promptTokens() > 1000, "the image should take most of the context: " + r.promptTokens());
            // the last token is sampled but never decoded, so it takes no cell
            assertEquals(1537, r.promptTokens() + r.generatedTokens(), r.promptTokens() + " + " + r.generatedTokens());
            String json = small.chatJson("{\"messages\": " + ChatRequests.messages(List.of(ask), new ArrayList<>(), new ArrayList<>())
                    + ", \"temperature\": 0}", List.of(Image.of(page)), null);
            assertEquals("length", Json.get(Json.parse(json), "finish_reason"), json);
        }
    }

    @Test
    void textOnlyStillWorks() {
        var r = model.chat(List.of(Message.user("Reply with exactly the word: fornax")), GREEDY);
        assertTrue(r.text().toLowerCase().contains("fornax"), r.text());
    }

    @Test
    void badImageIndexIsAnError() {
        String req = """
            {"messages": [{"role": "user", "content": [{"type": "image", "index": 3}, {"type": "text", "text": "hi"}]}]}""";
        var e = assertThrows(IllegalStateException.class, () -> model.chatJson(req, List.of(Image.of(redSquare())), null));
        assertTrue(e.getMessage().contains("out of range"), e.getMessage());
    }
}
