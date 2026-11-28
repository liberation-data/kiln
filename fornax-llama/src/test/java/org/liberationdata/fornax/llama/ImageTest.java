package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Image conversion and request building: no native library needed. */
class ImageTest {
    static BufferedImage twoPixels() {
        var img = new BufferedImage(2, 1, BufferedImage.TYPE_INT_ARGB);
        img.setRGB(0, 0, 0x80FF0000);  // half-transparent red: alpha is dropped
        img.setRGB(1, 0, 0xFF0080FF);
        return img;
    }

    @Test
    void packsRgbRowByRow() {
        Image i = Image.of(twoPixels());
        assertEquals(2, i.width());
        assertEquals(1, i.height());
        assertArrayEquals(new byte[] {(byte) 0xFF, 0, 0, 0, (byte) 0x80, (byte) 0xFF}, i.rgb());
    }

    @Test
    void decodesPngBytesAndFiles(@org.junit.jupiter.api.io.TempDir Path dir) throws Exception {
        var png = new ByteArrayOutputStream();
        ImageIO.write(twoPixels(), "png", png);
        assertArrayEquals(Image.of(twoPixels()).rgb(), Image.decode(png.toByteArray()).rgb());
        Path f = dir.resolve("two.png");
        Files.write(f, png.toByteArray());
        assertArrayEquals(Image.of(twoPixels()).rgb(), Image.read(f).rgb());
    }

    @Test
    void rejectsBadImages() {
        assertThrows(IllegalArgumentException.class, () -> new Image(2, 2, new byte[5]));
        assertThrows(IllegalArgumentException.class, () -> new Image(0, 2, new byte[0]));
        assertThrows(IllegalArgumentException.class, () -> Image.decode("not an image".getBytes()));
        assertThrows(java.io.UncheckedIOException.class, () -> Image.read(Path.of("no-such-image.png")));
    }

    @Test
    void messagesWithoutImagesOrTextAreNormalised() {
        assertEquals(List.of(), new Message("user", "hi", null).images());
        assertEquals(List.of(), Message.user("hi").images());
        Image a = Image.of(twoPixels());
        String json = ChatRequests.messages(List.of(new Message("user", null, List.of(a))), new ArrayList<>(), new ArrayList<>());
        assertEquals(Json.parse("""
            [{"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": ""}]}]"""), Json.parse(json));
    }

    @Test
    void messagesPutImagesBeforeTextAndNumberThemInOrder() {
        Image a = Image.of(twoPixels()), b = Image.of(twoPixels());
        List<Image> images = new ArrayList<>();
        String json = ChatRequests.messages(List.of(Message.system("Be brief."), Message.user("Compare \"these\".", a, b),
                Message.assistant("ok"), Message.user("And this?", a)), images, new ArrayList<>());
        assertEquals(List.of(a, b, a), images);
        assertEquals(Json.parse("""
            [{"role": "system", "content": "Be brief."},
             {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "image", "index": 1},
                                          {"type": "text", "text": "Compare \\"these\\"."}]},
             {"role": "assistant", "content": "ok"},
             {"role": "user", "content": [{"type": "image", "index": 2}, {"type": "text", "text": "And this?"}]}]"""),
                Json.parse(json));
    }

    @Test
    void extractRequestCarriesTheSchemaAndGreedySampling() {
        String schema = "{\"type\": \"object\", \"properties\": {\"total\": {\"type\": \"number\"}}}";
        Object req = Json.parse(ChatRequests.extract(List.of(Message.user("Total?")), schema,
                Options.DEFAULT.withTemperature(-1).withMaxTokens(100).withSeed(7), new ArrayList<>(), new ArrayList<>()));
        assertEquals(Json.parse(schema), Json.get(req, "json_schema"));
        assertEquals(0.0, Json.get(req, "temperature"));
        assertEquals(100L, Json.get(req, "max_tokens"));
        assertEquals(7L, Json.get(req, "seed"));
        assertEquals("Total?", Json.get(req, "messages", 0, "content"));
    }

    @Test
    void thinkingIsOnUnlessTurnedOffAndOnlyAnOffSwitchIsSent() {
        assertTrue(Options.DEFAULT.thinking());
        assertTrue(new Options(10, 0, 0.9f, 0.05f, 1).thinking());   // the five-value constructor, as before the switch
        Options off = Options.DEFAULT.withThinking(false).withSeed(3).withMaxTokens(9).withTemperature(0);
        assertFalse(off.thinking());   // the other withers keep it
        assertNull(Json.get(Json.parse(ChatRequests.chat(List.of(Message.user("Hi")), Options.DEFAULT, new ArrayList<>(), new ArrayList<>())),
                "enable_thinking"));
        Object req = Json.parse(ChatRequests.chat(List.of(Message.system("Be brief."), Message.user("Hi")), off, new ArrayList<>(), new ArrayList<>()));
        assertEquals(false, Json.get(req, "enable_thinking"));
        assertEquals(9L, Json.get(req, "max_tokens"));
        assertEquals(3L, Json.get(req, "seed"));
        assertEquals(0L, Json.get(req, "top_k"));
        assertEquals("Be brief.", Json.get(req, "messages", 0, "content"));
        Object extract = Json.parse(ChatRequests.extract(List.of(Message.user("Total?")), "{\"type\": \"object\"}", off,
                new ArrayList<>(), new ArrayList<>()));
        assertEquals(false, Json.get(extract, "enable_thinking"));
    }

    @Test
    void aChatJsonResponseBecomesAResult() {
        var r = ChatRequests.result("""
                {"message": {"role": "assistant", "content": "Hello.", "reasoning_content": "hmm"},
                 "finish_reason": "stop", "usage": {"prompt_tokens": 12, "completion_tokens": 3}}""");
        assertEquals(new LlamaModel.Result("Hello.", 12, 3), r);
        assertEquals("", ChatRequests.result("{\"message\": {\"role\": \"assistant\", \"content\": null, \"tool_calls\": []}}").text());
    }

    @Test
    void weightsByName() {
        assertEquals(LlamaModel.Weights.QWEN3_VL_4B, LlamaModel.Weights.named("qwen3-vl-4b"));
        assertEquals(LlamaModel.Weights.PADDLEOCR_VL, LlamaModel.Weights.named("paddleocr-vl"));
        assertEquals(LlamaModel.Weights.QWEN3_EMBEDDING_0_6B, LlamaModel.Weights.named("qwen3-embedding-0.6b"));
        assertEquals(LlamaModel.Weights.QWEN3_RERANKER_0_6B, LlamaModel.Weights.named("Qwen3-Reranker-0.6B"));
        assertEquals(LlamaModel.Weights.EMBEDDINGGEMMA_2, LlamaModel.Weights.named("embeddinggemma-2"));
        assertEquals(LlamaModel.Weights.QWEN3_RERANKER_4B, LlamaModel.Weights.named("qwen3-reranker-4b"));
        assertFalse(LlamaModel.Weights.QWEN3_EMBEDDING_0_6B.vision());
        assertTrue(LlamaModel.Weights.MEDGEMMA_4B.vision());
        assertEquals(LlamaModel.Weights.OCRIA_1, LlamaModel.Weights.named("ocria-1"));
        assertTrue(LlamaModel.Weights.OCRIA_1.vision());
        assertFalse(LlamaModel.Weights.QWEN3_4B.vision());
        assertNull(LlamaModel.Weights.QWEN3_4B.mmproj());  // text only: no mmproj, and none in its download command
        assertEquals("hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                LlamaModel.Weights.QWEN3_4B.downloadCommand());
        assertEquals("hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
                LlamaModel.Weights.QWEN3_VL_4B.downloadCommand());
    }
}
