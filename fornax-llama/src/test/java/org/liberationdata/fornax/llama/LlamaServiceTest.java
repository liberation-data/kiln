package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Scored;
import org.liberationdata.fornax.Wire;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** An isolated model's encoding of its calls, both ways, as LlamaService decodes them: no native library needed. */
class LlamaServiceTest {
    static final Image RED = new Image(1, 2, new byte[]{(byte) 255, 0, 0, (byte) 255, 0, 0});
    static final Image BLUE = new Image(2, 1, new byte[]{0, 0, (byte) 255, 0, 0, (byte) 255});
    static final Options ODD = new Options(-1, 0f, 0.5f, 0.25f, 42);

    @Test
    void chatArgumentsRoundTripWithImagesNullContentAndOptions() {
        var messages = List.of(Message.system("be brief"), Message.user("which colour? ünïcode", RED, BLUE),
                new Message("user", null, List.of(BLUE)), Message.assistant(""));
        var in = new Wire.In(LlamaService.chatArgs(messages, ODD, true));
        List<Message> back = LlamaService.messages(in);
        assertEquals(messages.size(), back.size());
        for (int i = 0; i < messages.size(); i++) {
            assertEquals(messages.get(i).role(), back.get(i).role());
            assertEquals(messages.get(i).content(), back.get(i).content());
            assertImages(messages.get(i).images(), back.get(i).images());
        }
        assertEquals(ODD, LlamaService.options(in));
        assertTrue(in.bool());
    }

    @Test
    void chatJsonAndCompleteArgumentsRoundTrip() {
        var in = new Wire.In(LlamaService.chatJsonArgs("{\"messages\": []}", List.of(RED), List.of(), false));
        assertEquals("{\"messages\": []}", in.string());
        assertImages(List.of(RED), LlamaService.images(in));
        assertFalse(in.bool());

        in = new Wire.In(LlamaService.completeArgs("<__media__> what?", List.of(), Options.DEFAULT, true));
        assertEquals("<__media__> what?", in.string());
        assertEquals(List.of(), LlamaService.images(in));
        assertEquals(Options.DEFAULT, LlamaService.options(in));
        assertTrue(in.bool());
    }

    @Test
    void resultsRoundTrip() {
        var r = new Result("forty-two ✓", 17, 3);
        assertEquals(r, LlamaService.decodeResult(LlamaService.result(r)));
        assertEquals("", LlamaService.decodeResult(LlamaService.result(new Result("", 0, 0))).text());
    }

    @Test
    void embedAndRerankArgumentsRoundTrip() {
        var texts = Arrays.asList("glaze ✓", "", null);
        var in = new Wire.In(LlamaService.embedArgs(texts, EmbedOptions.query("find kilns").withNormalize(false)));
        assertEquals(texts, LlamaService.strings(in));
        assertEquals("Instruct: find kilns\nQuery:", in.string());
        assertFalse(in.bool());
        in = new Wire.In(LlamaService.embedArgs(List.of("x"), EmbedOptions.DEFAULT));
        assertEquals(List.of("x"), LlamaService.strings(in));
        assertNull(in.string());
        assertTrue(in.bool());

        in = new Wire.In(LlamaService.rerankArgs("how hot?", List.of("1200 C", "")));
        assertEquals("how hot?", in.string());
        assertEquals(List.of("1200 C", ""), LlamaService.strings(in));
    }

    @Test
    void embeddingsAndScoresRoundTrip() {
        float[][] v = {{1f, -0.5f, Float.MIN_VALUE}, {0f, Float.NaN, 3.25f}};
        var back = LlamaService.decodeEmbeddings(LlamaService.embeddings(new LlamaModel.Embeddings(v, 17)));
        assertEquals(2, back.vectors().length);
        for (int i = 0; i < v.length; i++) assertArrayEquals(v[i], back.vectors()[i]);
        assertEquals(17, back.tokens());
        var none = LlamaService.decodeEmbeddings(LlamaService.embeddings(new LlamaModel.Embeddings(new float[0][], 0)));
        assertEquals(0, none.vectors().length);
        assertEquals(0, none.tokens());

        var scores = List.of(new Scored(2, 0.99f), new Scored(0, 0.5f), new Scored(1, 0f));
        assertEquals(scores, LlamaService.decodeScores(LlamaService.scores(scores)));
        assertEquals(List.of(), LlamaService.decodeScores(LlamaService.scores(List.of())));
    }

    static void assertImages(List<Image> expected, List<Image> actual) {
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i).width(), actual.get(i).width());
            assertEquals(expected.get(i).height(), actual.get(i).height());
            assertArrayEquals(expected.get(i).rgb(), actual.get(i).rgb());
        }
    }
}
