package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.Wire;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Audio in messages, requests and the worker's wire, and its conversion for the encoder: no native library needed. */
class AudioInputTest {
    static final Audio BEEP = Audio.ofMono(new float[]{0f, 0.5f, -0.25f, 1f}, 16000);
    static final Image RED = new Image(1, 1, new byte[]{(byte) 255, 0, 0});

    @Test
    void messagesCarryAudioAndStayCompatibleWithoutIt() {
        Message m = Message.user("Transcribe audio to text", BEEP);
        assertEquals(List.of(BEEP), m.audio());
        assertEquals(List.of(), m.images());
        assertEquals(List.of(), Message.user("hi").audio());
        assertEquals(List.of(), Message.user("look", RED).audio());       // the image overload, as before
        assertEquals(List.of(), new Message("user", "hi", List.of(RED)).audio());
        assertEquals(List.of(), new Message("user", "hi", null, null).audio());
        var list = new ArrayList<>(List.of(BEEP));
        Message copy = new Message("user", "x", List.of(), list);
        list.clear();
        assertEquals(List.of(BEEP), copy.audio());                         // copied, as images are
    }

    @Test
    void requestsPutImagesThenAudioBeforeTextAndNumberEachInOrder() {
        Audio other = Audio.ofMono(new float[]{0.1f}, 8000);
        List<Image> images = new ArrayList<>();
        List<Audio> audio = new ArrayList<>();
        String json = ChatRequests.messages(List.of(Message.system("Be brief."),
                new Message("user", "What is said about the picture?", List.of(RED), List.of(BEEP, other)),
                Message.assistant("ok"), Message.user("And here?", other)), images, audio);
        assertEquals(List.of(RED), images);
        assertEquals(List.of(BEEP, other, other), audio);
        assertEquals(Json.parse("""
            [{"role": "system", "content": "Be brief."},
             {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "audio", "index": 0},
                                          {"type": "audio", "index": 1},
                                          {"type": "text", "text": "What is said about the picture?"}]},
             {"role": "assistant", "content": "ok"},
             {"role": "user", "content": [{"type": "audio", "index": 2}, {"type": "text", "text": "And here?"}]}]"""),
                Json.parse(json));
    }

    @Test
    void aChatWithAudioKeepsChatsSamplingAndThinking() {
        List<Audio> audio = new ArrayList<>();
        Object req = Json.parse(ChatRequests.chat(List.of(Message.user("Transcribe audio to text", BEEP)),
                Options.DEFAULT.withTemperature(0).withSeed(5), new ArrayList<>(), audio));
        assertEquals(List.of(BEEP), audio);
        assertEquals(0.0, Json.get(req, "temperature"));
        assertEquals(5L, Json.get(req, "seed"));
        assertEquals(0L, Json.get(req, "top_k"));
        assertNull(Json.get(req, "enable_thinking"));   // thinking on: the template as it is
        assertEquals("audio", Json.get(req, "messages", 0, "content", 0, "type"));
        List<Audio> extracted = new ArrayList<>();
        ChatRequests.extract(List.of(Message.user("Who speaks?", BEEP)), "{\"type\": \"object\"}", Options.DEFAULT,
                new ArrayList<>(), extracted);
        assertEquals(List.of(BEEP), extracted);
    }

    @Test
    void chatJsonArgumentsRoundTripAudioAsMono() {
        Audio stereo = new Audio(new float[]{1f, 0f, -1f}, new float[]{0f, 0f, 1f}, 22050);
        var in = new Wire.In(LlamaService.chatJsonArgs("{\"messages\": []}", List.of(RED), List.of(BEEP, stereo), true));
        assertEquals("{\"messages\": []}", in.string());
        LlamaServiceTest.assertImages(List.of(RED), LlamaService.images(in));
        List<Audio> back = LlamaService.audio(in);
        assertEquals(2, back.size());
        assertArrayEquals(BEEP.left(), back.get(0).left());
        assertArrayEquals(BEEP.left(), back.get(0).right());
        assertEquals(16000, back.get(0).sampleRate());
        assertArrayEquals(new float[]{0.5f, 0f, 0f}, back.get(1).left());   // the sides mixed down
        assertEquals(22050, back.get(1).sampleRate());
        assertTrue(in.bool());

        in = new Wire.In(LlamaService.chatJsonArgs("{}", List.of(), List.of(), false));
        assertEquals("{}", in.string());
        assertEquals(List.of(), LlamaService.images(in));
        assertEquals(List.of(), LlamaService.audio(in));
        assertFalse(in.bool());
    }

    @Test
    void pcmIsMonoAtTheEncodersRate() {
        // mono at the encoder's rate goes as it is, bit for bit
        assertArrayEquals(BEEP.left(), LlamaModel.pcm(BEEP, 16000));
        // a stereo 48 kHz tone: mixed down, then a third as many samples, the tone kept
        int n = 4800;
        float[] l = new float[n], r = new float[n];
        for (int i = 0; i < n; i++) {
            l[i] = (float) Math.sin(2 * Math.PI * 440 * i / 48000.0);
            r[i] = -l[i] * 0.5f + 0.2f;   // mixed: 0.25 sin + 0.1
        }
        float[] pcm = LlamaModel.pcm(new Audio(l, r, 48000), 16000);
        assertEquals(n / 3, pcm.length);
        for (int i = 100; i < pcm.length - 100; i++)   // away from the edges, where the filter runs out of input
            assertEquals(0.25 * Math.sin(2 * Math.PI * 440 * i / 16000.0) + 0.1, pcm[i], 2e-3, "sample " + i);
    }

    @Test
    void asrWeightsTakeAudioAndNoImages() {
        assertEquals(Weights.QWEN3_ASR_0_6B, Weights.named("qwen3-asr-0.6b"));
        assertEquals(Weights.QWEN3_ASR_1_7B, Weights.named("Qwen3-ASR-1.7B"));
        assertTrue(Weights.QWEN3_ASR_0_6B.audio());
        assertFalse(Weights.QWEN3_ASR_0_6B.vision());
        assertTrue(Weights.QWEN3_VL_4B.vision());
        assertFalse(Weights.QWEN3_VL_4B.audio());
        assertFalse(Weights.QWEN3_4B.audio());
        for (Weights w : Weights.values()) {   // only the ASR weights take audio, and every other mmproj is a vision one
            boolean asr = w.name().startsWith("QWEN3_ASR_");
            assertEquals(asr, w.audio(), w.name());
            assertEquals(!asr && w.mmprojFile != null, w.vision(), w.name());
        }
        assertEquals("hf download ggml-org/Qwen3-ASR-0.6B-GGUF Qwen3-ASR-0.6B-Q8_0.gguf mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
                Weights.QWEN3_ASR_0_6B.downloadCommand());
    }
}
