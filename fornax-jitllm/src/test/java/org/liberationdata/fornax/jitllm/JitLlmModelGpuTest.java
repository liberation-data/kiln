package org.liberationdata.fornax.jitllm;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.liberationdata.fornax.Json;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Runs a real model: ./gradlew :fornax-jitllm:test -PjitllmModel=path/to/model.gguf, with TORNADOVM_HOME set. */
@EnabledIfSystemProperty(named = "jitllm.test.model", matches = ".+")
class JitLlmModelGpuTest {
    private static final String MODEL = System.getProperty("jitllm.test.model");

    @Test void chatsStreamsAndServesChatJson() {
        try (var llm = new JitLlmModel(MODEL, 2048)) {
            StringBuilder streamed = new StringBuilder();
            Result r = llm.chat(List.of(Message.system("Answer in one word."), Message.user("What is the capital of France?")),
                    Options.DEFAULT.withTemperature(0).withMaxTokens(16), streamed::append);
            System.out.println("chat: " + r);
            assertTrue(r.text().contains("Paris"), r.text());
            assertTrue(r.promptTokens() > 0 && r.generatedTokens() > 0);
            assertEquals(r.text(), streamed.toString());

            // a second call starts from an empty context: its prompt is only its own, shorter message
            Result again = llm.chat(List.of(Message.user("Say OK.")), Options.DEFAULT.withTemperature(0).withMaxTokens(8));
            assertTrue(again.promptTokens() < r.promptTokens(), "context carried over: " + again + " after " + r);

            String json = llm.chatJson("""
                    {"messages": [{"role": "user", "content": "Count from 1 to 50, separated by spaces."}],
                     "max_tokens": 8, "temperature": 0}""", null);
            System.out.println("chatJson: " + json);
            Object doc = Json.parse(json);
            assertEquals("length", Json.get(doc, "finish_reason"));
            assertEquals(8L, Json.get(doc, "usage", "completion_tokens"));
            assertFalse(((String) Json.get(doc, "message", "content")).isBlank());
        }
    }

    @Test void aPromptLongerThanTheContextIsABadRequest() {
        try (var llm = new JitLlmModel(MODEL, 64)) {
            String request = "{\"messages\": [{\"role\": \"user\", \"content\": " + Json.quote("kiln ".repeat(200)) + "}]}";
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> llm.chatJson(request, null));
            System.out.println("too long: " + e.getMessage());
            assertTrue(e.getMessage().contains("(rc=2): "), e.getMessage());
        }
    }

    @Test void anEmbeddingInputLongerThanTheContextIsABadRequest() {
        try (var llm = new JitLlmModel(MODEL, 64, JitLlmModel.Mode.EMBED)) {
            assertEquals(llm.embeddingSize(), llm.embed("a kiln").length);
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> llm.embed("kiln ".repeat(200)));
            assertTrue(e.getMessage().contains("more than the context holds"), e.getMessage());
        }
    }

    @Test void thinkingCanBeTurnedOff() {
        String thinker = System.getProperty("jitllm.test.thinkingModel");
        org.junit.jupiter.api.Assumptions.assumeTrue(thinker != null && !thinker.isBlank(), "-PjitllmThinkingModel=<Qwen3 GGUF>");
        try (var llm = new JitLlmModel(thinker, 2048)) {
            String ask = "{\"messages\": [{\"role\": \"user\", \"content\": \"What is 17*3?\"}], \"temperature\": 0, \"max_tokens\": 600";
            Object on = Json.parse(llm.chatJson(ask + "}", null));
            long t0 = System.nanoTime();
            Object off = Json.parse(llm.chatJson(ask + ", \"enable_thinking\": false}", null));
            long t1 = System.nanoTime();
            Object offAgain = Json.parse(llm.chatJson(ask + ", \"enable_thinking\": false}", null));
            long t2 = System.nanoTime();
            Object onAgain = Json.parse(llm.chatJson(ask + ", \"enable_thinking\": true}", null));
            System.out.printf("thinking off after on: %.2fs (session switch), off again: %.2fs%n", (t1 - t0) / 1e9, (t2 - t1) / 1e9);
            System.out.println("thinking on: " + on + "\nthinking off: " + off);
            assertEquals(Json.get(off, "message", "content"), Json.get(offAgain, "message", "content"));
            assertFalse(((String) Json.get(on, "message", "reasoning_content")).isBlank());
            assertTrue(Json.get(off, "message", "reasoning_content") == null
                    || ((String) Json.get(off, "message", "reasoning_content")).isBlank(), off.toString());
            assertTrue(((Long) Json.get(off, "usage", "completion_tokens")) < ((Long) Json.get(on, "usage", "completion_tokens")));
            assertTrue(((String) Json.get(off, "message", "content")).contains("51"), off.toString());
            assertFalse(((String) Json.get(onAgain, "message", "reasoning_content")).isBlank());
        }
    }

    @Test void aFileJitllmHasNoModelForIsABadRequest(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) throws Exception {
        java.nio.file.Path gguf = dir.resolve("vision.gguf");   // a GGUF whose architecture jitllm doesn't run
        java.io.ByteArrayOutputStream b = new java.io.ByteArrayOutputStream();
        java.nio.ByteBuffer h = java.nio.ByteBuffer.allocate(64).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        h.put("GGUF".getBytes()).putInt(3).putLong(0).putLong(1);
        b.write(h.array(), 0, h.position());
        byte[] key = "general.architecture".getBytes(), value = "clip".getBytes();
        java.nio.ByteBuffer kv = java.nio.ByteBuffer.allocate(64).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        kv.putLong(key.length).put(key).putInt(8).putLong(value.length).put(value);
        b.write(kv.array(), 0, kv.position());
        java.nio.file.Files.write(gguf, b.toByteArray());
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> new JitLlmModel(gguf.toString(), 256).close());
        System.out.println("unsupported: " + e.getMessage());
        assertTrue(e.getMessage().contains("(rc=7)") && e.getMessage().contains("clip"), e.getMessage());
    }

    @Test void cancelStopsAGeneration() throws Exception {
        try (var llm = new JitLlmModel(MODEL, 2048)) {
            var f = llm.chatAsync(List.of(Message.user("Write a very long story about a kiln.")),
                    Options.DEFAULT.withTemperature(0).withMaxTokens(2000));
            Thread.sleep(3000);
            llm.cancel();
            Result r = f.get();
            System.out.println("cancelled after " + r.generatedTokens() + " tokens");
            assertTrue(r.generatedTokens() < 2000, r.toString());
        }
    }
}
