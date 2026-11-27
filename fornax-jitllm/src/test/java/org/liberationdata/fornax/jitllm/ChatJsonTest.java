package org.liberationdata.fornax.jitllm;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Json;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatJsonTest {

    @Test void parsesMessagesAndSampling() {
        ChatJson.Request r = ChatJson.parse("""
                {"model": "x", "messages": [{"role": "system", "content": "Be brief."},
                 {"role": "user", "content": [{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}]}],
                 "max_completion_tokens": 20, "temperature": 0, "top_p": 0.5, "seed": 7, "stop": ["\\n\\n"]}""");
        assertEquals(2, r.messages().size());
        assertEquals("system", r.messages().get(0).role());
        assertEquals("Hi there", r.messages().get(1).content());
        assertEquals(20, r.maxTokens());
        assertEquals(0f, r.temperature());
        assertEquals(0.5f, r.topP());
        assertEquals(7, r.seed());
        assertEquals(List.of("\n\n"), r.stop());
    }

    @Test void defaultsWhatIsLeftOut() {
        ChatJson.Request r = ChatJson.parse("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"stop\": \"END\"}");
        assertEquals(Integer.MAX_VALUE, r.maxTokens(), "no limit but the context, as llama.cpp");
        assertEquals(-1, r.seed());
        assertEquals(0f, r.topP());
        assertNull(r.thinking());
        assertEquals(List.of("END"), r.stop());
    }

    @Test void readsThinkingAndClampsHugeTokenCounts() {
        assertEquals(false, ChatJson.parse("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"enable_thinking\": false}").thinking());
        assertEquals(true, ChatJson.parse("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"enable_thinking\": true}").thinking());
        assertEquals(Integer.MAX_VALUE, ChatJson.parse("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1e12}").maxTokens());
    }

    @Test void toolsThatMayNotBeCalledAreIgnored() {
        ChatJson.Request r = ChatJson.parse("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], "
                + "\"tools\": [{\"type\": \"function\"}], \"tool_choice\": \"none\"}");
        assertEquals(1, r.messages().size());
    }

    @Test void refusesWhatJitllmCannotDoAsABadRequest() {
        for (String req : List.of(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"tools\": [{\"type\": \"function\"}]}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"json_schema\": {\"type\": \"object\"}}",
                "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\"}]}]}",
                "{\"messages\": []}",
                "not json")) {
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> ChatJson.parse(req), req);
            assertTrue(e.getMessage().contains("(rc=7): "), e.getMessage());
        }
    }

    @Test void responseHasLlamaModelsShape() {
        Object doc = Json.parse(ChatJson.response(new ChatJson.Reply("Hello \"you\"", 5, 3, "length", false)));
        assertEquals("assistant", Json.get(doc, "message", "role"));
        assertEquals("Hello \"you\"", Json.get(doc, "message", "content"));
        assertNull(Json.get(doc, "message", "reasoning_content"));
        assertEquals("length", Json.get(doc, "finish_reason"));
        assertEquals(5L, Json.get(doc, "usage", "prompt_tokens"));
        assertEquals(3L, Json.get(doc, "usage", "completion_tokens"));
    }

    @Test void splitsALeadingThinkBlock() {
        Object doc = Json.parse(ChatJson.response(new ChatJson.Reply("<think>\nhmm\n</think>\n\nAnswer", 1, 1, "stop", false)));
        assertEquals("hmm", Json.get(doc, "message", "reasoning_content"));
        assertEquals("Answer", Json.get(doc, "message", "content"));
    }

    @Test void streamsAnOrdinaryReplyAtOnce() {
        List<String> out = new ArrayList<>();
        var c = new ChatJson.Content(out::add, false);
        c.accept(" ");
        c.accept("Let me");
        assertEquals(List.of(" Let me"), out, "nothing held back past the first characters");
        c.accept(" explain");
        c.finish();
        assertEquals(" Let me explain", String.join("", out));
        assertEquals("<b>bold</b>", streamed(false, "<", "b>bold</b>"));
    }

    @Test void streamsOnlyTheContent() {
        assertEquals("The answer", streamed(false, "<th", "ink>", "let me", " see</thi", "nk>\n\nThe", " answer"));
        assertEquals("", streamed(false, "<think>never closed"));
        assertEquals("51 ok", streamed(false, "<think>To solve it</think>", "", "\n\n", "51", " ok"));
    }

    @Test void splitsAReplyThatStartsInsideItsReasoning() {
        Object doc = Json.parse(ChatJson.response(new ChatJson.Reply("Okay, the user asks.\n</think>\n\n51", 1, 1, "stop", true)));
        assertEquals("Okay, the user asks.", Json.get(doc, "message", "reasoning_content"));
        assertEquals("51", Json.get(doc, "message", "content"));
        assertEquals("51", streamed(true, "Okay,", " the user asks.\n</th", "ink>\n\n", "51"));
    }

    @Test void aClosingTagInAnOrdinaryReplyIsText() {
        String text = "It closes the reasoning: </think> marks its end.";
        Object doc = Json.parse(ChatJson.response(new ChatJson.Reply(text, 1, 1, "stop", false)));
        assertEquals(text, Json.get(doc, "message", "content"));
        assertNull(Json.get(doc, "message", "reasoning_content"));
        assertEquals(text, streamed(false, "It closes the reasoning: </th", "ink> marks its end."));
    }

    @Test void refusesToolMessagesAndUnknownRoles() {
        for (String req : List.of(
                "{\"messages\": [{\"role\": \"tool\", \"content\": \"42\"}]}",
                "{\"messages\": [{\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"id\": \"1\"}]}]}",
                "{\"messages\": [{\"role\": \"wizard\", \"content\": \"hi\"}]}")) {
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> ChatJson.parse(req), req);
            assertTrue(e.getMessage().contains("(rc=7): "), e.getMessage());
        }
        assertEquals("developer", ChatJson.parse("{\"messages\": [{\"role\": \"developer\", \"content\": \"x\"}]}").messages().get(0).role());
    }

    private static String streamed(boolean opensInside, String... pieces) {
        List<String> out = new ArrayList<>();
        var c = new ChatJson.Content(out::add, opensInside);
        for (String p : pieces) c.accept(p);
        c.finish();
        return String.join("", out);
    }
}
