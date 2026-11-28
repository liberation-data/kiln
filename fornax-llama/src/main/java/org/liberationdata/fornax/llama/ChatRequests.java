package org.liberationdata.fornax.llama;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.util.List;

/** Builds {@link LlamaModel#chatJson} requests from {@link Message}s (apart from LlamaModel, so it needs no native library). */
final class ChatRequests {
    private ChatRequests() {}

    /**
     * The request for {@link LlamaModel#extract}. Adds the messages' images to images and their audio to audio, in the
     * order the request uses them.
     */
    static String extract(List<Message> messages, String jsonSchema, Options o, List<Image> images, List<Audio> audio) {
        return "{\"messages\": " + messages(messages, images, audio) + ", \"json_schema\": " + jsonSchema + sampling(o) + "}";
    }

    /**
     * The request for {@link LlamaModel#chat} when it goes through chatJson: with {@code thinking} false (chat's
     * sampling, the template's thinking off), or with audio, which only chatJson's path takes.
     */
    static String chat(List<Message> messages, Options o, List<Image> images, List<Audio> audio) {
        return "{\"messages\": " + messages(messages, images, audio) + sampling(o) + "}";
    }

    /** A chatJson response as {@link LlamaModel#chat} gives it: the message's content and the token counts. */
    static LlamaModel.Result result(String responseJson) {
        Object r = Json.parse(responseJson);
        Object text = Json.get(r, "message", "content");
        return new LlamaModel.Result(text instanceof String s ? s : "", count(r, "prompt_tokens"), count(r, "completion_tokens"));
    }

    private static int count(Object response, String name) {
        return Json.get(response, "usage", name) instanceof Number n ? n.intValue() : 0;
    }

    // As fornax_llama_chat samples: no top-k, temperature <= 0 greedy. enable_thinking only when it's off, so a request
    // with thinking on is the same as before there was a switch.
    private static String sampling(Options o) {
        return ", \"max_tokens\": " + o.maxTokens() + ", \"temperature\": " + Math.max(0f, o.temperature())
                + ", \"top_p\": " + o.topP() + ", \"min_p\": " + o.minP() + ", \"top_k\": 0, \"seed\": " + o.seed()
                + (o.thinking() ? "" : ", \"enable_thinking\": false");
    }

    /**
     * OpenAI-format messages, with {"type": "image", "index": i} parts for their images (added to images), then
     * {"type": "audio", "index": i} parts for their audio (added to audio), before the text.
     */
    static String messages(List<Message> messages, List<Image> images, List<Audio> audio) {
        StringBuilder b = new StringBuilder("[");
        for (Message m : messages) {
            if (b.length() > 1) b.append(", ");
            b.append("{\"role\": ").append(Json.quote(m.role())).append(", \"content\": ");
            String text = m.content() == null ? "" : m.content();
            if (m.images().isEmpty() && m.audio().isEmpty()) {
                b.append(Json.quote(text));
            } else {
                b.append('[');
                for (Image img : m.images()) {
                    b.append("{\"type\": \"image\", \"index\": ").append(images.size()).append("}, ");
                    images.add(img);
                }
                for (Audio a : m.audio()) {
                    b.append("{\"type\": \"audio\", \"index\": ").append(audio.size()).append("}, ");
                    audio.add(a);
                }
                b.append("{\"type\": \"text\", \"text\": ").append(Json.quote(text)).append("}]");
            }
            b.append('}');
        }
        return b.append(']').toString();
    }
}
