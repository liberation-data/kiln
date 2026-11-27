package org.liberationdata.fornax.snippets;

import org.liberationdata.fornax.Scored;
import org.liberationdata.fornax.Vectors;
import org.liberationdata.fornax.acestep.AceStepModel;
import org.liberationdata.fornax.jitllm.JitLlmModel;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.sd.DiffusionModel;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.List;

/** The snippets of README.md: compiled, never run. */
class Readme {
    void chat() {
        // --8<-- [start:chat]
        try (var llm = new LlamaModel("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")) {   // downloaded on first use, then loaded onto the GPU
            llm.chat(List.of(Message.user("Name a country song about a robot.")), Options.DEFAULT, System.out::print);
        }
        // --8<-- [end:chat]
    }

    void jitllm() {
        // --8<-- [start:jitllm]
        try (var llm = new JitLlmModel("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")) {   // same messages, options and result as LlamaModel
            llm.chat(List.of(Message.user("Name a country song about a robot.")), Options.DEFAULT, System.out::print);
        }
        // --8<-- [end:jitllm]
    }

    void models(String requestJson, String invoiceSchema) throws Exception {
        // --8<-- [start:models]
        try (var songs = new AceStepModel("ace-step")) {   // a catalogue set: its four files, downloaded on first use
            songs.generate(requestJson).audio().writeWav(new File("song.wav"));
        }

        try (var images = new DiffusionModel(DiffusionModel.Config.builder().model("flux2-klein").build())) {
            BufferedImage img = images.generate(new DiffusionModel.Request("a neon sign", 1024, 1024, 20, 4.0, 7)).get(0);
        }

        try (var vlm = new LlamaModel(LlamaModel.Weights.QWEN3_VL_4B)) {   // the GGUF plus its mmproj (vision encoder)
            String invoiceJson = vlm.extract(List.of(Message.user("Read this invoice.", Image.read(Path.of("invoice.png")))),
                    invoiceSchema, Options.DEFAULT.withTemperature(0));      // constrained to a JSON schema
        }
        // --8<-- [end:models]
    }

    void search(List<String> passages, String question, List<String> candidates) {
        // --8<-- [start:search]
        try (var emb = new LlamaModel(LlamaModel.Weights.QWEN3_EMBEDDING_0_6B, LlamaModel.Mode.EMBED)) {
            float[][] docs = emb.embed(passages);                                       // L2-normalised, batched
            float[] q = emb.embed(List.of(question), EmbedOptions.query("Given a question, retrieve passages that answer it"))[0];
            List<Scored> nearest = Vectors.topK(q, docs, 20);
        }
        try (var rr = new LlamaModel(LlamaModel.Weights.QWEN3_RERANKER_0_6B, LlamaModel.Mode.RERANK)) {
            List<Scored> best = rr.rerank(question, candidates);                        // best first
        }
        // --8<-- [end:search]
    }
}
