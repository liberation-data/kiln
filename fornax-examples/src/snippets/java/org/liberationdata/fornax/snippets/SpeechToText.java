package org.liberationdata.fornax.snippets;

// --8<-- [start:transcribe]
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.whisper.Transcript;
import org.liberationdata.fornax.whisper.WhisperModel;

import java.io.File;
// --8<-- [end:transcribe]
// --8<-- [start:qwen3asr]
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;

import java.util.List;
// --8<-- [end:qwen3asr]
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

/** The snippets of website/src/content/docs/docs/guides/speech-to-text.md: compiled, never run. */
class SpeechToText {
    void transcribe() throws Exception {
        // --8<-- [start:transcribe]
        try (var stt = new WhisperModel(WhisperModel.Weights.LARGE_V3_TURBO)) {
            Transcript t = stt.transcribe(Audio.readWav(new File("talk.wav")));
            System.out.println(t.text());
            for (Transcript.Word w : t.words())
                System.out.printf("%6.2f %6.2f %4.2f %s%n", w.start(), w.end(), w.probability(), w.text());
        }
        // --8<-- [end:transcribe]
    }

    void subtitles(Transcript t) throws Exception {
        // --8<-- [start:subtitles]
        Files.writeString(Path.of("talk.srt"), t.srt());          // one cue per segment
        Files.writeString(Path.of("talk.vtt"), t.vtt());          // WebVTT
        Files.writeString(Path.of("talk.words.tsv"), t.wordsTsv()); // start <TAB> end <TAB> word, one word per line
        // --8<-- [end:subtitles]
    }

    void files() {
        // --8<-- [start:files]
        new WhisperModel("/models/ggml-small.en.bin", WhisperModel.Alignment.SMALL_EN);
        new WhisperModel("/models/ggml-custom.bin");                 // no DTW
        new WhisperModel("/models/ggml-small.bin", WhisperModel.Alignment.SMALL, false);   // on the CPU
        // --8<-- [end:files]
    }

    void references() {
        // --8<-- [start:references]
        new WhisperModel("whisper-base.en");                                   // an alias, with DTW
        new WhisperModel("ggerganov/whisper.cpp:ggml-large-v3-turbo-q8_0.bin");  // a file in a repo, with DTW
        new WhisperModel("ggerganov/whisper.cpp:ggml-small.en.bin", WhisperModel.Alignment.SMALL_EN);
        // --8<-- [end:references]
    }

    WhisperModel.Options options() {
        // --8<-- [start:options]
        var options = WhisperModel.Options.DEFAULT
                .withLanguage("de")             // null or "auto" detects it; an English-only model always transcribes English
                .withTranslate(true)            // German speech, English text
                .withInitialPrompt("Fornax, ggml, ComfyUI")   // names and spellings to expect
                .withBeamSize(1)                // greedy: faster; 5 (the default) is a little more accurate
                .withThreads(8)                 // 0 or less is whisper.cpp's default
                .withTemperature(-1)            // below 0 is the default: 0, raised when a window fails its checks
                .withVad(WhisperModel.VadWeights.SILERO_V6_2_0);   // transcribe only the speech (see below)
        // --8<-- [end:options]
        return options;
    }

    WhisperModel.Options[] vad() {
        // --8<-- [start:vad]
        var options = WhisperModel.Options.DEFAULT.withVad(WhisperModel.VadWeights.SILERO_V6_2_0);
        // or a file, with whisper.cpp's VAD settings to tune
        var tuned = WhisperModel.Options.DEFAULT.withVad(WhisperModel.Vad.of("/models/ggml-silero-v6.2.0.bin")
                .withThreshold(0.5f)            // speech probability for a 32 ms frame to count as speech
                .withMinSpeechMs(250)           // shorter speech is dropped
                .withMinSilenceMs(100)          // this much silence ends a segment
                .withSpeechPadMs(30));          // padding either side of each segment
        // --8<-- [end:vad]
        return new WhisperModel.Options[] {options, tuned};
    }

    Transcript listener(WhisperModel stt, Audio audio) {
        // --8<-- [start:listener]
        Transcript t = stt.transcribe(audio, WhisperModel.Options.DEFAULT, new WhisperModel.Listener() {
            @Override public void progress(int percent) { System.out.printf("%d%%%n", percent); }
            @Override public void segment(Transcript.Segment s) { System.out.println(s.text()); }
        });
        // --8<-- [end:listener]
        return t;
    }

    void qwen3Asr() throws Exception {
        // --8<-- [start:qwen3asr]
        try (var asr = new LlamaModel(LlamaModel.Weights.QWEN3_ASR_0_6B)) {
            var greedy = LlamaModel.Options.DEFAULT.withTemperature(0).withMaxTokens(1024);
            Audio talk = Audio.readWav(new File("talk.wav"));     // any rate and channels: resampled to 16 kHz mono
            String text = asr.chat(List.of(Message.user("Transcribe audio to text", talk)), greedy).text();
            System.out.println(text);
        }
        // --8<-- [end:qwen3asr]
    }

    String qwen3AsrJson(LlamaModel asr) throws Exception {
        // --8<-- [start:qwen3asrjson]
        String mp3 = Base64.getEncoder().encodeToString(Files.readAllBytes(Path.of("talk.mp3")));
        String reply = asr.chatJson("""
            {"messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "%s", "format": "mp3"}},
                {"type": "text", "text": "Transcribe audio to text"}]}],
             "temperature": 0, "max_tokens": 1024}""".formatted(mp3), null);
        // --8<-- [end:qwen3asrjson]
        return reply;
    }
}
