import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.whisper.WhisperModel;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Word error rate and speed of speech recognition models on the explainer's narration: seven clips of synthesized
 * speech (videos/explainer/narration/1.mp3 to 7.mp3, 2.2 minutes in all), each against its exact script (N.txt, without
 * the voice's name on its first line or the [pause] marks). Words are compared in lower case with punctuation as
 * spaces and apostrophes dropped. Each clip is decoded with ffmpeg and given to the model as the same {@link Audio};
 * the times are per transcription, after one untimed warm-up, and leave out loading. The numbers in the
 * speech-to-text guide come from it.
 *
 * <pre>
 * ./gradlew :fornax-core:jar :fornax-llama:jar :fornax-whisper:jar
 * java --enable-native-access=ALL-UNNAMED -Dfornax.library.path=build/native/linux-x86_64-vulkan \
 *     -cp fornax-core/build/libs/fornax-core-VERSION.jar:fornax-llama/build/libs/fornax-llama-VERSION.jar:fornax-whisper/build/libs/fornax-whisper-VERSION.jar \
 *     scripts/AsrEval.java whisper large-v3-turbo [BEAM]
 * java ... scripts/AsrEval.java llama qwen3-asr-0.6b|qwen3-asr-1.7b
 * </pre>
 * Run from the repository root, with ffmpeg on the PATH. BEAM is Whisper's beam size (5, its default, or 1: greedy).
 * Qwen3-ASR runs greedy, as its model card does.
 */
public class AsrEval {
    static final Path DIR = Path.of("videos/explainer/narration");

    public static void main(String[] args) throws Exception {
        if (!(args.length >= 2 && args.length <= 3 && args[0].equals("whisper") || args.length == 2 && args[0].equals("llama"))) {
            System.err.println("usage: AsrEval whisper WEIGHTS [BEAM]   (WEIGHTS: large-v3-turbo, base.en, ...)");
            System.err.println("       AsrEval llama WEIGHTS            (WEIGHTS: qwen3-asr-0.6b, qwen3-asr-1.7b)");
            System.exit(2);
        }
        List<Audio> clips = new ArrayList<>();
        List<String> scripts = new ArrayList<>();
        Path tmp = Files.createTempDirectory("asr-eval");
        for (int i = 1; i <= 7; i++) {
            clips.add(decode(DIR.resolve(i + ".mp3"), tmp.resolve(i + ".wav")));
            List<String> lines = Files.readAllLines(DIR.resolve(i + ".txt"));
            scripts.add(String.join(" ", lines.subList(1, lines.size())).replaceAll("\\[pause[^]]*]", " "));
        }
        long t0 = System.nanoTime();
        if (args[0].equals("whisper")) {
            var o = WhisperModel.Options.DEFAULT.withBeamSize(args.length == 3 ? Integer.parseInt(args[2]) : 5);
            try (var m = new WhisperModel(WhisperModel.Weights.named(args[1]))) {
                System.out.printf("%s, beam %d: loaded in %.2fs%n", args[1], o.beamSize(), (System.nanoTime() - t0) / 1e9);
                run(clips, scripts, a -> m.transcribe(a, o).text());
            }
        } else {
            Options greedy = Options.DEFAULT.withTemperature(0).withMaxTokens(1024);
            try (var m = new LlamaModel(LlamaModel.Weights.named(args[1]))) {
                System.out.printf("%s: loaded in %.2fs%n", args[1], (System.nanoTime() - t0) / 1e9);
                run(clips, scripts, a -> m.chat(List.of(Message.user("Transcribe audio to text", a)), greedy).text());
            }
        }
    }

    static void run(List<Audio> clips, List<String> scripts, Function<Audio, String> transcribe) {
        transcribe.apply(clips.get(0));   // warm-up: shaders, first allocations
        int errors = 0, words = 0;
        double seconds = 0, audio = 0;
        for (int i = 0; i < clips.size(); i++) {
            long t = System.nanoTime();
            String text = transcribe.apply(clips.get(i)).strip();
            double s = (System.nanoTime() - t) / 1e9;
            List<String> ref = words(scripts.get(i));
            int e = distance(ref, words(text));
            errors += e;
            words += ref.size();
            seconds += s;
            audio += clips.get(i).seconds();
            System.out.printf("clip %d: %5.1fs of audio in %5.2fs, WER %5.1f%% (%d/%d)  %s%n", i + 1, clips.get(i).seconds(), s,
                    100.0 * e / ref.size(), e, ref.size(), text);
        }
        System.out.printf("all: WER %.1f%% (%d/%d words), %.2fs per clip on average, %.0fx real time%n", 100.0 * errors / words,
                errors, words, seconds / clips.size(), audio / seconds);
    }

    /** Lower case, punctuation as spaces (so "X-ray" is two words either way), apostrophes dropped. */
    static List<String> words(String s) {
        String n = s.toLowerCase().replaceAll("['’]", "").replaceAll("[^\\p{L}\\p{N}]+", " ").strip();
        return n.isEmpty() ? List.of() : List.of(n.split(" "));
    }

    /** Word edit distance: substitutions, insertions and deletions. */
    static int distance(List<String> r, List<String> h) {
        int[] prev = new int[h.size() + 1], cur = new int[h.size() + 1];
        for (int j = 0; j <= h.size(); j++) prev[j] = j;
        for (int i = 1; i <= r.size(); i++) {
            cur[0] = i;
            for (int j = 1; j <= h.size(); j++)
                cur[j] = Math.min(Math.min(cur[j - 1], prev[j]) + 1, prev[j - 1] + (r.get(i - 1).equals(h.get(j - 1)) ? 0 : 1));
            int[] t = prev;
            prev = cur;
            cur = t;
        }
        return prev[h.size()];
    }

    static Audio decode(Path mp3, Path wav) throws Exception {
        Process p = new ProcessBuilder("ffmpeg", "-v", "error", "-i", mp3.toString(), "-y", wav.toString()).inheritIO().start();
        if (p.waitFor() != 0) throw new IllegalStateException("ffmpeg failed on " + mp3);
        return Audio.readWav(wav.toFile());
    }
}
