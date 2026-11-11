import array
import unittest
import wave
from io import BytesIO

from sysaudio_stt.audio import Resampler, downmix_s16, pcm16_to_wav, strip_overlap
from sysaudio_stt.capture import parse_header
from sysaudio_stt.cli import apply_script, missing_credential_text, parse_env_assignments, resolve_language, select_model, select_output
from sysaudio_stt.local_polish import apply_glossary, collapse_repeats, digits_to_hanzi
from sysaudio_stt.polish import accept_polish, build_polish_messages, load_llm_config, parse_model_ids, parse_responses_text
from sysaudio_stt.devices import blackhole_routed_name, routing_advice
from sysaudio_stt.stt import (
    build_transcription_body,
    drop_hallucination,
    needs_api_key,
    parse_transcription_response,
    redact,
    resolve_transformers_model,
)
from sysaudio_stt.vad import Segmenter


def pcm_const(n, value, channels=1):
    data = array.array("h", [value] * (n * channels))
    return data.tobytes()


class OutputChoiceTests(unittest.TestCase):
    def test_blank_choice_picks_highest_rms(self):
        items = [
            {"name": "扬声器", "uid": "spk", "rms": 0, "is_default_system_output": False},
            {"name": "耳机", "uid": "hp", "rms": 3000, "is_default_system_output": True},
        ]
        self.assertEqual(select_output(items, "").get("uid"), "hp")
        self.assertEqual(select_output(items, "1").get("name"), "扬声器")
        self.assertEqual(select_output(items, "耳机").get("uid"), "hp")

    def test_model_choice_defaults_to_turbo(self):
        self.assertEqual(select_model(""), "openai/whisper-small")
        self.assertEqual(select_model("2"), "openai/whisper-large-v3-turbo")
        self.assertEqual(select_model("turbo"), "openai/whisper-large-v3-turbo")

    def test_simplified_language_converts_traditional(self):
        chosen = resolve_language("zh-Hans")
        self.assertEqual(chosen["whisper"], "zh")
        self.assertEqual(apply_script("現在讓我聽", chosen["script"]), "现在让我听")

    def test_polish_prompt_keeps_examples_and_rejects_invention(self):
        messages = build_polish_messages("幻风下来", "zh-cn", "换峰")
        self.assertIn("换峰下来", messages[0]["content"])
        self.assertIn("你看这个八", messages[0]["content"])
        custom = build_polish_messages("原文", "zh-cn", "", "只改标点。")
        self.assertTrue(custom[0]["content"].startswith("只改标点。"))
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from sysaudio_stt.polish import load_polish_prompt, save_polish_prompt
        with TemporaryDirectory() as folder:
            path = Path(folder) / "prompt.txt"
            save_polish_prompt("下次还用这句。", path)
            self.assertEqual(load_polish_prompt(path), "下次还用这句。")
        from sysaudio_stt.polish import load_glossary, save_glossary
        with TemporaryDirectory() as folder:
            path = Path(folder) / "glossary.txt"
            save_glossary("换峰 八号线", path)
            self.assertEqual(load_glossary(path), "换峰 八号线")
        self.assertEqual(accept_polish("幻风下来", ""), "幻风下来")
        self.assertEqual(accept_polish("短句", "这是一段凭空写出来的很长很长的解释" * 8), "短句")

    def test_llm_config_uses_responses_gateway(self):
        config = load_llm_config()
        self.assertEqual(config["env_key"], "LLM_API_KEY")
        self.assertEqual(config["wire_api"], "responses")
        self.assertTrue(config["base_url"].endswith("/api/openai"))
        body = '{"output":[{"type":"message","content":[{"type":"output_text","text":"换峰下来。"}]}]}'.encode("utf-8")
        self.assertEqual(parse_responses_text(body), "换峰下来。")
        ids = parse_model_ids(b'{"data":[{"id":"demo-model"},{"id":"other"}]}')
        self.assertEqual(ids, ["demo-model", "other"])
        self.assertEqual(collapse_repeats("然后然后然后"), "然后")
        self.assertEqual(digits_to_hanzi("你看这个8"), "你看这个八")
        self.assertEqual(apply_glossary("幻风下来", "换峰"), "换峰下来")


class AudioTests(unittest.TestCase):
    def test_downmix_stereo_average(self):
        raw = array.array("h", [1000, 3000, -1000, -3000]).tobytes()
        mixed = array.array("h")
        mixed.frombytes(downmix_s16(raw, 2))
        self.assertEqual(list(mixed), [2000, -2000])

    def test_resample_dc_keeps_level_and_rate(self):
        samples = 4800
        stereo = pcm_const(samples, 1000, channels=2)
        out = Resampler(48000, 16000).push(stereo, 2)
        values = array.array("h")
        values.frombytes(out)
        self.assertGreater(len(values), 1400)
        self.assertLess(len(values), 1800)
        middle = values[100:-100]
        mean = sum(middle) / len(middle)
        self.assertAlmostEqual(mean, 1000, delta=80)

    def test_wav_roundtrip(self):
        raw = pcm_const(320, 100)
        wav = pcm16_to_wav(raw)
        self.assertTrue(wav.startswith(b"RIFF"))
        with wave.open(BytesIO(wav), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnframes(), 320)
            self.assertEqual(handle.readframes(320), raw)

    def test_strip_overlap(self):
        self.assertEqual(strip_overlap("今天我们讨论一下项目", "讨论一下项目的进度"), "的进度")
        self.assertEqual(strip_overlap("", "你好"), "你好")


class VadTests(unittest.TestCase):
    def _segmenter(self):
        return Segmenter(
            threshold=300,
            start_frames=3,
            silence_ms=200,
            min_speech_ms=200,
            max_speech_ms=5000,
            preroll_ms=40,
            tail_ms=40,
        )

    def test_silence_emits_nothing(self):
        seg = self._segmenter()
        self.assertEqual(seg.push(pcm_const(16000, 0)), [])
        self.assertEqual(seg.flush(), [])

    def test_short_blip_is_dropped(self):
        seg = self._segmenter()
        blip = pcm_const(int(16000 * 0.04), 8000)
        self.assertEqual(seg.push(blip + pcm_const(16000, 0)), [])

    def test_phrase_emits_one_segment(self):
        seg = self._segmenter()
        phrase = pcm_const(int(16000 * 0.4), 8000)
        silence = pcm_const(int(16000 * 0.4), 0)
        parts = seg.push(phrase + silence)
        self.assertEqual(len(parts), 1)
        duration = len(parts[0]) / 2 / 16000
        self.assertGreater(duration, 0.2)
        self.assertLess(duration, 1.2)


class SttTests(unittest.TestCase):
    def test_multipart_has_audio_and_model_not_secret(self):
        body, boundary = build_transcription_body(
            wav=b"RIFF-secret-not-here",
            model="whisper-1",
            language="zh",
            prompt="项目",
            task="transcribe",
            include_optional=True,
        )
        self.assertIn(b"whisper-1", body)
        self.assertIn(b"RIFF-secret-not-here", body)
        self.assertIn(b'name="file"', body)
        self.assertIn(boundary.encode("ascii"), body)
        self.assertNotIn(b"sk-supersecretvalue", body)
        self.assertNotIn(b"Authorization", body)

    def test_redact_hides_bearer_and_sk(self):
        hidden = redact("Authorization: Bearer sk-supersecretvalue")
        self.assertNotIn("sk-supersecretvalue", hidden)
        self.assertIn("<redacted>", hidden)

    def test_parse_transcription_and_chat(self):
        self.assertEqual(parse_transcription_response('{"text":" 你好 "}'.encode()), "你好")
        payload = '{"choices":[{"message":{"content":"你好世界"}}]}'.encode()
        self.assertEqual(parse_transcription_response(payload), "你好世界")

    def test_key_requirement(self):
        self.assertFalse(needs_api_key("local", "https://api.openai.com/v1"))
        self.assertFalse(needs_api_key("transformers", "https://api.openai.com/v1"))
        self.assertFalse(needs_api_key("openai", "http://127.0.0.1:8000/v1"))
        self.assertTrue(needs_api_key("openai", "https://api.openai.com/v1"))
        self.assertEqual(resolve_transformers_model("turbo"), "openai/whisper-large-v3-turbo")
        self.assertEqual(drop_hallucination("谢谢观看"), "")
        self.assertEqual(drop_hallucination("换峰下来"), "换峰下来")
        self.assertEqual(resolve_transformers_model("large-v3"), "openai/whisper-large-v3")
        self.assertEqual(resolve_transformers_model("openai/whisper-large-v3"), "openai/whisper-large-v3")
        text = missing_credential_text("https://api.openai.com/v1")
        self.assertIn("STT_API_KEY", text)
        self.assertNotIn("sk-", text)


class DeviceTests(unittest.TestCase):
    def test_blackhole_only_when_routed(self):
        routed = [
            {"name": "BlackHole 2ch", "uid": "BH"},
            {"name": "扬声器", "uid": "SPK"},
            {
                "name": "多输出设备",
                "uid": "MULTI",
                "is_default_system_output": True,
                "is_aggregate": True,
                "subdevice_uids": ["BH", "SPK"],
            },
            {"name": "麦克风", "uid": "MIC", "is_default_input": True},
        ]
        self.assertEqual(blackhole_routed_name(routed), "BlackHole 2ch")
        isolated = [
            {"name": "BlackHole 2ch", "uid": "BH"},
            {"name": "扬声器", "uid": "SPK", "is_default_system_output": True, "subdevice_uids": []},
        ]
        self.assertIsNone(blackhole_routed_name(isolated))
        advice = "\n".join(routing_advice(isolated))
        self.assertIn("不会在系统采集失败时自动改用麦克风", advice)
        self.assertIn("不在当前系统输出里", advice)

    def test_header_and_env_file(self):
        fields = parse_header(b"SYSAUDIO1\nrate=48000\nchannels=2\nencoding=s16le\ndevice=BlackHole 2ch")
        self.assertEqual(fields["rate"], "48000")
        self.assertEqual(fields["channels"], "2")
        values = parse_env_assignments("# c\nexport STT_API_KEY='secret-value'\nSTT_MODEL=whisper-1\n")
        self.assertEqual(values["STT_API_KEY"], "secret-value")
        self.assertEqual(values["STT_MODEL"], "whisper-1")


if __name__ == "__main__":
    unittest.main()
