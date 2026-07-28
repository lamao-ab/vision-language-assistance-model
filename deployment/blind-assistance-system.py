"""
blind-assist-jetson-demo.py  — CORRECTED, on-device build
==========================================================
Changes vs. the original demo:
  * LOCAL speech-to-text (faster-whisper OR vosk) — no cloud, no network
    dependency, no recognize_google. Audio never leaves the device.
  * SAFE command matching: whole-word/phrase only (no more "don't turn off the
    stove" triggering a shutdown), with a negation guard and an explicit
    multi-word phrase required for power-off.
  * subprocess list-form TTS everywhere — the os.system fallback (shell
    injection on quotes/$/backticks in model output) is gone.
  * listen() distinguishes "heard nothing" from "couldn't understand" and gives
    the user audible feedback instead of silent failure.
  * Camera capture aligned with bench_latency.py (BUFFERSIZE=1, short flush,
    no arbitrary sleep) so the deployed app matches the benchmarked pipeline.
  * Camera auto-reconnect; clean serial loop with feedback at each step.

Install (pick ONE backend):
    faster-whisper:  pip install faster-whisper
    vosk:            pip install vosk   + download a small model, set VOSK_MODEL
Audio tools:         sudo apt install alsa-utils pulseaudio-utils espeak
"""
import cv2, torch, subprocess, os, time, io, wave, re, datetime
import numpy as np
import speech_recognition as sr
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID    = "lamao-ab/paligemma-blind-assist-qlora-merged-v1"
STT_BACKEND = "whisper"            # "whisper" or "vosk"
WHISPER_SIZE = "tiny.en"           # tiny.en is plenty for short commands
VOSK_MODEL  = "/opt/vosk/vosk-model-small-en-us-0.15"
WAKE_WORD   = None                 # e.g. "assistant" to require a prefix; None = off
TEMP_WAV    = "/tmp/assist_voice.wav"
SAVE_CAPTURES = True               # save every captured frame (off for deployment/privacy)
CAPTURE_DIR   = "captures"         # folder for saved frames

# ── Mute noisy ALSA logs ──────────────────────────────────────────────────
# The handler MUST stay referenced at module scope, or Python garbage-collects
# the ctypes callback and its C pointer dangles -> collides with another
# function and segfaults when ALSA fires an error (e.g. on mic open).
_ALSA_HANDLER_TYPE = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def _alsa_error_silence(filename, line, function, err, fmt):
    pass
_ALSA_HANDLER = _ALSA_HANDLER_TYPE(_alsa_error_silence)   # module-level: kept alive
try:
    cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_ALSA_HANDLER)
except Exception:
    pass


# ── Local STT backends ──────────────────────────────────────────────────────
def _wav_to_int16(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

class WhisperSTT:
    """faster-whisper on CPU (int8) so it never competes with the VLM for VRAM."""
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    def transcribe(self, audio):
        pcm = _wav_to_int16(audio.get_wav_data(convert_rate=16000, convert_width=2))
        samples = pcm.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(samples, language="en", beam_size=1)
        return " ".join(s.text for s in segments).strip().lower()

class VoskSTT:
    """Fully offline Kaldi recognizer; lightest install on aarch64."""
    def __init__(self):
        from vosk import Model, KaldiRecognizer
        self._Rec = KaldiRecognizer
        self.model = Model(VOSK_MODEL)
    def transcribe(self, audio):
        import json
        pcm = audio.get_wav_data(convert_rate=16000, convert_width=2)
        rec = self._Rec(self.model, 16000)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip().lower()


# ── Safe intent matching ──────────────────────────────────────────────────
_NEG = ("don't", "do not", "not ", "never", "isn't", "wouldn't")
def _said(text, *phrases):
    return any(re.search(r'\b' + re.escape(p) + r'\b', text) for p in phrases)
def _negated(text):
    return any(n in text for n in _NEG)


class BlindAssistWearable:
    def __init__(self):
        print("\n" + "="*50)
        print("BLIND ASSIST WEARABLE — INITIALIZING")
        print("="*50)
        self.init_audio()
        self.init_camera()
        self.init_model()     # load PaliGemma FIRST, on a clean CUDA context
        self.init_stt()       # then init local STT (CPU) — avoids NVML assert
        self.speak("System is ready.")

    def init_audio(self):
        print("[INFO] Audio...")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.5
        self.recognizer.dynamic_energy_threshold = True
        self.mic = sr.Microphone()
        with self.mic as src:
            self.recognizer.adjust_for_ambient_noise(src, duration=2)

    def init_stt(self):
        print(f"[INFO] Local STT backend: {STT_BACKEND}")
        self.stt = WhisperSTT() if STT_BACKEND == "whisper" else VoskSTT()

    def init_camera(self):
        print("[INFO] Camera...")
        self._open_camera()

    def _open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        for _ in range(30): self.cap.grab()
        time.sleep(0.5)

    def init_model(self):
        print("[INFO] Loading PaliGemma...")
        self.processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        print("[INFO] AI online.")

    # ── TTS: subprocess list form only (no shell, no injection) ────────────
    def speak(self, text):
        text = (text or "").strip() or "."
        print(f"\n[ASSISTANT]: {text}")
        try:
            subprocess.run(['espeak', '-s', '165', '-v', 'en+m3', '-w', TEMP_WAV, text],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['paplay', TEMP_WAV], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            try:
                subprocess.run(['espeak', text], check=False)   # list form, safe
            except Exception as e:
                print(f"[WARN] TTS failed: {e}")
        finally:
            if os.path.exists(TEMP_WAV):
                try: os.remove(TEMP_WAV)
                except OSError: pass

    # ── Camera: aligned with the benchmark, with one reconnect attempt ─────
    def capture_image(self):
        if not self.cap.isOpened():
            self._open_camera()
        for _ in range(5): self.cap.grab()          # flush stale frames (no sleep)
        ret, frame = self.cap.read()
        if not ret:                                  # try a single reconnect
            self.cap.release(); self._open_camera()
            for _ in range(5): self.cap.grab()
            ret, frame = self.cap.read()
        if not ret:
            return None
        if SAVE_CAPTURES:
            try:
                os.makedirs(CAPTURE_DIR, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(os.path.join(CAPTURE_DIR, f"view_{ts}.jpg"), frame)
            except Exception as e:
                print(f"[WARN] could not save capture: {e}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ── Listen: capture (VAD) then transcribe locally ─────────────────────
    def listen(self):
        with self.mic as src:
            print("\n[INFO] Listening...")
            try:
                audio = self.recognizer.listen(src, timeout=5, phrase_time_limit=12)
            except sr.WaitTimeoutError:
                return None                          # heard nothing — stay quiet
        try:
            text = self.stt.transcribe(audio)
        except Exception as e:
            print(f"[WARN] STT error: {e}")
            self.speak("Speech recognition error.")
            return None
        if not text:
            self.speak("Sorry, I didn't catch that.")
            return None
        print(f"[USER]: {text}")
        return text

    def run(self):
        while True:
            text = self.listen()
            if not text:
                continue

            if WAKE_WORD:
                if not _said(text, WAKE_WORD):
                    continue
                text = text.replace(WAKE_WORD, "", 1).strip()

            # explicit, multi-word control phrases + negation guard
            if not _negated(text) and _said(text, "power off the system",
                                            "shut down the system", "shutdown the system"):
                self.speak("Shutting down the system.")
                self.cleanup()
                os.system("sudo shutdown -h now")
                break
            if not _negated(text) and _said(text, "close the application",
                                            "stop the assistant", "exit application"):
                self.speak("Closing the application.")
                break

            self.speak("Processing.")
            image = self.capture_image()
            if image is None:
                self.speak("Camera error. Please try again.")
                continue

            if _said(text, "describe", "what is this", "looking at", "what do you see"):
                prompt, max_tokens = "<image>Describe this scene for a blind person.", 64
            else:
                prompt, max_tokens = f"<image>Assist a blind person: {text}", 30

            try:
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                response = self.processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            except Exception as e:
                print(f"[WARN] Inference error: {e}")
                self.speak("I had trouble processing that. Please try again.")
                continue
            self.speak(response)

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    app = BlindAssistWearable()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()