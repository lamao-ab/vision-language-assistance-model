#!/usr/bin/env python3
"""
Real-time Blind Assistance Demo on Jetson Orin Nano
Runs PaliGemma with QLoRA for live camera inference
"""
import cv2
import torch
import numpy as np
import speech_recognition as sr
import subprocess
import os
import time
from PIL import Image, ImageEnhance
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from ctypes import *

# === MUTE ALSA LOGS ===
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass
# ======================


class BlindAssistWearable:
    def __init__(self):
        print("\n" + "="*50)
        print("🚀 BLIND ASSIST WEARABLE SYSTEM INITIALIZING")
        print("="*50)

        self.temp_wav = "/tmp/assist_voice.wav"
        self.init_audio()
        self.init_camera()
        self.init_model()

        self.speak("System is ready.")

    # ------------------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------------------

    def init_audio(self):
        print("[INFO] Initializing Audio...")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.5
        self.recognizer.dynamic_energy_threshold = True
        self.mic = sr.Microphone()
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)

    # ------------------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------------------

    def init_camera(self):
        print("[INFO] Initializing Camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # Warm up: let auto-exposure converge before first capture
        print("[INFO] Warming up camera...")
        for _ in range(30):
            self.cap.grab()
        time.sleep(1.5)
        print("[INFO] Camera ready.")

    def enhance_frame(self, pil_image):
        """
        Gentle enhancement using PIL only — no aggressive CV2 filters.
        Keeps the image natural so the model does not flag it as corrupted.
        """
        pil_image = ImageEnhance.Contrast(pil_image).enhance(1.2)
        pil_image = ImageEnhance.Sharpness(pil_image).enhance(1.3)
        pil_image = ImageEnhance.Brightness(pil_image).enhance(1.05)
        return pil_image

    def capture_image(self):
        # Flush stale frames so we get a fresh exposure
        for _ in range(20):
            self.cap.grab()
        time.sleep(0.8)

        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            return None

        # Retry once if severely under/overexposed
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if brightness < 30 or brightness > 230:
            print(f"[WARN] Poor brightness ({brightness:.1f}), retrying...")
            time.sleep(1.5)
            for _ in range(15):
                self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                return None

        # Convert to PIL RGB — let the processor handle resizing.
        # Do NOT manually resize; the processor knows the correct target size.
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply gentle PIL-based enhancement
        pil_image = self.enhance_frame(pil_image)

        # Save captured image to home directory for inspection
        save_path = os.path.expanduser("~/last_view.jpg")
        pil_image.save(save_path)
        print(f"[INFO] Image saved to {save_path}")

        return pil_image

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------

    def init_model(self):
        print("[INFO] Loading PaliGemma AI...")
        model_id = "lamao-ab/paligemma-blind-assist-jetson-ready"
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map={"": "cuda:0"},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print("[INFO] AI is online.")

    # ------------------------------------------------------------------
    # SPEECH
    # ------------------------------------------------------------------

    def speak(self, text):
        print(f"\n🗣️  [ASSISTANT]: {text}")
        try:
            subprocess.run(
                ['espeak', '-s', '165', '-v', 'en+m3', '-w', self.temp_wav, text],
                check=True
            )
            subprocess.run(['paplay', self.temp_wav], check=True)
            if os.path.exists(self.temp_wav):
                os.remove(self.temp_wav)
        except Exception:
            os.system(f'espeak "{text}"')

    def listen(self):
        with self.mic as source:
            print("\n🎙️  [INFO] Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=12)
                text = self.recognizer.recognize_google(audio)
                print(f"👤 [USER]: {text}")
                return text.lower()
            except Exception:
                return None

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        while True:
            prompt_text = self.listen()
            if not prompt_text:
                continue

            if any(x in prompt_text for x in ["power off", "shutdown system", "turn off"]):
                self.speak("Shutting down the system.")
                self.cleanup()
                os.system("sudo shutdown -h now")
                break

            if any(x in prompt_text for x in ["exit", "stop", "close"]):
                self.speak("Closing the application.")
                break

            self.speak("Processing...")
            image = self.capture_image()

            if image is None:
                self.speak("Camera error.")
                continue

            if any(w in prompt_text for w in ["describe", "what is this", "looking at"]):
                full_prompt = "<image>Describe this scene for a blind person."
                max_tokens = 64
            else:
                full_prompt = f"<image>Assist a blind person: {prompt_text}"
                max_tokens = 30

            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt"
            ).to("cuda")

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )

            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            self.speak(response)

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    app = BlindAssistWearable()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()
