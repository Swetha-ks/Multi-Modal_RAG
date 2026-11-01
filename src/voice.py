"""
voice.py
────────
VOICE MODULE — Google Assistant Style
────────────────────────────────────────
Voice Input:  Microphone → Google Speech Recognition → Text
Voice Output: Text → gTTS (Google Text-to-Speech) → Natural voice
"""

import os
import threading
import tempfile


class VoiceOutput:
    """Converts text to speech using Google Text-to-Speech (gTTS)."""

    def __init__(self, language="en-IN"):
        self.language = language
        self.is_speaking = False
        try:
            from gtts import gTTS
            import pygame
            self.gTTS = gTTS
            pygame.mixer.init()
            self.pygame = pygame
            self.available = True
            print("✅ Google-style voice output ready!")
        except ImportError:
            print("⚠️ gTTS or pygame not installed. Voice output disabled.")
            self.available = False
        except Exception as e:
            print(f"⚠️ Voice output error: {e}")
            self.available = False

    def speak(self, text: str):
        if not self.available:
            return
        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()

    def _speak_thread(self, text: str):
        try:
            self.is_speaking = True
            clean = (text
                     .replace("**", "").replace("*", "")
                     .replace("#", "").replace("`", "")
                     .replace("₹", "rupees ")
                     .replace("Rs.", "rupees ")
                     .replace("\n", ". "))
            if len(clean) > 300:
                clean = clean[:300] + "..."

            tts = self.gTTS(text=clean, lang=self.language, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
                tts.save(tmp_path)

            self.pygame.mixer.music.load(tmp_path)
            self.pygame.mixer.music.play()
            while self.pygame.mixer.music.get_busy():
                self.pygame.time.Clock().tick(10)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.is_speaking = False

    def stop(self):
        if self.available:
            try:
                self.pygame.mixer.music.stop()
                self.is_speaking = False
            except:
                pass


class VoiceInput:
    """Converts microphone speech to text using Google Speech Recognition."""

    def __init__(self):
        try:
            import speech_recognition as sr
            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.available = True
            print("✅ Voice input ready!")
        except ImportError:
            print("⚠️ SpeechRecognition not installed.")
            self.available = False

    def listen(self, timeout: int = 5) -> dict:
        if not self.available:
            return {"success": False, "error": "Voice input not available"}
        try:
            with self.sr.Microphone() as source:
                print("🎤 Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

            text = self.recognizer.recognize_google(audio, language="en-IN")
            print(f"Heard: {text}")
            return {"success": True, "text": text}

        except self.sr.UnknownValueError:
            return {"success": False, "error": "Could not understand. Please speak clearly."}
        except self.sr.RequestError:
            return {"success": False, "error": "Speech service unavailable. Check internet."}
        except self.sr.WaitTimeoutError:
            return {"success": False, "error": "No speech detected. Please try again."}
        except OSError:
            return {"success": False, "error": "Microphone not found."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_microphone(self) -> bool:
        if not self.available:
            return False
        try:
            with self.sr.Microphone():
                return True
        except OSError:
            return False
