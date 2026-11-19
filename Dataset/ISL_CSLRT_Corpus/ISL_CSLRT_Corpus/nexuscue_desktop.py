import threading
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import cv2
import requests
import speech_recognition as sr
from PIL import Image, ImageTk

API_BASE = "http://127.0.0.1:8000"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "desktop_generated_videos"
OUTPUT_DIR.mkdir(exist_ok=True)

# NexusCue visual theme: deep navy, blue, cyan and emerald.
BG = "#0B1220"
HEADER_BG = "#0F1B2D"
PANEL = "#15243A"
PANEL_ALT = "#1C304B"
PREVIEW_BG = "#070D18"
ACCENT = "#38BDF8"
ACCENT_DARK = "#082F49"
GREEN = "#34D399"
GREEN_DARK = "#064E3B"
PURPLE = "#A78BFA"
TEXT = "#F1F5F9"
MUTED = "#A7B6CA"
BORDER = "#29415F"
ORANGE = "#FBBF24"
RED = "#FB7185"


class NexusCueDesktop:
    def __init__(self, root):
        self.root = root
        root.title("NexusCue | Speech to Indian Sign Language")
        # A larger starting window gives the sign preview substantially more room.
        root.geometry("1560x960")
        root.minsize(1120, 700)
        root.configure(bg=BG)

        # Scrollable application surface: the preview can remain large even
        # when the monitor/window is shorter than the complete layout.
        shell = tk.Frame(root, bg=BG)
        shell.pack(fill="both", expand=True)

        self.app_canvas = tk.Canvas(shell, bg=BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(
            shell, orient="vertical", command=self.app_canvas.yview,
            width=14, relief="flat", bd=0, bg=PANEL,
            troughcolor=BG, activebackground=ACCENT
        )
        scrollbar.pack(side="right", fill="y")
        self.app_canvas.pack(side="left", fill="both", expand=True)
        self.app_canvas.configure(yscrollcommand=scrollbar.set)

        content = tk.Frame(self.app_canvas, bg=BG)
        self.canvas_window = self.app_canvas.create_window(
            (0, 0), window=content, anchor="nw"
        )
        content.bind(
            "<Configure>",
            lambda _e: self.app_canvas.configure(
                scrollregion=self.app_canvas.bbox("all")
            )
        )
        self.app_canvas.bind(
            "<Configure>",
            lambda e: self.app_canvas.itemconfigure(
                self.canvas_window, width=e.width
            )
        )

        def wheel_scroll(event):
            self.app_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units"
            )
        self.app_canvas.bind_all("<MouseWheel>", wheel_scroll)
        self.app_canvas.bind_all(
            "<Button-4>", lambda _e: self.app_canvas.yview_scroll(-3, "units")
        )
        self.app_canvas.bind_all(
            "<Button-5>", lambda _e: self.app_canvas.yview_scroll(3, "units")
        )

        # All visible UI is placed in this scrollable content frame.
        root = content

        self.recognizer = sr.Recognizer()
        self.listening = False
        self.generating = False
        self.video = None
        self.video_job = None
        self.photo = None
        self.current_video = None

        # Header / branding
        header = tk.Frame(root, bg=HEADER_BG, highlightbackground=BORDER,
                          highlightthickness=1)
        header.pack(fill="x", padx=24, pady=(20, 14))

        logo = tk.Frame(header, bg=ACCENT, padx=14, pady=10)
        logo.pack(side="left", padx=(16, 12), pady=14)
        tk.Label(logo, text="NQ", bg=ACCENT, fg=ACCENT_DARK,
                 font=("Segoe UI", 16, "bold")).pack()

        brand = tk.Frame(header, bg=HEADER_BG)
        brand.pack(side="left", pady=12)
        tk.Label(brand, text="NexusCue", bg=HEADER_BG, fg=TEXT,
                 font=("Segoe UI", 25, "bold")).pack(anchor="w")
        tk.Label(brand, text="SPEECH TO INDIAN SIGN LANGUAGE",
                 bg=HEADER_BG, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(1, 0))

        self.backend_badge = tk.Label(
            header, text="● CHECKING BACKEND", bg=PANEL_ALT, fg=ORANGE,
            padx=14, pady=10, font=("Segoe UI", 9, "bold")
        )
        self.backend_badge.pack(side="right", padx=18)

        tk.Label(
            root,
            text="Speak naturally. NexusCue recognizes your words and generates a visual sign representation.",
            bg=BG, fg=MUTED, font=("Segoe UI", 10)
        ).pack(anchor="w", padx=28, pady=(0, 14))

        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=24, pady=(0, 14))

        # Left input panel
        left = tk.Frame(main, bg=PANEL, padx=20, pady=20, width=350,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)

        tk.Label(left, text="01  /  VOICE INPUT", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(left, text="Speak or type a sentence", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(12, 6))
        tk.Label(left, text="Use your microphone or enter a sentence manually.",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                 wraplength=300, justify="left").pack(anchor="w", pady=(0, 18))

        self.speak_btn = tk.Button(
            left, text="🎙  Start Speaking", command=self.start_listening,
            bg=ACCENT, fg=ACCENT_DARK, activebackground="#7DD3FC",
            activeforeground=ACCENT_DARK, relief="flat", bd=0,
            padx=12, pady=13, font=("Segoe UI", 11, "bold"), cursor="hand2"
        )
        self.speak_btn.pack(fill="x")

        tk.Label(left, text="RECOGNIZED SENTENCE", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(24, 9))

        transcript_wrap = tk.Frame(left, bg=PANEL_ALT,
                                   highlightbackground=BORDER, highlightthickness=1)
        transcript_wrap.pack(fill="x")
        self.transcript = tk.Text(
            transcript_wrap, height=8, wrap="word", bg=PANEL_ALT, fg=TEXT,
            insertbackground=ACCENT, selectbackground="#2563EB",
            relief="flat", bd=0, padx=12, pady=12,
            font=("Segoe UI", 11), highlightthickness=0
        )
        self.transcript.pack(fill="both", expand=True)

        tk.Label(
            left, text="You can edit the recognized text before generation.",
            bg=PANEL, fg=MUTED, font=("Segoe UI", 8),
            wraplength=300, justify="left"
        ).pack(anchor="w", pady=(8, 0))

        self.generate_btn = tk.Button(
            left, text="✨  Generate Sign Video", command=self.generate_video,
            bg=GREEN, fg=GREEN_DARK, activebackground="#6EE7B7",
            activeforeground=GREEN_DARK, relief="flat", bd=0,
            padx=12, pady=13, font=("Segoe UI", 11, "bold"), cursor="hand2"
        )
        self.generate_btn.pack(fill="x", pady=(22, 0))

        self.mode = tk.StringVar(value="MODE: WAITING")
        self.mode_badge = tk.Label(
            left, textvariable=self.mode, bg=PANEL_ALT, fg=MUTED,
            padx=12, pady=12, font=("Segoe UI", 9, "bold"), anchor="w",
            highlightbackground=BORDER, highlightthickness=1
        )
        self.mode_badge.pack(fill="x", pady=(18, 0))

        # Right preview panel gets the majority of the window width and height.
        right = tk.Frame(main, bg=PANEL, padx=18, pady=18,
                         highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True)

        top = tk.Frame(right, bg=PANEL)
        top.pack(fill="x", pady=(0, 12))
        title_group = tk.Frame(top, bg=PANEL)
        title_group.pack(side="left")
        tk.Label(title_group, text="02  /  SIGN VIDEO PREVIEW",
                 bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(title_group, text="Review the generated sign sequence",
                 bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(5, 0))

        self.time_label = tk.Label(top, text="Waiting for video",
                                   bg=PANEL_ALT, fg=MUTED,
                                   padx=12, pady=8, font=("Segoe UI", 9, "bold"))
        self.time_label.pack(side="right", anchor="n")

        # Larger preview canvas. Video is scaled to fill this area while
        # preserving its original aspect ratio (no cropping or stretching).
        self.video_area = tk.Frame(
            right, bg=PREVIEW_BG, height=760,
            highlightbackground="#4B6F96", highlightthickness=2
        )
        # Keep a genuinely tall viewing surface; scrolling reveals its lower
        # controls on smaller displays instead of shrinking the signer image.
        self.video_area.pack(fill="both", expand=True, pady=(0, 14))
        self.video_area.pack_propagate(False)
        self.video_label = tk.Label(
            self.video_area,
            text="Your generated sign video will appear here.\n\n"
                 "Generate a video to view the sign sequence at a larger size.",
            bg=PREVIEW_BG, fg=MUTED, font=("Segoe UI", 13),
            justify="center"
        )
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        controls = tk.Frame(right, bg=PANEL)
        controls.pack(fill="x")
        self.play_btn = tk.Button(
            controls, text="▶  Play", command=self.play_video,
            state="disabled", bg=ACCENT, fg=ACCENT_DARK,
            activebackground="#7DD3FC", relief="flat", bd=0,
            padx=22, pady=10, font=("Segoe UI", 10, "bold"), cursor="hand2"
        )
        self.play_btn.pack(side="left")
        self.stop_btn = tk.Button(
            controls, text="■  Stop", command=self.stop_video,
            state="disabled", bg=PANEL_ALT, fg=TEXT,
            activebackground="#29415F", relief="flat", bd=0,
            padx=22, pady=10, font=("Segoe UI", 10, "bold"), cursor="hand2"
        )
        self.stop_btn.pack(side="left", padx=10)
        tk.Label(controls, text="Original video proportions preserved",
                 bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="right", padx=(8, 2))

        footer = tk.Frame(root, bg=BG)
        footer.pack(fill="x", padx=28, pady=(0, 16))
        self.status = tk.StringVar(value="Checking backend...")
        tk.Label(footer, textvariable=self.status, bg=BG, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w").pack(fill="x")
        tk.Label(
            footer, text="Tip: Scroll vertically to view the complete preview and playback controls.",
            bg=BG, fg="#7186A3", font=("Segoe UI", 8), anchor="w"
        ).pack(fill="x", pady=(5, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(250, self.check_backend)

    def set_status(self, text):
        self.status.set(text)

    def check_backend(self):
        def work():
            try:
                r = requests.get(API_BASE + "/health", timeout=5)
                r.raise_for_status()
                ok = r.json().get("renderer_exists", False)
                badge = ("● BACKEND CONNECTED", GREEN) if ok else ("● RENDERER MISSING", RED)
                msg = "Backend connected. Ready for speech input." if ok else "API is running, but renderer file is missing."
            except Exception:
                badge, msg = ("● BACKEND OFFLINE", RED), "Start the NexusCue FastAPI server on port 8000."
            self.root.after(0, lambda: self.backend_badge.config(text=badge[0], fg=badge[1]))
            self.root.after(0, lambda: self.set_status(msg))
        threading.Thread(target=work, daemon=True).start()

    def start_listening(self):
        if self.listening or self.generating:
            return
        self.listening = True
        self.speak_btn.config(state="disabled", text="Listening...")
        self.set_status("Listening. Speak after the microphone opens.")
        threading.Thread(target=self.listen_worker, daemon=True).start()

    def listen_worker(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = self.recognizer.listen(source, timeout=6, phrase_time_limit=12)
            self.root.after(0, lambda: self.set_status("Converting speech to text..."))
            recognized = self.recognizer.recognize_google(audio, language="en-IN")
            self.root.after(0, lambda: self.set_transcript(recognized))
            self.root.after(0, lambda: self.set_status("Speech recognized. Review it and generate the sign video."))
        except sr.WaitTimeoutError:
            self.root.after(0, lambda: self.set_status("No speech detected. Click Start Speaking and try again."))
        except sr.UnknownValueError:
            self.root.after(0, lambda: self.set_status("Speech was unclear. Please try again."))
        except sr.RequestError as exc:
            msg = f"Speech recognition service error: {exc}"
            self.root.after(0, lambda: self.set_status(msg))
        except OSError as exc:
            msg = f"Microphone unavailable. Check microphone access and PyAudio: {exc}"
            self.root.after(0, lambda: self.set_status(msg))
        except Exception as exc:
            msg = f"Speech input error: {exc}"
            self.root.after(0, lambda: self.set_status(msg))
        finally:
            self.listening = False
            self.root.after(0, lambda: self.speak_btn.config(state="normal", text="🎙  Start Speaking"))

    def set_transcript(self, text):
        self.transcript.delete("1.0", "end")
        self.transcript.insert("1.0", text)

    def generate_video(self):
        if self.generating or self.listening:
            return
        sentence = self.transcript.get("1.0", "end").strip()
        if not sentence:
            messagebox.showwarning("NexusCue", "Speak or type a sentence first.")
            return
        self.generating = True
        self.generate_btn.config(state="disabled", text="Generating...")
        self.speak_btn.config(state="disabled")
        self.set_status("Sending sentence to backend and generating sign video...")
        threading.Thread(target=self.generate_worker, args=(sentence,), daemon=True).start()

    def generate_worker(self, sentence):
        try:
            response = requests.post(
                API_BASE + "/api/generate",
                json={"sentence": sentence},
                headers={"Accept": "video/mp4"},
                timeout=360
            )
            if response.status_code != 200:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(f"Backend returned {response.status_code}: {detail}")
            if not response.content:
                raise RuntimeError("Backend returned an empty video.")
            path = OUTPUT_DIR / f"nexcue_desktop_{int(time.time())}.mp4"
            path.write_bytes(response.content)
            mode = response.headers.get("X-NexusCue-Mode", "unknown")
            size = len(response.content)
            self.root.after(0, lambda: self.generation_done(path, mode, size))
        except Exception as exc:
            error = str(exc)
            self.root.after(0, lambda: self.generation_failed(error))

    def generation_done(self, path, mode, size):
        self.generating = False
        self.generate_btn.config(state="normal", text="✨  Generate Sign Video")
        self.speak_btn.config(state="normal")
        self.current_video = path
        if mode == "english_fingerspelling_fallback":
            self.mode.set("MODE: ENGLISH FINGERSPELLING FALLBACK")
            self.mode_badge.config(fg=ORANGE)
            self.set_status(f"Fallback video generated ({size:,} bytes). Not a verified ISL sentence translation.")
        elif mode == "dataset_sentence_or_gloss":
            self.mode.set("MODE: DATASET SENTENCE / GLOSS")
            self.mode_badge.config(fg=GREEN)
            self.set_status(f"Dataset-based sign video generated successfully ({size:,} bytes).")
        else:
            self.mode.set("MODE: " + mode.upper())
            self.set_status(f"Video generated ({size:,} bytes).")
        self.open_preview(path)

    def generation_failed(self, error):
        self.generating = False
        self.generate_btn.config(state="normal", text="✨  Generate Sign Video")
        self.speak_btn.config(state="normal")
        self.set_status("Video generation failed.")
        messagebox.showerror("NexusCue generation error", error)

    def open_preview(self, path):
        self.stop_video()
        self.video = cv2.VideoCapture(str(path))
        if not self.video.isOpened():
            self.video_label.config(text="Could not open generated MP4.", image="")
            return
        self.play_btn.config(state="normal")
        self.stop_btn.config(state="normal")
        self.video_label.config(text="Video ready. Press Play.", image="")
        self.time_label.config(text="Ready")

    def play_video(self):
        if not self.current_video:
            return
        self.stop_video()
        self.video = cv2.VideoCapture(str(self.current_video))
        if self.video.isOpened():
            self.play_frame()

    def play_frame(self):
        if self.video is None or not self.video.isOpened():
            return
        ok, frame = self.video.read()
        if not ok:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.time_label.config(text="Playback finished")
            return

        # Use the actual expanded preview dimensions so the full frame is
        # rendered as large as possible without distortion or cropping.
        self.root.update_idletasks()
        pw = max(self.video_area.winfo_width() - 36, 1)
        ph = max(self.video_area.winfo_height() - 36, 1)
        h, w = frame.shape[:2]
        scale = min(pw / w, ph / h)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.video_label.config(image=self.photo, text="")
        fps = self.video.get(cv2.CAP_PROP_FPS) or 10
        self.time_label.config(text=f"{self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000:.1f}s")
        self.video_job = self.root.after(max(15, int(1000 / fps)), self.play_frame)

    def stop_video(self):
        if self.video_job is not None:
            try:
                self.root.after_cancel(self.video_job)
            except Exception:
                pass
            self.video_job = None
        if self.video is not None:
            self.video.release()
            self.video = None

    def close(self):
        self.stop_video()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    NexusCueDesktop(root)
    root.mainloop()
