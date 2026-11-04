"""Desktop app for drawing a digit and classifying it with one of the trained models.

Draw with the mouse, choose a model and press Predict. The drawing is converted to a
28x28 image prepared the same way as MNIST digits (cropped, scaled to fit a 20x20 box,
centered by center of mass) before it goes to the model.

Run from the repository root::

    python gui/gui.py
"""

import os
import tkinter as tk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# high DPI on Windows to prevent blurry text
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# repo root, used to find the saved models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# colors
BG_DARK = "#121214"
CARD_BG = "#1a1a1e"
BORDER_COLOR = "#2a2a30"
SURFACE = "#25252b"
ACCENT = "#10b981"
ACCENT_HOVER = "#059669"
TEXT_LIGHT = "#f4f4f5"
TEXT_DIM = "#a1a1aa"
TEXT_MUTED = "#71717a"
CANVAS_BG = "#09090b"

# models shown in the app: display name -> file path, type ("custom" or "keras") and a short label
MODELS: dict[str, dict[str, str]] = {
    "Custom CNN (NumPy)": {
        "path": os.path.join(BASE_DIR, "models", "custom_cnn.pkl"),
        "type": "custom",
        "desc": "Scratch NumPy"
    },
    "Keras CNN": {
        "path": os.path.join(BASE_DIR, "models", "digit-recognizer-without-augmentation.keras"),
        "type": "keras",
        "desc": "Baseline"
    },
    "Keras CNN (augmented)": {
        "path": os.path.join(BASE_DIR, "models", "digit-recognizer-with-augmentation.keras"),
        "type": "keras",
        "desc": "Augmented"
    }
}


class DigitRecognizerApp:
    """Main window: drawing canvas, model choice, prediction and class probabilities.

    The drawing is kept twice: on the Tkinter canvas (what the user sees) and in a
    grayscale PIL image of the same size (what goes to the model).

    Args:
        root: Main Tkinter window.

    Attributes:
        model: Loaded model (``my_cnn.Model`` or a Keras model), ``None`` if loading failed.
        model_type: ``"custom"`` or ``"keras"``.
        tool: Current tool, ``"brush"`` or ``"eraser"``.
        image: Grayscale drawing, white strokes on black, ``canvas_size x canvas_size``.
        undo_stack: Copies of ``image`` from before each stroke or clear.
    """
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Digit Recognition")
        self.root.configure(bg=BG_DARK)
        self.root.state("zoomed")
        self.root.minsize(920, 620)

        # my_cnn.Model or a Keras model, so the type is left open
        self.model: Any = None
        self.model_type: str | None = None

        self.canvas_size: int = 400
        self.brush_size: int = 18
        self.tool: str = "brush"
        self.last_x: int | None = None
        self.last_y: int | None = None
        self.undo_stack: list[Image.Image] = []

        self.image: Image.Image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_engine: ImageDraw.ImageDraw = ImageDraw.Draw(self.image)
        self.canvas_photo: ImageTk.PhotoImage | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._bind_shortcuts()
        self._load_model(list(MODELS.keys())[0])

    def _build_ui(self) -> None:
        """Create all widgets: header, canvas with tools, model choice, prediction and probability bars."""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # header
        header = tk.Frame(self.root, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=16, pady=10)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 8))

        tk.Label(header, text="Digit Recognition", font=("Segoe UI", 16, "bold"), bg=CARD_BG, fg=TEXT_LIGHT).pack(side="left")
        self.status_label = tk.Label(header, text="", font=("Segoe UI", 10), bg=CARD_BG, fg=TEXT_DIM)
        self.status_label.pack(side="right")

        # main container
        body = tk.Frame(self.root, bg=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # left panel (canvas & tools)
        left = tk.Frame(body, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=16, pady=14)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg=CANVAS_BG, cursor="crosshair", highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._draw_start)
        self.canvas.bind("<B1-Motion>", self._draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._draw_end)

        # tools row
        tools = tk.Frame(left, bg=CARD_BG)
        tools.pack(fill="x", pady=(10, 12))

        self.btn_brush = tk.Button(tools, text="Brush", font=("Segoe UI", 9), bg=ACCENT, fg="#ffffff", relief="flat", bd=0, padx=12, pady=5, cursor="hand2", command=lambda: self._set_tool("brush"))
        self.btn_brush.pack(side="left", padx=(0, 5))

        self.btn_eraser = tk.Button(tools, text="Eraser", font=("Segoe UI", 9), bg=SURFACE, fg=TEXT_DIM, relief="flat", bd=0, padx=12, pady=5, cursor="hand2", command=lambda: self._set_tool("eraser"))
        self.btn_eraser.pack(side="left", padx=(0, 5))

        tk.Button(tools, text="Undo", font=("Segoe UI", 9), bg=SURFACE, fg=TEXT_DIM, relief="flat", bd=0, padx=12, pady=5, cursor="hand2", command=self._undo).pack(side="left", padx=(0, 5))
        tk.Button(tools, text="Clear", font=("Segoe UI", 9), bg=SURFACE, fg=TEXT_DIM, relief="flat", bd=0, padx=12, pady=5, cursor="hand2", command=self._clear).pack(side="left")

        tk.Button(left, text="Predict", font=("Segoe UI", 11, "bold"), bg=ACCENT, fg="#ffffff", activebackground=ACCENT_HOVER, activeforeground="#ffffff", relief="flat", bd=0, pady=8, cursor="hand2", command=self._predict).pack(fill="x")

        # right panel (model, prediction, probabilities)
        right = tk.Frame(body, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        # model card
        m_card = tk.Frame(right, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=16, pady=12)
        m_card.pack(fill="x", pady=(0, 10))
        tk.Label(m_card, text="Model", font=("Segoe UI", 10, "bold"), bg=CARD_BG, fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))

        self.model_var = tk.StringVar(value=list(MODELS.keys())[0])
        for name, info in MODELS.items():
            row = tk.Frame(m_card, bg=CARD_BG)
            row.pack(fill="x", pady=2)
            tk.Radiobutton(row, text=name, variable=self.model_var, value=name, font=("Segoe UI", 10), bg=CARD_BG, fg=TEXT_LIGHT, selectcolor=SURFACE, activebackground=CARD_BG, activeforeground=TEXT_LIGHT, command=lambda: self._load_model(self.model_var.get())).pack(side="left")
            tk.Label(row, text=f"({info['desc']})", font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_MUTED).pack(side="left", padx=(8, 0))

        # prediction card
        r_card = tk.Frame(right, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=16, pady=12)
        r_card.pack(fill="x", pady=(0, 10))
        tk.Label(r_card, text="Prediction", font=("Segoe UI", 10, "bold"), bg=CARD_BG, fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))

        hero = tk.Frame(r_card, bg=CARD_BG)
        hero.pack(fill="x")

        digit_box = tk.Frame(hero, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER_COLOR, width=90, height=90)
        digit_box.pack(side="left", padx=(0, 16))
        digit_box.pack_propagate(False)
        self.digit_label = tk.Label(digit_box, text="-", font=("Segoe UI", 44, "bold"), bg=SURFACE, fg=TEXT_LIGHT)
        self.digit_label.pack(expand=True)

        self.conf_label = tk.Label(hero, text="Draw a digit and click Predict", font=("Segoe UI", 11), bg=CARD_BG, fg=TEXT_DIM, anchor="w")
        self.conf_label.pack(side="left", fill="y")

        p_box = tk.Frame(hero, bg=CARD_BG)
        p_box.pack(side="right")
        tk.Label(p_box, text="Input (28x28)", font=("Segoe UI", 8), bg=CARD_BG, fg=TEXT_MUTED).pack(pady=(0, 2))
        self.preview_canvas = tk.Canvas(p_box, width=84, height=84, bg=CANVAS_BG, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.preview_canvas.pack()

        # probabilities card
        p_card = tk.Frame(right, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER_COLOR, padx=16, pady=12)
        p_card.pack(fill="both", expand=True)
        tk.Label(p_card, text="Probabilities", font=("Segoe UI", 10, "bold"), bg=CARD_BG, fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))

        self.prob_canvas = tk.Canvas(p_card, bg=CARD_BG, highlightthickness=0, height=255)
        self.prob_canvas.pack(fill="both", expand=True)
        self._update_prob_bars(None, -1)

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts.

        Ctrl+Z: undo, Delete / C: clear, B: brush, E: eraser, Enter: predict.
        """
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-Z>", lambda e: self._undo())
        self.root.bind("<Delete>", lambda e: self._clear())
        self.root.bind("<c>", lambda e: self._clear() if e.widget == self.root or isinstance(e.widget, tk.Canvas) else None)
        self.root.bind("<b>", lambda e: self._set_tool("brush"))
        self.root.bind("<e>", lambda e: self._set_tool("eraser"))
        self.root.bind("<Return>", lambda e: self._predict())

    def _load_model(self, name: str) -> None:
        """Load a model from ``MODELS`` and show the result in the header.

        TensorFlow is imported only when a Keras model is chosen, so the custom
        NumPy model works without it. If loading fails, ``model`` stays ``None``
        and the error is printed to the console.

        Args:
            name: Key in ``MODELS``.
        """
        info = MODELS[name]
        self.model = None
        self.model_type = info["type"]
        self.status_label.config(text=f"loading: {name}", fg="#eab308")
        self.root.update()

        try:
            if info["type"] == "keras":
                import tensorflow as tf
                self.model = tf.keras.models.load_model(info["path"])
            else:
                import my_cnn
                self.model = my_cnn.Model.load(info["path"])

            self.status_label.config(text=f"loaded: {name}", fg=ACCENT)
        except Exception as e:
            self.status_label.config(text=f"error: {name}", fg="#ef4444")
            print(f"Failed to load model {name}: {e}")

    def _set_tool(self, tool: str) -> None:
        """Switch between brush and eraser and highlight the active button.

        Args:
            tool: ``"brush"`` or ``"eraser"``.
        """
        self.tool = tool
        if tool == "brush":
            self.btn_brush.config(bg=ACCENT, fg="#ffffff")
            self.btn_eraser.config(bg=SURFACE, fg=TEXT_DIM)
        else:
            self.btn_brush.config(bg=SURFACE, fg=TEXT_DIM)
            self.btn_eraser.config(bg=ACCENT, fg="#ffffff")


    def _draw_start(self, event: tk.Event) -> None:
        """Start a stroke: save the current drawing for undo and draw a dot.

        The undo history keeps at most 20 steps.

        Args:
            event: Mouse button press on the canvas.
        """
        if len(self.undo_stack) >= 20:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.image.copy())

        self.last_x, self.last_y = event.x, event.y
        color_val = 255 if self.tool == "brush" else 0
        color_hex = "#ffffff" if self.tool == "brush" else CANVAS_BG
        r = self.brush_size

        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill=color_hex, outline=color_hex)
        self.draw_engine.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=color_val)

    def _draw_move(self, event: tk.Event) -> None:
        """Continue the stroke with a line from the last mouse position.

        Draws on both the visible canvas and the PIL image.

        Args:
            event: Mouse movement with the button held down.
        """
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = event.x, event.y
            return

        color_val = 255 if self.tool == "brush" else 0
        color_hex = "#ffffff" if self.tool == "brush" else CANVAS_BG
        r = self.brush_size

        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill=color_hex, width=r * 2, capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self.draw_engine.line([(self.last_x, self.last_y), (event.x, event.y)], fill=color_val, width=r * 2)
        self.draw_engine.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=color_val)

        self.last_x, self.last_y = event.x, event.y

    def _draw_end(self, event: tk.Event) -> None:
        """End the stroke.

        Args:
            event: Mouse button release.
        """
        self.last_x = None
        self.last_y = None

    def _undo(self) -> None:
        """Restore the drawing from before the last stroke or clear."""
        if not self.undo_stack:
            return
        self.image = self.undo_stack.pop()
        self.draw_engine = ImageDraw.Draw(self.image)
        self.canvas_photo = ImageTk.PhotoImage(self.image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")

    def _clear(self) -> None:
        """Erase the whole drawing and the prediction. Can be undone."""
        if self.image.getbbox() is not None:
            self.undo_stack.append(self.image.copy())
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_engine = ImageDraw.Draw(self.image)
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self._reset_results()

    def _reset_results(self) -> None:
        """Clear the predicted digit, the 28x28 preview and the probability bars."""
        self.digit_label.config(text="-", fg=TEXT_LIGHT)
        self.conf_label.config(text="Draw a digit and click Predict", fg=TEXT_DIM)
        self.preview_canvas.delete("all")
        self._update_prob_bars(None, -1)

    def _preprocess(self) -> tuple[Image.Image | None, np.ndarray | None]:
        """Convert the drawing into a 28x28 image prepared like MNIST digits.

        Steps: crop to the drawn strokes, scale so the longer side is 20 px,
        paste into the center of a 28x28 image, then shift so the center of mass
        lands at the image center (without pushing strokes out of the frame).
        If SciPy is not available, a whole-pixel shift with ``np.roll`` is used instead.

        Returns:
            ``(image, tensor)``: the 28x28 image for the preview and the model input of
            shape ``(1, 28, 28, 1)`` with pixel values 0–255. ``(None, None)`` if the
            canvas is empty.
        """
        bbox = self.image.getbbox()
        if bbox is None:
            return None, None

        cropped = self.image.crop(bbox)
        w, h = cropped.size
        ratio = 20.0 / max(w, h)
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        resized = cropped.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # Paste into center of 28x28 canvas
        base_img = Image.new("L", (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        base_img.paste(resized, (paste_x, paste_y))

        arr = np.array(base_img, dtype=np.float32)

        # Center by center of mass with boundary clipping
        try:
            from scipy.ndimage import center_of_mass, shift
            cy, cx = center_of_mass(arr)
            if not np.isnan(cy) and not np.isnan(cx):
                shift_y = 14.0 - cy
                shift_x = 14.0 - cx

                # Keep digit strokes inside the 28x28 frame
                active_rows = np.where(arr > 10)[0]
                active_cols = np.where(arr > 10)[1]
                if len(active_rows) > 0:
                    min_r, max_r = active_rows.min(), active_rows.max()
                    min_c, max_c = active_cols.min(), active_cols.max()
                    shift_y = float(np.clip(shift_y, -(min_r - 1), 26 - max_r))
                    shift_x = float(np.clip(shift_x, -(min_c - 1), 26 - max_c))

                arr = shift(arr, [shift_y, shift_x], mode="constant", cval=0.0)
                arr = np.clip(arr, 0.0, 255.0)
        except Exception:
            total_mass = arr.sum()
            if total_mass > 0:
                cy = np.sum(np.arange(28)[:, None] * arr) / total_mass
                cx = np.sum(np.arange(28)[None, :] * arr) / total_mass
                shift_y = int(round(14.0 - cy))
                shift_x = int(round(14.0 - cx))
                arr = np.roll(arr, shift_y, axis=0)
                arr = np.roll(arr, shift_x, axis=1)

        final = Image.fromarray(arr.astype(np.uint8))
        tensor = arr.reshape(1, 28, 28, 1).astype("float32")
        return final, tensor

    def _predict(self) -> None:
        """Classify the drawing with the loaded model and show the results.

        Shows the predicted digit, its confidence, the 28x28 model input and the
        probability of every class.
        """
        if self.model is None:
            self.conf_label.config(text="No model loaded", fg="#ef4444")
            return

        final_img, tensor = self._preprocess()
        if final_img is None:
            self.conf_label.config(text="Canvas is empty", fg="#eab308")
            return

        preview_scaled = final_img.resize((84, 84), resample=Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(preview_scaled)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor="nw")

        if self.model_type == "keras":
            pred = self.model.predict(tensor, verbose=0)
        else:
            pred = self.model.predict(tensor)

        probs = pred.flatten()
        digit = int(np.argmax(probs))
        confidence = float(probs[digit]) * 100

        self.digit_label.config(text=str(digit), fg=ACCENT)
        self.conf_label.config(text=f"Digit {digit}  ({confidence:.1f}% confidence)", fg=ACCENT)
        self._update_prob_bars(probs, digit)

    def _update_prob_bars(self, probs: np.ndarray | None, top_digit: int) -> None:
        """Draw one horizontal bar per digit with its probability.

        Args:
            probs: Probabilities of digits 0–9, shape ``(10,)``, or ``None`` for empty bars.
            top_digit: Predicted digit, highlighted in the accent color (``-1`` for none).
        """
        self.prob_canvas.delete("all")
        w = self.prob_canvas.winfo_width()
        if w < 100:
            w = 400

        row_h = 24
        bar_x = 30
        bar_w = max(120, w - 100)

        for digit in range(10):
            y = 8 + digit * row_h
            p = float(probs[digit]) if probs is not None else 0.0
            is_winner = (digit == top_digit) and (probs is not None)

            label_fg = "#ffffff" if is_winner else TEXT_MUTED
            self.prob_canvas.create_text(14, y + 8, text=str(digit), fill=label_fg, font=("Segoe UI", 10, "bold" if is_winner else "normal"))
            self.prob_canvas.create_rectangle(bar_x, y + 3, bar_x + bar_w, y + 13, fill="#26262d", outline="")

            filled_len = int(p * bar_w)
            if filled_len > 0:
                bar_color = ACCENT if is_winner else "#3f3f4a"
                self.prob_canvas.create_rectangle(bar_x, y + 3, bar_x + filled_len, y + 13, fill=bar_color, outline="")

            pct_str = f"{p * 100:5.1f}%" if probs is not None else "  --.-%"
            pct_fg = TEXT_LIGHT if is_winner else TEXT_MUTED
            self.prob_canvas.create_text(bar_x + bar_w + 10, y + 8, text=pct_str, fill=pct_fg, font=("Segoe UI", 9, "bold" if is_winner else "normal"), anchor="w")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
