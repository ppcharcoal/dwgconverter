"""Tkinter desktop application."""
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import traceback
import webbrowser

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from . import __version__
from .export import Scale, available_engines, save
from .geometry import arc_params
from .raster import load_image
from .vectorize import Options, vectorize_file

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
ODA_URL = "https://www.opendesign.com/guestfiles/oda_file_converter"


def sample_entities(entities):
    """Turn result entities into polylines (Nx2 arrays, image coords)."""
    out = []

    def arc_pts(pa, pm, pb):
        ap = arc_params(pa, pm, pb)
        if ap is None:
            return np.array([pa, pb])
        a0 = math.atan2(pa[1] - ap["cy"], pa[0] - ap["cx"])
        n = max(8, int(abs(ap["sweep"]) * ap["r"] / 2))
        t = a0 + np.linspace(0, ap["sweep"], min(n, 400))
        return np.column_stack([ap["cx"] + ap["r"] * np.cos(t), ap["cy"] + ap["r"] * np.sin(t)])

    for e in entities:
        k = e[0]
        if k == "line":
            out.append(("line", np.array([e[1], e[2]])))
        elif k == "arc":
            out.append(("arc", arc_pts(e[1], e[2], e[3])))
        elif k == "circle":
            n = max(16, min(400, int(e[2])))
            t = np.linspace(0, 2 * math.pi, n)
            out.append(("circle", np.column_stack([e[1][0] + e[2] * np.cos(t),
                                                   e[1][1] + e[2] * np.sin(t)])))
        elif k == "poly":
            pts, mids, closed = e[1], e[2], e[3]
            seq = []
            n = len(pts)
            for i in range(n if closed else n - 1):
                a, b = pts[i], pts[(i + 1) % n]
                if mids[i] is not None:
                    seq.append(arc_pts(a, mids[i], b))
                else:
                    seq.append(np.array([a, b]))
            if seq:
                out.append(("poly", np.vstack(seq)))
    return out


class PreviewWindow(tk.Toplevel):
    COLORS = {"line": "#e0201b", "arc": "#1b5fe0", "circle": "#1b5fe0", "poly": "#13a35a"}

    def __init__(self, master, path, result):
        super().__init__(master)
        self.title(f"Preview – {os.path.basename(path)}")
        self.geometry("1100x780")
        gray, _ = load_image(path)
        self.img = Image.fromarray(gray).convert("L")
        self.W, self.H = self.img.size
        self.polys = sample_entities(result.entities)
        self.bboxes = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
                       for _, p in self.polys]
        self.ends = []
        for e in result.entities:
            if e[0] == "line":
                self.ends += [e[1], e[2]]
            elif e[0] == "arc":
                self.ends += [e[1], e[3]]
        self.show_img = tk.BooleanVar(value=True)
        self.show_vec = tk.BooleanVar(value=True)
        self.show_ends = tk.BooleanVar(value=False)

        bar = ttk.Frame(self)
        bar.pack(side="top", fill="x")
        ttk.Checkbutton(bar, text="Image", variable=self.show_img, command=self.redraw).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="Vectors", variable=self.show_vec, command=self.redraw).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="Endpoints", variable=self.show_ends, command=self.redraw).pack(side="left", padx=4)
        ttk.Button(bar, text="Fit", command=self.fit).pack(side="left", padx=4)
        ttk.Button(bar, text="1:1", command=lambda: self.set_zoom(1.0)).pack(side="left", padx=4)
        st = ", ".join(f"{v} {k}{'s' if v != 1 else ''}" for k, v in result.stats.items() if k != "seconds")
        ttk.Label(bar, text=f"   {st}   (red = lines, blue = arcs/circles, green = outlines)").pack(side="left")
        ttk.Label(bar, text="Wheel = zoom, drag = pan").pack(side="right", padx=6)

        self.cv = tk.Canvas(self, bg="white", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.zoom = 1.0
        self.ox = 0.0   # image coord at canvas left
        self.oy = 0.0
        self.cv.bind("<Configure>", lambda e: self.redraw())
        self.cv.bind("<MouseWheel>", self.on_wheel)
        self.cv.bind("<Button-4>", lambda e: self.on_wheel(e, 120))
        self.cv.bind("<Button-5>", lambda e: self.on_wheel(e, -120))
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_drag)
        self._photo = None
        self.after(50, self.fit)

    def fit(self):
        cw, ch = max(10, self.cv.winfo_width()), max(10, self.cv.winfo_height())
        self.zoom = min(cw / self.W, ch / self.H)
        self.ox = -(cw / self.zoom - self.W) / 2
        self.oy = -(ch / self.zoom - self.H) / 2
        self.redraw()

    def set_zoom(self, z, cx=None, cy=None):
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        cx = cw / 2 if cx is None else cx
        cy = ch / 2 if cy is None else cy
        ix, iy = self.ox + cx / self.zoom, self.oy + cy / self.zoom
        self.zoom = max(0.02, min(40.0, z))
        self.ox, self.oy = ix - cx / self.zoom, iy - cy / self.zoom
        self.redraw()

    def on_wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        self.set_zoom(self.zoom * (1.25 if d > 0 else 0.8), e.x, e.y)

    def on_press(self, e):
        self._drag = (e.x, e.y, self.ox, self.oy)

    def on_drag(self, e):
        x0, y0, ox, oy = self._drag
        self.ox = ox - (e.x - x0) / self.zoom
        self.oy = oy - (e.y - y0) / self.zoom
        self.redraw()

    def redraw(self):
        cv = self.cv
        cv.delete("all")
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw < 2 or ch < 2:
            return
        z = self.zoom
        x0, y0 = self.ox, self.oy
        x1, y1 = x0 + cw / z, y0 + ch / z
        if self.show_img.get():
            l, t = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
            r, b = min(self.W, int(math.ceil(x1))), min(self.H, int(math.ceil(y1)))
            if r > l and b > t:
                crop = self.img.crop((l, t, r, b))
                dw, dh = max(1, int(round((r - l) * z))), max(1, int(round((b - t) * z)))
                crop = crop.resize((dw, dh), Image.NEAREST if z >= 1 else Image.BILINEAR)
                crop = crop.point(lambda v: 255 - (255 - v) * 0.4)
                self._photo = ImageTk.PhotoImage(crop)
                cv.create_image((l - x0) * z, (t - y0) * z, image=self._photo, anchor="nw")
        if self.show_vec.get():
            # pixel centres are at integer coords -> +0.5 to sit on the image
            for (kind, p), bb in zip(self.polys, self.bboxes):
                if bb[2] < x0 - 1 or bb[0] > x1 + 1 or bb[3] < y0 - 1 or bb[1] > y1 + 1:
                    continue
                q = (p + 0.5 - [x0, y0]) * z
                cv.create_line(*q.ravel().tolist(), fill=self.COLORS[kind], width=1)
        if self.show_ends.get():
            r = 3
            for e in self.ends:
                if x0 <= e[0] <= x1 and y0 <= e[1] <= y1:
                    X, Y = (e[0] + 0.5 - x0) * z, (e[1] + 0.5 - y0) * z
                    cv.create_oval(X - r, Y - r, X + r, Y + r, outline="#222")


class App(tk.Tk):
    def __init__(self, files=()):
        super().__init__()
        self.title(f"Image → DWG Converter {__version__}")
        self.geometry("980x700")
        self.minsize(860, 600)
        try:
            ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
        except tk.TclError:
            pass
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self._build()
        for f in files:
            self.add_file(f)
        self.refresh_engines()
        self.after(100, self.poll)

    # ------------------------------------------------------------------ UI
    def _build(self):
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        # files
        lf = ttk.LabelFrame(main, text="Images", padding=6)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        self.lst = tk.Listbox(lf, selectmode="extended", activestyle="none")
        self.lst.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.lst.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.lst.configure(yscrollcommand=sb.set)
        self.lst.bind("<Double-Button-1>", lambda e: self.preview())
        bf = ttk.Frame(lf)
        bf.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(bf, text="Add images…", command=self.pick_files).pack(side="left")
        ttk.Button(bf, text="Add folder…", command=self.pick_folder).pack(side="left", padx=4)
        ttk.Button(bf, text="Remove", command=self.remove_sel).pack(side="left")
        ttk.Button(bf, text="Clear", command=lambda: self.lst.delete(0, "end")).pack(side="left", padx=4)

        # settings
        sf = ttk.Frame(main)
        sf.grid(row=0, column=1, sticky="ns")
        self.v = {
            "mode": tk.StringVar(value="auto"),
            "tol": tk.DoubleVar(value=1.5),
            "auto_thr": tk.BooleanVar(value=True),
            "thr": tk.IntVar(value=128),
            "ink": tk.StringVar(value="auto"),
            "maxw": tk.DoubleVar(value=12.0),
            "speck": tk.IntVar(value=6),
            "arcs": tk.BooleanVar(value=True),
            "snap": tk.BooleanVar(value=True),
            "poly": tk.BooleanVar(value=False),
            "fill": tk.BooleanVar(value=False),
            "denoise": tk.BooleanVar(value=False),
            "frame": tk.BooleanVar(value=False),
            "units": tk.StringVar(value="Pixels (1 px = 1 unit)"),
            "dpi": tk.StringVar(value="auto"),
            "fmt": tk.StringVar(value="DWG"),
            "ver": tk.StringVar(value="2018"),
            "engine": tk.StringVar(value="auto"),
            "keepdxf": tk.BooleanVar(value=False),
            "outdir": tk.StringVar(value=""),
        }
        v = self.v
        g = ttk.LabelFrame(sf, text="Tracing", padding=6)
        g.pack(fill="x")
        r = 0

        def row(label, widget, tip=None):
            nonlocal r
            ttk.Label(g, text=label).grid(row=r, column=0, sticky="w", pady=2)
            widget.grid(row=r, column=1, sticky="ew", pady=2, padx=(6, 0))
            r += 1

        row("Mode", ttk.Combobox(g, textvariable=v["mode"], state="readonly", width=16,
                                 values=["auto", "centerline", "outline"]))
        row("Tolerance (px)", ttk.Spinbox(g, textvariable=v["tol"], from_=0.3, to=10, increment=0.1, width=8))
        tf = ttk.Frame(g)
        ttk.Checkbutton(tf, text="auto", variable=v["auto_thr"]).pack(side="left")
        ttk.Spinbox(tf, textvariable=v["thr"], from_=1, to=254, width=5).pack(side="left", padx=4)
        row("Threshold", tf)
        row("Ink colour", ttk.Combobox(g, textvariable=v["ink"], state="readonly", width=16,
                                       values=["auto", "dark", "light"]))
        row("Max line width (px)", ttk.Spinbox(g, textvariable=v["maxw"], from_=2, to=200, increment=1, width=8))
        row("Ignore specks < (px²)", ttk.Spinbox(g, textvariable=v["speck"], from_=0, to=500, width=8))
        for key, text in (("arcs", "Detect arcs and circles"), ("snap", "Snap near horizontal/vertical"),
                          ("poly", "Join segments into polylines"), ("fill", "Solid-fill filled areas"),
                          ("denoise", "Smooth JPEG noise"), ("frame", "Add image border rectangle")):
            ttk.Checkbutton(g, text=text, variable=v[key]).grid(row=r, column=0, columnspan=2, sticky="w")
            r += 1

        g = ttk.LabelFrame(sf, text="Scale / units", padding=6)
        g.pack(fill="x", pady=6)
        r = 0
        row("Units", ttk.Combobox(g, textvariable=v["units"], state="readonly", width=20,
                                  values=["Pixels (1 px = 1 unit)", "Millimetres", "Centimetres", "Inches"]))
        row("DPI", ttk.Entry(g, textvariable=v["dpi"], width=10))

        g = ttk.LabelFrame(sf, text="Output", padding=6)
        g.pack(fill="x")
        r = 0
        row("Format", ttk.Combobox(g, textvariable=v["fmt"], state="readonly", width=10, values=["DWG", "DXF"]))
        row("DWG version", ttk.Combobox(g, textvariable=v["ver"], state="readonly", width=10,
                                        values=["2000", "2004", "2007", "2010", "2013", "2018"]))
        row("DWG engine", ttk.Combobox(g, textvariable=v["engine"], state="readonly", width=10,
                                       values=["auto", "oda", "libredwg"]))
        ttk.Checkbutton(g, text="Also keep a DXF copy", variable=v["keepdxf"]).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        of = ttk.Frame(g)
        ttk.Entry(of, textvariable=v["outdir"], width=16).pack(side="left", fill="x", expand=True)
        ttk.Button(of, text="…", width=3, command=self.pick_outdir).pack(side="left")
        row("Output folder", of)
        ttk.Label(g, text="(empty = next to each image)", foreground="#666").grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.eng_lbl = ttk.Label(g, text="", foreground="#555", wraplength=260)
        self.eng_lbl.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 0))
        r += 1
        self.oda_btn = ttk.Button(g, text="Get ODA File Converter (free)…",
                                  command=lambda: webbrowser.open(ODA_URL))
        self.oda_btn.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        # bottom
        bot = ttk.Frame(main)
        bot.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bot.columnconfigure(2, weight=1)
        self.btn_prev = ttk.Button(bot, text="Preview selected", command=self.preview)
        self.btn_prev.grid(row=0, column=0)
        self.btn_conv = ttk.Button(bot, text="Convert all", command=self.convert)
        self.btn_conv.grid(row=0, column=1, padx=6)
        self.pb = ttk.Progressbar(bot, mode="determinate")
        self.pb.grid(row=0, column=2, sticky="ew")
        self.status = ttk.Label(bot, text="Ready")
        self.status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.log = tk.Text(main, height=8, wrap="word", state="disabled", background="#fafafa")
        self.log.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def refresh_engines(self):
        e = available_engines()
        parts = []
        if e["oda"]:
            parts.append("ODA File Converter found (best DWG compatibility).")
            self.oda_btn.grid_remove()
        else:
            parts.append("ODA File Converter not installed.")
        parts.append("Built-in LibreDWG: " + ("available (writes DWG 2000)." if e["libredwg"] else "missing."))
        self.eng_lbl.configure(text=" ".join(parts))

    # ------------------------------------------------------------ helpers
    def add_file(self, f):
        f = os.path.abspath(f)
        if os.path.isdir(f):
            for n in sorted(os.listdir(f)):
                if n.lower().endswith(IMAGE_TYPES):
                    self.add_file(os.path.join(f, n))
            return
        if f.lower().endswith(IMAGE_TYPES) and f not in self.lst.get(0, "end"):
            self.lst.insert("end", f)

    def pick_files(self):
        for f in filedialog.askopenfilenames(
                title="Choose images",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]):
            self.add_file(f)

    def pick_folder(self):
        d = filedialog.askdirectory(title="Folder with images")
        if d:
            self.add_file(d)

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.v["outdir"].set(d)

    def remove_sel(self):
        for i in reversed(self.lst.curselection()):
            self.lst.delete(i)

    def write_log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def options(self):
        v = self.v
        return Options(
            mode=v["mode"].get(), tolerance=float(v["tol"].get()),
            threshold=None if v["auto_thr"].get() else int(v["thr"].get()),
            ink=v["ink"].get(), denoise=v["denoise"].get(), min_speck=int(v["speck"].get()),
            max_line_width=float(v["maxw"].get()), detect_arcs=v["arcs"].get(),
            snap_orthogonal=v["snap"].get(), join_polylines=v["poly"].get(),
            fill_outlines=v["fill"].get())

    def scale_for(self, image_dpi):
        u = self.v["units"].get()
        if u.startswith("Pixels"):
            return Scale(1.0, "unitless")
        units = {"Millimetres": "mm", "Centimetres": "cm", "Inches": "in"}[u]
        d = self.v["dpi"].get().strip().lower()
        try:
            dpi = float(d)
        except ValueError:
            dpi = image_dpi or 300.0
        return Scale.from_dpi(dpi, units)

    def set_busy(self, b):
        self.busy = b
        st = "disabled" if b else "normal"
        self.btn_prev.configure(state=st)
        self.btn_conv.configure(state=st)

    # -------------------------------------------------------------- actions
    def preview(self):
        if self.busy:
            return
        sel = self.lst.curselection()
        if not sel:
            if self.lst.size() == 0:
                messagebox.showinfo("Preview", "Add an image first.")
                return
            sel = (0,)
        path = self.lst.get(sel[0])
        try:
            opts = self.options()
        except (tk.TclError, ValueError) as ex:
            messagebox.showerror("Settings", str(ex))
            return
        self.set_busy(True)
        self.pb.configure(mode="indeterminate")
        self.pb.start(12)

        def work():
            try:
                res = vectorize_file(path, opts, progress=lambda m: self.q.put(("status", m)))
                self.q.put(("preview", (path, res)))
            except Exception as ex:
                self.q.put(("error", f"{os.path.basename(path)}: {ex}\n{traceback.format_exc()}"))
            self.q.put(("done", None))
        threading.Thread(target=work, daemon=True).start()

    def convert(self):
        if self.busy:
            return
        files = list(self.lst.get(0, "end"))
        if not files:
            messagebox.showinfo("Convert", "Add one or more images first.")
            return
        try:
            opts = self.options()
        except (tk.TclError, ValueError) as ex:
            messagebox.showerror("Settings", str(ex))
            return
        fmt = self.v["fmt"].get().lower()
        ver = self.v["ver"].get()
        engine = self.v["engine"].get()
        keep = self.v["keepdxf"].get()
        frame = self.v["frame"].get()
        outdir = self.v["outdir"].get().strip()
        self.set_busy(True)
        self.pb.configure(mode="determinate", maximum=len(files), value=0)

        def work():
            ok = 0
            for i, f in enumerate(files):
                try:
                    self.q.put(("status", f"[{i + 1}/{len(files)}] {os.path.basename(f)}"))
                    res = vectorize_file(f, opts, progress=lambda m, i=i: self.q.put(
                        ("status", f"[{i + 1}/{len(files)}] {os.path.basename(f)}: {m}")))
                    base = os.path.splitext(os.path.basename(f))[0]
                    d = outdir or os.path.dirname(f)
                    os.makedirs(d, exist_ok=True)
                    info = save(res, os.path.join(d, base + "." + fmt), self.scale_for(res.dpi),
                                fmt, engine, ver, frame, keep)
                    st = ", ".join(f"{v} {k}" for k, v in res.stats.items() if k != "seconds")
                    self.q.put(("log", f"✓ {os.path.basename(f)} → {', '.join(info['files'])}  "
                                       f"[{info['engine']}]  {st}"))
                    ok += 1
                except Exception as ex:
                    self.q.put(("log", f"✗ {os.path.basename(f)}: {ex}"))
                self.q.put(("progress", i + 1))
            self.q.put(("status", f"Finished: {ok} of {len(files)} converted"))
            self.q.put(("done", None))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "status":
                    self.status.configure(text=data)
                elif kind == "log":
                    self.write_log(data)
                elif kind == "progress":
                    self.pb.configure(value=data)
                elif kind == "preview":
                    path, res = data
                    st = ", ".join(f"{v} {k}" for k, v in res.stats.items())
                    self.status.configure(text=f"Preview ready: {st}")
                    PreviewWindow(self, path, res)
                elif kind == "error":
                    self.write_log("✗ " + data.splitlines()[0])
                    messagebox.showerror("Error", data.splitlines()[0])
                elif kind == "done":
                    self.pb.stop()
                    self.set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self.poll)


def selftest(files, out_dir):
    """Used by CI: open the window, run a preview and a conversion, save
    screenshots, exit with a status code."""
    from PIL import ImageGrab
    os.makedirs(out_dir, exist_ok=True)
    app = App(files)
    app.geometry("1100x760+20+20")
    app.v["fmt"].set("DXF")
    app.v["outdir"].set(out_dir)
    state = {"rc": 1, "step": 0}

    def shot(name, win):
        win.update()
        x, y = win.winfo_rootx(), win.winfo_rooty()
        w, h = win.winfo_width(), win.winfo_height()
        try:
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(os.path.join(out_dir, name))
        except Exception as ex:  # no display grab available
            print("screenshot failed:", ex)

    def step():
        st = state["step"]
        if st == 0:
            app.lst.selection_set(0)
            app.preview()
            state["step"] = 1
        elif st == 1 and not app.busy:
            wins = [w for w in app.winfo_children() if isinstance(w, PreviewWindow)]
            if wins:
                wins[0].geometry("1100x760+40+40")
                wins[0].fit()
                shot("preview.png", wins[0])
                wins[0].destroy()
            state["step"] = 2 if wins else 9
        elif st == 2:
            app.convert()
            state["step"] = 3
        elif st == 3 and not app.busy:
            shot("main.png", app)
            outs = [f for f in os.listdir(out_dir) if f.lower().endswith(".dxf")]
            print("converted:", outs)
            print(app.log.get("1.0", "end"))
            state["rc"] = 0 if len(outs) == len(files) else 1
            app.destroy()
            return
        elif st == 9:
            print("preview window did not open")
            app.destroy()
            return
        app.after(300, step)

    app.after(800, step)
    app.after(180000, app.destroy)
    app.mainloop()
    return state["rc"]


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        sys.exit(selftest(args[2:], args[1]))
    files = [a for a in args if not a.startswith("-")]
    App(files).mainloop()


if __name__ == "__main__":
    main()
