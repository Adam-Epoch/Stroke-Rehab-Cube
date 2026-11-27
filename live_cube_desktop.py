import re
import sys
import time
from collections import deque

import numpy as np
import serial
from scipy.spatial.transform import Rotation
from threading import Event, Lock, Thread
import tkinter as tk
from tkinter import font as tkfont

BG = "#101820"
PANEL = "#1a2632"
PLOT = "#111c26"
TEXT = "#edf3f6"
MUTED = "#aebfca"
ACCENT = "#88ddce"

PORT = "COM3"              # Change if your XIAO uses a different port
BAUD_RATE = 115200
GRAVITY = 9.81
TILT_CORRECTION_TIME = 0.1  # Seconds; smaller = faster accelerometer correction

# Startup plot limits in X, Y, Z order. Use None for automatic scaling
# or a (lower, upper) pair per axis, e.g. ((-50, 50), None, None).
DEFAULT_BAR_LIMITS = {
    "Hall": ((-2500, 2500), (-2500, 2500), (-2500, 2500)),   # X, Y, Z in µT
    "Hall2": ((-2500, 2500), (-2500, 2500), (-2500, 2500)),
}

DEFAULT_AXIS_LIMITS = {
    "Hall":  ((-2500, 2500), (-2500, 2500), (-2500, 2500)),  # Magnetic field (µT)
    "Hall2": ((-2500, 2500), (-2500, 2500), (-2500, 2500)),
    "Accel": ((-10,10), (-10,10), (-10,10)),  # Acceleration (m/s²)
    "Gyro":  ((-10,10), (-10,10), (-10,10)),  # Angular velocity (rad/s)
}

# Matches the Arduino sketch's "X: ...  Y: ...  Z: ..." format.
NUMBER = r"([-+]?\d+(?:\.\d+)?)"
AXES = re.compile(rf"X:\s*{NUMBER}\s+Y:\s*{NUMBER}\s+Z:\s*{NUMBER}")

WORLD_UP = np.array([0.0, 0.0, 1.0])


def rotation_between(a, b):
    """Shortest rotation that takes unit vector a onto unit vector b."""
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis)
    cosine = np.clip(np.dot(a, b), -1.0, 1.0)

    if sine > 1e-8:
        return Rotation.from_rotvec(axis * (np.arctan2(sine, cosine) / sine))
    if cosine > 0:
        return Rotation.identity()

    # Exactly upside down: choose an axis perpendicular to a.
    basis = np.eye(3)[np.argmin(np.abs(a))]
    axis = np.cross(a, basis)
    return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis))


# Model coordinates follow the board axes. The camera is fixed above the cube.
FACES = (
    ((0, 3, 2, 1), (0, 0, -1), "#30588d"),
    ((4, 5, 6, 7), (0, 0, 1), "#43b860"),
    ((1, 2, 6, 5), (1, 0, 0), "#df5454"),
    ((0, 4, 7, 3), (-1, 0, 0), "#467fc0"),
    ((3, 7, 6, 2), (0, 1, 0), "#386da8"),
    ((0, 1, 5, 4), (0, -1, 0), "#538cd0"),
)
VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
]) * 0.5
CAMERA = np.array([1.0, -1.6, 0.9])
CAMERA /= np.linalg.norm(CAMERA)
RIGHT = np.cross(WORLD_UP, CAMERA)
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(CAMERA, RIGHT)

latest_orientation = Rotation.identity()
SENSORS = {
    "Hall": ("Hall sensor 1", "µT", 0.1),
    "Hall2": ("Hall sensor 2", "µT", 0.1),
    "Accel": ("Acceleration", "m/s²", 0.01),
    "Gyro": ("Angular velocity", "rad/s", 0.01),
}
history = {key: deque(maxlen=4096) for key in SENSORS}
hall_slots = {}
hall_channel = re.compile(r"^Hall CH([0-3])\b")
hall_status = "Waiting for Hall readings"
sensor_lock = Lock()
offsets = {key: np.zeros(3) for key in SENSORS}
totals = {}
counts = {}
recording = False
TRACE_SECONDS = 10
stop = Event()
serial_status = f"Connecting to {PORT}..."


def draw_cube(orientation, canvas):
    vertices = orientation.apply(VERTICES)
    width, height = canvas.winfo_width(), canvas.winfo_height()
    scale = 0.53 * min(width, height)
    points = np.column_stack((
        width / 2 + scale * (vertices @ RIGHT),
        height / 2 - scale * (vertices @ UP),
    ))
    canvas.delete("cube")
    for indices, normal, fill in FACES:
        if orientation.apply(normal) @ CAMERA > 0:
            canvas.create_polygon(
                *points[list(indices)].ravel(), fill=fill,
                outline="#142233", width=2, tags="cube",
            )



def draw_traces(canvas, samples, now, offset, resolution, limits, dpi_scale=1.0):
    canvas.delete("trace")
    left, right = 72, canvas.winfo_width() / dpi_scale - 16
    top, bottom = 12, canvas.winfo_height() / dpi_scale - 24
    if right <= left or bottom <= top:
        return

    lane_height = (bottom - top) / 3
    start = now - TRACE_SECONDS
    times = [timestamp for timestamp, _ in samples]
    positions = [left + (timestamp - start) / TRACE_SECONDS * (right - left)
                 for timestamp in times]
    for second in range(TRACE_SECONDS + 1):
        x = left + second / TRACE_SECONDS * (right - left)
        canvas.create_line(x, top, x, bottom, fill="#304152", tags="trace")
        if second % 2 == 0:
            label = "now" if second == TRACE_SECONDS else f"{second - TRACE_SECONDS} s"
            canvas.create_text(x, bottom + 12, text=label, fill="#b8c8d7", tags="trace")

    for axis, (label, color) in enumerate(zip("XYZ", ("#ff8989", "#64dc98", "#77c4ff"))):
        lane_top = top + axis * lane_height
        lane_bottom = lane_top + lane_height
        canvas.create_line(left, lane_bottom, right, lane_bottom, fill="#425668", tags="trace")
        canvas.create_line(left, (lane_top + lane_bottom) / 2, right,
                           (lane_top + lane_bottom) / 2, fill="#304152", tags="trace")
        canvas.create_text(10, (lane_top + lane_bottom) / 2, text=label,
                           anchor="w", fill=color, tags="trace")
        if not samples:
            continue
        values = [field[axis] - offset[axis] for _, field in samples]
        midpoint = (min(values) + max(values)) / 2
        half_span = max(resolution, (max(values) - min(values)) * 0.55)
        low, high = midpoint - half_span, midpoint + half_span
        if limits[axis] is not None:
            low, high = limits[axis]
            values = np.clip(values, low, high)
        if lane_height >= 25:
            for limit, y, anchor in ((high, lane_top, "ne"), (low, lane_bottom, "se")):
                canvas.create_text(left - 5, y, text=f"{limit:g}", anchor=anchor,
                                   font=("TkDefaultFont", 8), fill=MUTED, tags="trace")
        points = [coordinate for x, value in zip(positions, values)
                  for coordinate in (x, lane_bottom - (value - low) / (high - low) * lane_height)]
        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=2 * dpi_scale, tags="trace")
        else:
            x, y = points
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color,
                               outline="", tags="trace")
    canvas.scale("trace", 0, 0, dpi_scale, dpi_scale)


def read_serial():
    global latest_orientation, serial_status, hall_status

    orientation = Rotation.identity()
    accel = None
    last_time = None
    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=0.1) as port:
            serial_status = f"Connected to {PORT}. Waiting for sensor readings..."
            while not stop.is_set():
                line = port.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("TMAG5273 at"):
                    hall_status = line
                    continue
                match = AXES.search(line)
                if not match:
                    if line.startswith("Hall"):
                        hall_status = "Invalid Hall reading: " + line[:80]
                    continue
                values = np.array([float(n) for n in match.groups()])
                key = line.split(maxsplit=1)[0]
                if key == "Hall":
                    channel = hall_channel.match(line)
                    source = channel.group(1) if channel else "direct"
                    if source not in hall_slots:
                        if len(hall_slots) == 2:
                            continue
                        hall_slots[source] = ("Hall", "Hall2")[len(hall_slots)]
                    key = hall_slots[source]
                if key not in history:
                    continue
                if key == "Hall":
                    hall_status = "Hall readings stopped"
                with sensor_lock:
                    history[key].append((time.monotonic(), values))
                    if recording:
                        totals[key] += values
                        counts[key] += 1
                if line.startswith("Accel"):
                    accel = values
                    continue
                if not line.startswith("Gyro") or accel is None:
                    continue
                gyro = values  # Your Adafruit sketch already prints rad/s
                now = time.monotonic()
                magnitude = np.linalg.norm(accel)
                near_gravity = 0.85 * GRAVITY < magnitude < 1.15 * GRAVITY
                if last_time is None:
                    if not near_gravity:
                        continue
                    # Your convention: measured -Z means the +Z face points up.
                    orientation = rotation_between(-accel / magnitude, WORLD_UP)
                else:
                    dt = now - last_time
                    if 0 < dt < 0.75:
                        # Gyroscope measures rotation in the board's own axes.
                        orientation = orientation * Rotation.from_rotvec(gyro * dt)
                        if near_gravity:
                            measured_up = orientation.apply(-accel / magnitude)
                            correction = rotation_between(measured_up, WORLD_UP)
                            weight = 1 - np.exp(-dt / TILT_CORRECTION_TIME)
                            orientation = (
                                Rotation.from_rotvec(weight * correction.as_rotvec())
                                * orientation
                            )
                last_time = now
                accel = None
                latest_orientation = orientation
                serial_status = f'Live on {PORT}'
    except (serial.SerialException, OSError) as error:
        serial_status = f"Could not use {PORT}: {error}"


def draw_hall_bars(canvas, readings, limits, dpi_scale=1.0):
    canvas.delete("all")
    width, height = canvas.winfo_width() / dpi_scale, canvas.winfo_height() / dpi_scale
    if width < 80 or height < 70:
        return
    peak = max((max(abs(values)) for _, values in readings if values is not None), default=0)
    step = 10 ** np.floor(np.log10(max(peak, 1.0)))
    limit = np.ceil(max(peak, 1.0) / step) * step
    for column, (name, values) in enumerate(readings):
        centre = (column + 0.5) * width / 2
        radius = width / 4 - 22
        canvas.create_text(centre, 12, text=name, fill="white")
        if values is None:
            canvas.create_text(centre, height / 2, text="Waiting for readings", fill="#b8c8d7")
            continue
        for axis, (value, color) in enumerate(zip(values, ("#ff8989", "#64dc98", "#77c4ff"))):
            y = 24 + (axis + 0.5) * (height - 44) / 3
            canvas.create_text(centre - radius - 9, y, text="XYZ"[axis], fill=color)
            low, high = limits[("Hall", "Hall2")[column]][axis] or (-limit, limit)
            left = centre - radius
            scale = 2 * radius / (high - low)
            origin = left + (np.clip(0, low, high) - low) * scale
            end = left + (np.clip(value, low, high) - low) * scale
            canvas.create_line(origin, y, end, y, fill=color, width=8 * dpi_scale)
            canvas.create_line(origin, y - 5, origin, y + 5, fill="#b8c8d7")
            canvas.create_text(left, y + 6, text=f"{low:g}", anchor="nw", fill="#b8c8d7")
            canvas.create_text(centre + radius, y + 6, text=f"{high:g}", anchor="ne", fill="#b8c8d7")
    canvas.scale("all", 0, 0, dpi_scale, dpi_scale)


def enable_native_dpi():
    if sys.platform == "win32":
        import ctypes
        user32 = ctypes.windll.user32
        try:
            # Per-monitor v2. This must run before Tk creates the first HWND.
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
        except AttributeError:
            pass
        user32.SetProcessDPIAware()


def main():
    enable_native_dpi()
    window = tk.Tk()
    window.title("Cube monitor")
    dpi_scale = window.winfo_fpixels("1i") / 96 if sys.platform == "win32" else 1.0
    width = min(round(1260 * dpi_scale), window.winfo_screenwidth() - 40)
    height = min(round(840 * dpi_scale), window.winfo_screenheight() - 90)
    window.geometry(f"{width}x{height}")
    window.minsize(min(width, round(840 * dpi_scale)), min(height, round(620 * dpi_scale)))
    window.configure(bg=BG)
    ui_font = "Inter" if "Inter" in tkfont.families(window) else "Segoe UI"
    window.option_add("*Font", (ui_font, 9))
    window.option_add("*Frame.Background", BG)
    window.option_add("*Label.Background", BG)
    window.option_add("*Label.Foreground", TEXT)
    window.option_add("*Button.Background", "#304351")
    window.option_add("*Button.Foreground", TEXT)
    window.option_add("*Button.ActiveBackground", "#425d6c")
    window.option_add("*Button.ActiveForeground", TEXT)
    window.option_add("*Button.Relief", "flat")
    window.option_add("*Button.BorderWidth", 0)
    window.columnconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)
    top = tk.Frame(window)
    top.grid(row=0, column=0, sticky="ew", padx=14, pady=(9, 7))
    tk.Label(top, text="Cube monitor", font=(ui_font, 18, "bold")).pack(side="left")
    status = tk.StringVar(value=serial_status)
    tk.Label(top, textvariable=status, fg=MUTED, wraplength=500, justify="right").pack(side="right")
    body = tk.Frame(window)
    body.grid(row=1, column=0, sticky="nsew", padx=12)
    body.columnconfigure(0, weight=2, uniform="panels")
    body.columnconfigure(1, weight=3, uniform="panels")
    body.rowconfigure(0, weight=1)
    left = tk.Frame(body, bg=PANEL)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    left.columnconfigure(0, weight=1)
    left.rowconfigure(1, weight=3)
    left.rowconfigure(4, weight=2, minsize=round(140 * dpi_scale))
    right = tk.Frame(body)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    charts = {}
    limits = {key: list(DEFAULT_AXIS_LIMITS[key]) for key in SENSORS}

    bar_limits = {key: list(value) for key, value in DEFAULT_BAR_LIMITS.items()}

    def edit_limits(key, target=None):
        from tkinter import messagebox

        if target is None:
            target = limits
        dialog = tk.Toplevel(window)
        name, unit, _ = SENSORS[key]
        for source, slot in hall_slots.copy().items():
            if slot == key:
                name = "Hall direct" if source == "direct" else f"Hall CH{source}"
        kind = "bar" if target is bar_limits else "trace"
        dialog.title(f"{name} · {kind} limits ({unit})")
        dialog.configure(bg=BG, padx=12, pady=10)
        dialog.transient(window)
        dialog.focus_set()
        dialog.grab_set()
        entries = []
        for column, title in enumerate(("Axis", "Lower", "Upper")):
            tk.Label(dialog, text=title).grid(row=0, column=column)
        for index, axis in enumerate("XYZ"):
            tk.Label(dialog, text=axis).grid(row=index + 1, column=0)
            pair = []
            for column in range(2):
                entry = tk.Entry(dialog, width=12)
                entry.grid(row=index + 1, column=column + 1, padx=4, pady=3)
                if target[key][index] is not None:
                    entry.insert(0, str(target[key][index][column]))
                pair.append(entry)
            entries.append(pair)
        tk.Label(dialog, text="Leave both limits blank for automatic scaling.").grid(
            row=4, columnspan=3, padx=8, pady=4)

        def apply_limits():
            updated = []
            for axis, pair in zip("XYZ", entries):
                lower, upper = (entry.get().strip() for entry in pair)
                if not lower and not upper:
                    updated.append(None)
                    continue
                try:
                    low, high = float(lower), float(upper)
                    if not (np.isfinite(low) and np.isfinite(high) and low < high):
                        raise ValueError
                except ValueError:
                    messagebox.showerror("Invalid limits", f"{axis} needs two finite values with lower < upper.", parent=dialog)
                    return
                updated.append((low, high))
            target[key] = updated
            dialog.destroy()

        tk.Button(dialog, text="Apply", command=apply_limits).grid(row=5, columnspan=3, pady=5)
        dialog.bind("<Return>", lambda event: apply_limits())
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        entries[0][0].focus_set()

    for index, (key, (name, unit, _)) in enumerate(SENSORS.items()):
        row = index * 2
        label = tk.StringVar(value=f"{name} ({unit})")
        header = tk.Frame(right, bg=PANEL)
        header.grid(row=row, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        tk.Button(header, text="Scale…", command=lambda key=key: edit_limits(key)).grid(row=0, column=1, rowspan=2, padx=6)
        tk.Label(header, textvariable=label, bg=PANEL, font=(ui_font, 10, "bold")).grid(row=0, column=0, sticky="w", padx=8)
        readout = tk.StringVar()
        tk.Label(header, textvariable=readout, bg=PANEL, fg=MUTED, width=1, font="TkFixedFont", anchor="w").grid(row=1, column=0, sticky="ew", padx=8)
        plot = tk.Canvas(right, width=1, height=1, background=PLOT, highlightthickness=0)
        plot.grid(row=row + 1, column=0, sticky="nsew", pady=(0, 5))
        right.rowconfigure(row + 1, weight=1)
        charts[key] = (label, readout, plot)

    tk.Label(left, text="Orientation", bg=PANEL, font=(ui_font, 11, "bold")).grid(row=0, sticky="w", padx=10, pady=6)
    canvas = tk.Canvas(left, width=1, height=1, background=PLOT, highlightthickness=0)
    canvas.grid(row=1, column=0, sticky="nsew")
    tk.Label(left, text="Green   +Z top      Red   +X face", bg=PLOT, fg=MUTED).grid(row=2, sticky="ew")
    bars = tk.Canvas(left, width=1, height=1, background=PLOT, highlightthickness=0)
    controls = tk.Frame(left, bg=PANEL)
    controls.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
    tk.Label(controls, text="Magnetic field (µT)", bg=PANEL, font=(ui_font, 10, "bold")).pack(anchor="w")
    bar_buttons = {}
    for key in ("Hall", "Hall2"):
        bar_buttons[key] = tk.StringVar(value=f"{SENSORS[key][0]} scale…")
        tk.Button(controls, textvariable=bar_buttons[key], command=lambda key=key: edit_limits(key, bar_limits)).pack(side="left", expand=True)
    bars.grid(row=4, column=0, sticky="nsew")
    footer = tk.Frame(window)
    footer.grid(row=2, sticky="ew", padx=14, pady=8)
    baseline_status = tk.StringVar(value="Hold the sensors still to capture background levels.")
    baseline_button = tk.Button(footer, text="Record baseline", bg=ACCENT, fg=BG, padx=10,
                                command=lambda: toggle_baseline(None))
    baseline_button.pack(side="left")
    tk.Label(footer, text="Space", fg=MUTED).pack(side="left", padx=5)
    tk.Label(footer, textvariable=baseline_status, fg=MUTED).pack(side="left", padx=8)

    def toggle_baseline(event):
        global recording
        with sensor_lock:
            if recording:
                applied = []
                for key in SENSORS:
                    if counts[key]:
                        offsets[key] = totals[key] / counts[key]
                        applied.append(key)
                message = "Baseline applied to " + ", ".join(applied) if applied else "No samples received"
                message += " · Space to record again"
            else:
                for key in SENSORS:
                    totals[key] = np.zeros(3)
                    counts[key] = 0
                message = "Recording baseline · Hold still · Space to finish"
            recording = not recording
        baseline_status.set(message)
        baseline_button.configure(text="Finish baseline" if recording else "Record baseline",
                                  bg="#ffd18f" if recording else ACCENT)

    space_down = False

    def space_press(event):
        if isinstance(event.widget, tk.Button):
            return
        nonlocal space_down
        if not space_down:
            space_down = True
            toggle_baseline(event)

    def space_release(event):
        nonlocal space_down
        space_down = False

    window.bind("<KeyPress-space>", space_press)
    window.bind("<KeyRelease-space>", space_release)
    window.bind("<FocusOut>", space_release)

    def refresh():
        draw_cube(latest_orientation, canvas)
        status.set(serial_status)
        window.after(16, refresh)

    def refresh_traces():
        now = time.monotonic()
        hall_readings = []
        for key, (label, readout, plot) in charts.items():
            with sensor_lock:
                data = history[key]
                while data and data[0][0] < now - TRACE_SECONDS:
                    data.popleft()
                samples = list(data)
                offset = offsets[key]
            name, unit, resolution = SENSORS[key]
            for source, slot in hall_slots.copy().items():
                if slot == key:
                    name = "Hall direct" if source == "direct" else f"Hall CH{source}"
            caption = f"{name} ({unit})"
            if samples:
                vector = samples[-1][1] - offset
                x, y, z = vector
                readout.set(f"X{x:8.2f} Y{y:8.2f} Z{z:8.2f} |v|{np.linalg.norm(vector):8.2f}")
            else:
                readout.set(hall_status if key == "Hall" else "Waiting for readings")
            label.set(caption)
            if key in ("Hall", "Hall2"):
                bar_buttons[key].set(f"{name} scale…")
                hall_readings.append((name, vector if samples else None))
            draw_traces(plot, samples, now, offset, resolution, limits[key], dpi_scale)
        draw_hall_bars(bars, hall_readings, bar_limits, dpi_scale)
        window.after(50, refresh_traces)

    def close():
        stop.set()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    reader = Thread(target=read_serial, daemon=True)
    reader.start()
    window.after(0, refresh)
    window.after(0, refresh_traces)
    try:
        window.mainloop()
    finally:
        stop.set()
        reader.join()


if __name__ == "__main__":
    main()
