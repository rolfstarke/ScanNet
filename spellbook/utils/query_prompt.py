import os


WIDTH = 480
HEIGHT = 116


def window_position(viewer_rect, width=WIDTH, height=HEIGHT):
    if not viewer_rect:
        return 100, 100
    x, y, w, h = viewer_rect
    return int(x + (w - width) / 2), int(y + (h - height) / 2)


def run(result_out, parent_pid, mode="search", viewer_rect=None):
    import tkinter as tk

    root = tk.Tk()
    root.title("ScanNet Query")
    root.resizable(False, False)
    x, y = window_position(viewer_rect)
    root.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")
    root.attributes("-topmost", True)

    label = "Image path" if mode == "image" else "Class"
    tk.Label(root, text=label, anchor="w").pack(fill="x", padx=14, pady=(14, 4))
    value = tk.StringVar()
    entry = tk.Entry(root, textvariable=value, font=("Sans", 14))
    entry.pack(fill="x", padx=14, pady=(0, 14))
    sent = False

    def finish(answer):
        nonlocal sent
        if sent:
            return
        sent = True
        try:
            result_out.send(answer)
        except (BrokenPipeError, EOFError, OSError):
            pass
        try:
            result_out.close()
        except OSError:
            pass
        root.destroy()

    def submit(_event=None):
        answer = value.get().strip()
        if answer:
            finish(answer)
        return "break"

    def cancel(_event=None):
        finish(None)
        return "break"

    def focus():
        root.lift()
        root.focus_force()
        entry.focus_force()
        entry.icursor("end")
        root.grab_set()

    def check_parent():
        try:
            os.kill(parent_pid, 0)
        except OSError:
            finish(None)
            return
        root.after(500, check_parent)

    root.bind("<Return>", submit)
    root.bind("<KP_Enter>", submit)
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.after(50, focus)
    root.after(500, check_parent)
    root.mainloop()
