"""
Matplotlib plotting utility for both interactive (VNC/laptop) and headless (pipeline) use.
Automatically selects backend and safely handles plt.show().

Created on Oct 9, 2025
@author: Athira
"""

import os
import sys
import threading
import matplotlib
import matplotlib.pyplot as plt
from queue import Queue
import psutil


# ---------- DISPLAY DETECTION ----------

def has_display():
    """
    Detect whether a display (GUI session) is available.
    Works for macOS, VNC, and Linux X11/Wayland.
    """
    env = os.environ

    # Basic X11 or Wayland sessions
    if (
        "DISPLAY" in env
        or "WAYLAND_DISPLAY" in env
        or env.get("XDG_SESSION_TYPE") in ("x11", "wayland")
    ):
        return True

    # macOS native GUI (no DISPLAY)
    if sys.platform == "darwin":
        if "SSH_CONNECTION" not in env:
            # Detect WindowServer (indicates GUI available)
            output = os.popen("ps -A").read()
            return "WindowServer" in output
    return False


# ---------- SAFE BACKEND SELECTION ----------

def safe_backend():
    """
    Choose an interactive backend if display exists,
    otherwise use Agg (headless).
    """
    if has_display():
        try:
            matplotlib.use("QtAgg", force=True)
            print("[matplotlib_plot] Using QtAgg (interactive).")
        except Exception:
            try:
                matplotlib.use("TkAgg", force=True)
                print("[matplotlib_plot] Using TkAgg (fallback interactive).")
            except Exception:
                matplotlib.use("Agg", force=True)
                print("[matplotlib_plot] Using Agg (no GUI).")
    else:
        matplotlib.use("Agg", force=True)
        print("[matplotlib_plot] Using Agg (headless).")


# ---------- THREAD-SAFE SHOW ----------

def show_in_main_thread(fig):
    """
    Safely display a Matplotlib figure even if called from a non-main thread.
    Spawns a temporary helper thread if needed.
    """
    if not has_display():
        print("[matplotlib_plot] No display available; skipping interactive window.")
        return

    if threading.current_thread() is threading.main_thread():
        plt.show(block=True)
        print("[matplotlib_plot] Displayed plot in main thread.")
    else:
        q = Queue()

        def _show():
            plt.show(block=True)
            q.put(None)

        t = threading.Thread(target=_show, daemon=True)
        t.start()
        q.get()
        print("[matplotlib_plot] Displayed plot via helper thread.")


# ---------- MAIN ENTRY ----------

def mpl_plot(fig=None, show=True, save=True, filename="quicklook_plot.png"):
    """
    Display or save a Matplotlib figure safely.
    """
    safe_backend()

    fig = fig or plt.gcf()
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)

    if save:
        fig.savefig(filepath, bbox_inches="tight", dpi=150)
        print(f"[matplotlib_plot] Saved figure to {filepath}")

    if show:
        try:
            show_in_main_thread(fig)
        except Exception as e:
            print(f"[matplotlib_plot] Could not display plot: {e}")

    plt.close(fig)
    print("[matplotlib_plot] Cleared figure after display/save.")


# ---------- CLEANUP / UTILITIES ----------

def mpl_clear():
    """Clear the current figure and axes."""
    plt.clf()
    plt.close("all")
    print("[matplotlib_plot] Cleared all active figures.")


def check_running_process(process=None):
    """
    Check if a process with a given name substring is running.
    Returns True if found.
    """
    for proc in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            for command in proc.cmdline():
                if process and process in command:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def mpl_save(fig=None, filename="plot.png"):
    """Save a Matplotlib figure to a static file."""
    fig = fig or plt.gcf()
    os.makedirs(os.path.dirname(filename) or "plots", exist_ok=True)
    filepath = filename if os.path.isabs(filename) else os.path.join("plots", filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"[matplotlib_plot] Saved figure to {filepath}")
