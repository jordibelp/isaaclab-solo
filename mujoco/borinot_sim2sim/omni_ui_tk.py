"""The small ``omni.ui`` subset used by the Borinot play panels, drawn with Tkinter.

Windows, stacks, collapsible/rebuildable frames, labels, buttons, check boxes and float models.
Callbacks run in the main thread when the backend pumps Tk (every rendered frame), so the
play scripts keep their original "UI queues intent, the loop applies it" behaviour.
"""

from __future__ import annotations

import importlib
import math
import tkinter as tk

_root = None
_parents = []  # container stack while "with" blocks build a window
_styles = [{}]
_windows = []
_pending_labels = set()
# Dark palette like omni.ui's default theme: panel styles (light text colours) assume it.
_THEME = {"*Background": "#24272c", "*Foreground": "#dde1e6", "*activeBackground": "#4a515b",
          "*activeForeground": "#ffffff", "*highlightBackground": "#24272c", "*highlightColor": "#5b8dd6",
          "*troughColor": "#15171a", "*selectColor": "#15171a", "*insertBackground": "#dde1e6",
          "*Button.Background": "#3b4048", "*Entry.Background": "#15171a", "*disabledForeground": "#7b818a"}


def load_module():
    return importlib.import_module(__name__)


def _tk_root():
    global _root
    if _root is None:
        _root = tk.Tk()
        _root.withdraw()  # every panel is a Toplevel
        for pattern, value in _THEME.items():
            _root.option_add(pattern, value)
    return _root


def pump():
    if _root is not None:
        try:
            labels = tuple(_pending_labels)
            _pending_labels.clear()
            for label in labels:
                if label.widget.winfo_exists():  # a Frame rebuild may have removed the old label
                    label.widget.configure(text=label._text)
            _root.update()
        except tk.TclError:
            pass


def shutdown():
    global _root
    _pending_labels.clear()
    for window in list(_windows):
        window.destroy()
    if _root is not None:
        try:
            _root.destroy()
        except tk.TclError:
            pass
        _root = None


def _color(value):
    """omni.ui 0xAABBGGRR -> Tk #rrggbb."""
    return f"#{value & 0xFF:02x}{(value >> 8) & 0xFF:02x}{(value >> 16) & 0xFF:02x}"


def _widget_style(kind, style):
    merged = dict(_styles[-1].get(kind, {}))
    merged.update(style or {})
    options = {}
    if "font_size" in merged:
        options["font"] = ("TkDefaultFont", -int(merged["font_size"]))
    if "color" in merged:
        options["fg"] = _color(merged["color"])
    if "background_color" in merged:
        options["bg"] = _color(merged["background_color"])
    return options


class DockPreference:
    RIGHT_TOP = "right_top"
    LEFT_BOTTOM = "left_bottom"


class DockPolicy:
    CURRENT_WINDOW_IS_ACTIVE = "current_window_is_active"
    DO_NOTHING = "do_nothing"


class _Subscription:
    def __init__(self, listeners, fn):
        self._listeners, self._fn = listeners, fn
        listeners.append(fn)

    def unsubscribe(self):
        if self._fn in self._listeners:
            self._listeners.remove(self._fn)


class _Model:
    def __init__(self, value):
        self._value = value
        self._listeners, self._views = [], []

    def subscribe_value_changed_fn(self, fn):
        return _Subscription(self._listeners, fn)

    def add_value_changed_fn(self, fn):
        return self.subscribe_value_changed_fn(fn)

    def set_value(self, value):
        value = self._coerce(value)
        if value == self._value:
            return
        self._value = value
        for view in list(self._views):
            view(value)
        for fn in list(self._listeners):
            fn(self)

    def get_value_as_float(self):
        return float(self._value)

    def get_value_as_bool(self):
        return bool(self._value)

    def get_value_as_int(self):
        return int(self._value)

    def get_value_as_string(self):
        return str(self._value)

    as_float = property(get_value_as_float)
    as_bool = property(get_value_as_bool)


class SimpleFloatModel(_Model):
    def __init__(self, value=0.0, min=None, max=None, **kwargs):  # noqa: A002 - omni.ui names
        super().__init__(float(value))
        self.min, self.max = min, max

    @staticmethod
    def _coerce(value):
        return float(value)


class SimpleBoolModel(_Model):
    def __init__(self, value=False, **kwargs):
        super().__init__(bool(value))

    @staticmethod
    def _coerce(value):
        return bool(value)


class _Widget:
    """Pack a Tk widget into the current container: rows in a VStack, columns in an HStack."""

    def _place(self, widget, width=None, height=None, expand=None):
        container = _parents[-1]
        horizontal = getattr(container, "_horizontal", False)
        if expand is None:
            expand = horizontal and width is None
        if horizontal:
            widget.pack(side=tk.LEFT, fill=tk.BOTH if expand else tk.Y, expand=expand, padx=2)
        else:
            widget.pack(side=tk.TOP, fill=tk.X, pady=1)
        self.widget = widget


class _Container(_Widget):
    horizontal = False

    def __init__(self, spacing=0, height=None, width=None, **kwargs):
        frame = tk.Frame(_parents[-1])
        frame._horizontal = self.horizontal
        self._place(frame, width=width, expand=True if self.horizontal is False else None)
        self.inner = frame

    def __enter__(self):
        _parents.append(self.inner)
        return self

    def __exit__(self, *exc):
        _parents.pop()
        return False


class VStack(_Container):
    horizontal = False


class HStack(_Container):
    horizontal = True


class Frame(_Container):
    """A container whose build callback can replace its contents, e.g. waypoint rows."""

    def __init__(self, build_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._build_fn = build_fn
        if build_fn is not None:
            self.rebuild()

    def rebuild(self):
        for child in self.inner.winfo_children():
            child.destroy()
        if self._build_fn is not None:
            with self:
                self._build_fn()


class CollapsableFrame(_Container):
    """Clickable header with a persistent body; hiding it does not discard edited values."""

    def __init__(self, title="", collapsed=False, **kwargs):
        super().__init__(**kwargs)
        outer, self._title = self.inner, title
        self._header = tk.Button(outer, anchor="w", command=self._toggle)
        self._header.pack(side=tk.TOP, fill=tk.X)
        self.inner = tk.Frame(outer)
        self.collapsed = collapsed

    @property
    def collapsed(self):
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value):
        self._collapsed = bool(value)
        self._header.configure(text=f"{'>' if self._collapsed else 'v'} {self._title}")
        if self._collapsed:
            self.inner.pack_forget()
        else:
            self.inner.pack(side=tk.TOP, fill=tk.X)

    def _toggle(self):
        self.collapsed = not self.collapsed


class ScrollingFrame(_Widget):
    def __init__(self, **kwargs):
        outer = tk.Frame(_parents[-1])
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        bar = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(canvas)
        item = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
        for sequence, step in (("<Button-4>", -1), ("<Button-5>", 1)):
            canvas.bind_all(sequence, lambda e, s=step: canvas.yview_scroll(s, "units"))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.widget = outer

    def __enter__(self):
        _parents.append(self.inner)
        return self

    def __exit__(self, *exc):
        _parents.pop()
        return False


class Label(_Widget):
    def __init__(self, text="", width=None, height=None, word_wrap=False, style=None, **kwargs):
        options = _widget_style("Label", style)
        wrap = 520 if word_wrap else 0
        label = tk.Label(_parents[-1], text=text, anchor="w", justify=tk.LEFT, wraplength=wrap, **options)
        if width:
            label.configure(width=max(1, int(width) // 8))
        self._place(label, width=width)
        self._text = text

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        if self._text != value:
            self._text = value
            _pending_labels.add(self)


class Button(_Widget):
    def __init__(self, text="", clicked_fn=None, width=None, height=None, style=None, **kwargs):
        options = _widget_style("Button", style)
        options.pop("fg", None)  # keep the theme's light button text
        button = tk.Button(_parents[-1], text=text, command=clicked_fn or (lambda: None), **options)
        if width:
            button.configure(width=max(1, int(width) // 9))
        self._place(button, width=width)

    @property
    def text(self):
        return self.widget.cget("text")

    @text.setter
    def text(self, value):
        self.widget.configure(text=value)


class CheckBox(_Widget):
    def __init__(self, model=None, width=None, enabled=True, **kwargs):
        self.model = model or SimpleBoolModel(False)
        variable = tk.BooleanVar(value=self.model.get_value_as_bool())
        box = tk.Checkbutton(_parents[-1], variable=variable,
                             command=lambda: self.model.set_value(variable.get()),
                             state=tk.NORMAL if enabled else tk.DISABLED)
        self.model._views.append(variable.set)
        self._variable = variable
        self._place(box, width=width or 22, expand=False)


class FloatField(_Widget):
    def __init__(self, model=None, width=None, **kwargs):
        self.model = model or SimpleFloatModel(0.0)
        self._text = tk.StringVar(value=self._format(self.model.get_value_as_float()))
        entry = tk.Entry(_parents[-1], textvariable=self._text, width=max(4, int(width or 80) // 9))
        entry.bind("<Return>", self._commit)
        entry.bind("<FocusOut>", self._commit)
        self.model._views.append(lambda value: self._text.set(self._format(value)))
        self._place(entry, width=width, expand=False)

    @staticmethod
    def _format(value):
        return f"{value:.12g}"

    def _commit(self, event=None):
        try:
            value = float(self._text.get())
        except ValueError:
            value = math.nan
        if math.isfinite(value):
            self.model.set_value(value)
        else:
            self._text.set(self._format(self.model.get_value_as_float()))


class FloatSlider(_Widget):
    def __init__(self, model=None, min=0.0, max=1.0, width=None, **kwargs):  # noqa: A002 - omni.ui names
        self.model = model or SimpleFloatModel(0.0)
        resolution = (max - min) / 1000.0 if max > min else 0.001
        scale = tk.Scale(_parents[-1], from_=min, to=max, resolution=resolution, orient=tk.HORIZONTAL,
                         showvalue=False, command=self._moved)
        self._place(scale, width=width)
        self._show(self.model.get_value_as_float())
        self.model._views.append(self._show)

    def _show(self, value):
        self.widget.set(value)
        # Tk invokes Scale.command later, during the event pump. Remember the
        # displayed (rounded/clamped) value so it cannot replace an exact field.
        self._shown = float(self.widget.get())

    def _moved(self, text):
        value = float(text)
        if value != self._shown:
            self._shown = value
            self.model.set_value(value)


class _WindowFrame:
    def __init__(self, window):
        self._window = window

    def set_style(self, style):
        _styles.append(dict(style))
        self._window._style_pushed += 1

    def __enter__(self):
        _parents.append(self._window.body)
        return self

    def __exit__(self, *exc):
        _parents.pop()
        for _ in range(self._window._style_pushed):
            if len(_styles) > 1:
                _styles.pop()
        self._window._style_pushed = 0
        return False


class Window:
    def __init__(self, title="", width=400, height=600, dock_preference=None, visible=True, **kwargs):
        root = _tk_root()
        self._top = tk.Toplevel(root)
        self._top.title(title)
        x = max(0, root.winfo_screenwidth() - int(width) - 20)
        self._top.geometry(f"{int(width)}x{min(int(height), root.winfo_screenheight() - 80)}+{x}+40")
        self._top.protocol("WM_DELETE_WINDOW", self._top.withdraw)  # closing hides; playback continues
        self.body = tk.Frame(self._top)
        self.body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._style_pushed = 0
        self.frame = _WindowFrame(self)
        self.visible = visible
        _windows.append(self)

    def deferred_dock_in(self, *args, **kwargs):
        pass

    def destroy(self):
        if self in _windows:
            _windows.remove(self)
        try:
            self._top.destroy()
        except tk.TclError:
            pass
