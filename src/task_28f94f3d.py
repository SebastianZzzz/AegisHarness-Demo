import math
import datetime
import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

# Core Exceptions
class CalculatorError(Exception):
    pass

# Data Models
class MemoryRegister:
    def __init__(self, value: float = 0.0, is_empty: bool = True):
        self.value = value
        self.is_empty = is_empty

class HistoryEntry:
    def __init__(self, expression: str, result: float, timestamp: datetime.datetime):
        self.expression = expression
        self.result = result
        self.timestamp = timestamp

class UserConfig:
    def __init__(self,
                 angle_mode: str = "radians",
                 decimal_precision: int = 6,
                 theme: str = "system",
                 layout_preset: str = "",
                 history_enabled: bool = True):
        self.angle_mode = angle_mode
        self.decimal_precision = decimal_precision
        self.theme = theme
        self.layout_preset = layout_preset
        self.history_enabled = history_enabled

# Function Registry (safe wrappers around math)
class FunctionRegistry:
    def __init__(self):
        self.angle_mode = "radians"  # or "degrees"

    def set_angle_mode(self, mode: str):
        if mode not in ("radians", "degrees"):
            raise CalculatorError("Invalid angle mode.")
        self.angle_mode = mode

    # Trigonometric helpers with angle mode
    def _to_radians(self, x: float) -> float:
        return math.radians(x) if self.angle_mode == "degrees" else x

    def _to_degrees(self, x: float) -> float:
        return math.degrees(x) if self.angle_mode == "degrees" else x

    def sin(self, x: float) -> float:
        return math.sin(self._to_radians(x))

    def cos(self, x: float) -> float:
        return math.cos(self._to_radians(x))

    def tan(self, x: float) -> float:
        return math.tan(self._to_radians(x))

    def asin(self, x: float) -> float:
        # domain check is inherent in math.asin
        val = math.asin(x)
        return self._to_degrees(val)

    def acos(self, x: float) -> float:
        val = math.acos(x)
        return self._to_degrees(val)

    def atan(self, x: float) -> float:
        val = math.atan(x)
        return self._to_degrees(val)

    def sqrt(self, x: float) -> float:
        if x < 0:
            raise CalculatorError("Domain error: sqrt of negative number.")
        return math.sqrt(x)

    def log(self, x: float) -> float:
        # base 10
        if x <= 0:
            raise CalculatorError("Domain error: log must have positive input.")
        return math.log10(x)

    def ln(self, x: float) -> float:
        if x <= 0:
            raise CalculatorError("Domain error: ln must have positive input.")
        return math.log(x)

    def exp(self, x: float) -> float:
        return math.exp(x)

    def absf(self, x: float) -> float:
        return abs(x)

    def floor(self, x: float) -> float:
        return math.floor(x)

    def ceil(self, x: float) -> float:
        return math.ceil(x)

    def factorial(self, x: float) -> float:
        # Domain: non-negative integer
        if x < 0 or not float(x).is_integer():
            raise CalculatorError("Domain error: factorial input must be a non-negative integer.")
        n = int(x)
        if n > 170:
            # Python's factorial overflows beyond 170! for float display
            raise CalculatorError("Domain error: factorial result too large.")
        return math.factorial(n)

    def cbrt(self, x: float) -> float:
        # real cube root
        if x >= 0:
            return x ** (1.0/3.0)
        else:
            return -((-x) ** (1.0/3.0))

    # General wrapper
    def call_function(self, name: str, arg: float) -> float:
        fname = name.lower()
        if fname == "sin": return self.sin(arg)
        if fname == "cos": return self.cos(arg)
        if fname == "tan": return self.tan(arg)
        if fname == "asin": return self.asin(arg)
        if fname == "acos": return self.acos(arg)
        if fname == "atan": return self.atan(arg)
        if fname == "sqrt": return self.sqrt(arg)
        if fname == "log": return self.log(arg)
        if fname == "ln": return self.ln(arg)
        if fname == "exp": return self.exp(arg)
        if fname == "abs": return self.absf(arg)
        if fname == "floor": return self.floor(arg)
        if fname == "ceil": return self.ceil(arg)
        if fname == "factorial" or fname == "fact":
            return self.factorial(arg)
        if fname == "cbrt": return self.cbrt(arg)
        raise CalculatorError(f"Unsupported function: {name}")

    def is_function_supported(self, name: str) -> bool:
        return name.lower() in {
            "sin","cos","tan","asin","acos","atan",
            "sqrt","log","ln","exp","abs","floor","ceil",
            "factorial","fact","cbrt"
        }

# Parser and Evaluator
class ExpressionParser:
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
        self.functions = self._init_functions()

    def _init_functions(self) -> set:
        return {
            "sin","cos","tan","asin","acos","atan",
            "sqrt","log","ln","exp","abs","floor","ceil",
            "factorial","fact","cbrt"
        }

    # Token types
    # NUMBER, CONST, FUNC, OP, LPAREN, RPAREN
    class Token:
        def __init__(self, ttype: str, value: Any = None):
            self.type = ttype
            self.value = value
        def __repr__(self):
            return f"Token({self.type}, {self.value})"

    def tokenize(self, s: str) -> List['ExpressionParser.Token']:
        tokens: List[ExpressionParser.Token] = []
        i = 0
        n = len(s)
        last_type = None  # type: Optional[str]
        s = s.strip()
        while i < n:
            ch = s[i]
            if ch.isspace():
                i += 1
                continue
            # Numbers
            if ch.isdigit() or ch == '.':
                start = i
                i += 1
                while i < n and (s[i].isdigit() or s[i] == '_'):
                    i += 1
                if i < n and s[i] == '.':
                    i += 1
                    while i < n and (s[i].isdigit() or s[i] == '_'):
                        i += 1
                if i < n and (s[i] in 'eE'):
                    i += 1
                    if i < n and s[i] in '+-':
                        i += 1
                    while i < n and s[i].isdigit():
                        i += 1
                num_str = s[start:i].replace('_','')
                try:
                    val = float(num_str)
                except ValueError:
                    raise CalculatorError("Invalid numeric value.")
                tokens.append(self.Token("NUMBER", val))
                last_type = "NUMBER"
                continue
            # Identifiers: functions and constants
            if ch.isalpha():
                start = i
                i += 1
                while i < n and (s[i].isalnum() or s[i] == '_'):
                    i += 1
                name = s[start:i].lower()
                if name in ("pi", "e"):
                    tokens.append(self.Token("CONST", name))
                elif self.registry.is_function_supported(name) or name in self.functions:
                    tokens.append(self.Token("FUNC", name))
                else:
                    raise CalculatorError(f"Unknown identifier: {name}")
                last_type = "FUNC" if name in self.functions or self.registry.is_function_supported(name) else "CONST"
                continue
            # Parentheses
            if ch == '(':
                tokens.append(self.Token("LPAREN", ch))
                i += 1
                last_type = "LPAREN"
                continue
            if ch == ')':
                tokens.append(self.Token("RPAREN", ch))
                i += 1
                last_type = "RPAREN"
                continue
            # Operators (including multi-char **)
            if ch == '*' and i + 1 < n and s[i+1] == '*':
                op = '**'
                i += 2
            else:
                op = ch
                i += 1
            if op == '-' and (not tokens or tokens[-1].type in ("OP","LPAREN")):
                # Unary minus
                tokens.append(self.Token("OP", "u-"))
            elif op in ('+','-','*','/','%','^','**'):
                # Normalize to single token '^' or '**'
                if op == '**':
                    op = '**'
                tokens.append(self.Token("OP", op))
            else:
                raise CalculatorError(f"Invalid operator: {op}")
            last_type = "OP"
        return tokens

    def _op_precedence(self, op: str) -> int:
        if op == 'u-':
            return 3  # unary minus lower than exponent
        if op in ('**', '^'):
            return 4  # exponent
        if op in ('*','/','%'):
            return 2
        if op in ('+','-'):
            return 1
        return 0

    def _op_right_assoc(self, op: str) -> bool:
        return op in ('**','^','u-')

    def parse_infix_to_rpn(self, tokens: List['ExpressionParser.Token']) -> List['ExpressionParser.Token']:
        output: List['ExpressionParser.Token'] = []
        stack: List['ExpressionParser.Token'] = []

        for tok in tokens:
            if tok.type in ("NUMBER","CONST"):
                output.append(tok)
            elif tok.type == "FUNC":
                stack.append(tok)
            elif tok.type == "RPAREN":
                # pop until LPAREN
                while stack and stack[-1].type != "LPAREN":
                    output.append(stack.pop())
                if not stack:
                    raise CalculatorError("Mismatched parentheses.")
                stack.pop()  # remove LPAREN
                # If top is function, pop to output
                if stack and stack[-1].type == "FUNC":
                    output.append(stack.pop())
            elif tok.type == "LPAREN":
                stack.append(tok)
            elif tok.type == "OP":
                op1 = tok.value
                prec1 = self._op_precedence(op1)
                right_assoc = self._op_right_assoc(op1)
                while stack:
                    top = stack[-1]
                    if top.type == "OP":
                        prec2 = self._op_precedence(top.value)
                        if (prec2 > prec1) or (prec2 == prec1 and not right_assoc):
                            output.append(stack.pop())
                            continue
                    if top.type == "FUNC":
                        output.append(stack.pop())
                        continue
                    break
                stack.append(tok)
            else:
                raise CalculatorError("Invalid token in expression.")
        # Drain stack
        while stack:
            top = stack.pop()
            if top.type in ("LPAREN","RPAREN"):
                raise CalculatorError("Mismatched parentheses.")
            output.append(top)
        return output

# Calculation Engine
class CalculationEngine:
    def __init__(self, angle_mode: str = "radians", decimal_precision: int = 6):
        self.registry = FunctionRegistry()
        self.registry.set_angle_mode(angle_mode)
        self.precision = decimal_precision
        self.parser = ExpressionParser(self.registry)
        self.last_result: float = 0.0
        self.memory = MemoryRegister()
        self.history: List[HistoryEntry] = []
        self.max_history = 100

    # Public API
    def set_angle_mode(self, mode: str):
        self.registry.set_angle_mode(mode)

    def get_last_result(self) -> float:
        return self.last_result

    def clear(self):
        self.last_result = 0.0

    def evaluate_expression(self, expression_str: str) -> float:
        tokens = self.parser.tokenize(expression_str)
        if not tokens:
            raise CalculatorError("Empty expression.")
        rpn = self.parser.parse_infix_to_rpn(tokens)
        result = self._evaluate_rpn(rpn)
        self._store_history(expression_str, result)
        self.last_result = result
        return result

    def _evaluate_rpn(self, rpn: List['ExpressionParser.Token']) -> float:
        stack: List[float] = []
        for tok in rpn:
            if tok.type == "NUMBER":
                stack.append(float(tok.value))
            elif tok.type == "CONST":
                if tok.value == "pi":
                    stack.append(math.pi)
                elif tok.value == "e":
                    stack.append(math.e)
                else:
                    raise CalculatorError("Unknown constant.")
            elif tok.type == "FUNC":
                if not stack:
                    raise CalculatorError("Insufficient arguments for function.")
                arg = stack.pop()
                if not self.registry.is_function_supported(tok.value):
                    raise CalculatorError(f"Unsupported function: {tok.value}")
                res = self.registry.call_function(tok.value, arg)
                stack.append(res)
            elif tok.type == "OP":
                op = tok.value
                if op == 'u-':
                    if not stack:
                        raise CalculatorError("Insufficient operands for unary minus.")
                    a = stack.pop()
                    stack.append(-a)
                else:
                    if len(stack) < 2:
                        raise CalculatorError("Insufficient operands.")
                    b = stack.pop()
                    a = stack.pop()
                    val = self._apply_binary_op(a, b, op)
                    stack.append(val)
            else:
                raise CalculatorError("Invalid token during evaluation.")
        if len(stack) != 1:
            raise CalculatorError("Invalid expression.")
        return stack[0]

    def _apply_binary_op(self, a: float, b: float, op: str) -> float:
        if op == '+':
            return a + b
        if op == '-':
            return a - b
        if op == '*':
            return a * b
        if op == '/':
            if b == 0:
                raise CalculatorError("Division by zero.")
            return a / b
        if op in ('^','**'):
            try:
                # guard too large exponents
                if abs(b) > 1e4 and abs(a) > 1e2:
                    raise CalculatorError("Result too large.")
                return a ** b
            except (ValueError, OverflowError) as e:
                raise CalculatorError("Invalid exponentiation.")
        if op == '%':
            if b == 0:
                raise CalculatorError("Division by zero in modulo.")
            return a % b
        raise CalculatorError(f"Unknown operator: {op}")

    def _store_history(self, expression: str, result: float):
        if not isinstance(result, float) and not isinstance(result, int):
            return
        # Timestamp
        ts = datetime.datetime.now()
        entry = HistoryEntry(expression, result, ts)
        self.history.append(entry)
        # cap history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    # Memory operations
    def memory_add(self, value: float):
        if self.memory.is_empty:
            self.memory.value = value
            self.memory.is_empty = False
        else:
            self.memory.value += value

    def memory_sub(self, value: float):
        if self.memory.is_empty:
            self.memory.value = -value
            self.memory.is_empty = False
        else:
            self.memory.value -= value

    def memory_recall(self) -> float:
        return 0.0 if self.memory.is_empty else self.memory.value

    def memory_clear(self):
        self.memory.value = 0.0
        self.memory.is_empty = True

# Memory and History stores (lightweight)
class MemoryStore:
    def __init__(self):
        self.value: float = 0.0
        self.is_empty: bool = True

    def add(self, v: float):
        if self.is_empty:
            self.value = v
            self.is_empty = False
        else:
            self.value += v

    def sub(self, v: float):
        if self.is_empty:
            self.value = -v
            self.is_empty = False
        else:
            self.value -= v

    def recall(self) -> float:
        return 0.0 if self.is_empty else self.value

    def clear(self):
        self.value = 0.0
        self.is_empty = True

class HistoryStore:
    def __init__(self, max_entries: int = 100):
        self.entries: List[HistoryEntry] = []
        self.max_entries = max_entries

    def add_entry(self, expression: str, result: float, timestamp: datetime.datetime):
        self.entries.append(HistoryEntry(expression, result, timestamp))
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_recent(self, n: int) -> List[HistoryEntry]:
        return self.entries[-n:]

# Persistence
class ConfigManager:
    def __init__(self, appname: str = "SafeCalc"):
        self.appname = appname
        self.config_path = self._config_path()
        self.data: UserConfig = self.load_config()

    def _config_path(self) -> Path:
        home = Path.home()
        cfg_dir = home / f".{self.appname}"
        cfg_dir.mkdir(exist_ok=True)
        return cfg_dir / "config.json"

    def load_config(self) -> UserConfig:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg = UserConfig(
                    angle_mode=data.get("angle_mode","radians"),
                    decimal_precision=int(data.get("decimal_precision",6)),
                    theme=data.get("theme","system"),
                    layout_preset=data.get("layout_preset",""),
                    history_enabled=bool(data.get("history_enabled",True))
                )
                return cfg
            except Exception:
                pass
        return UserConfig()

    def save_config(self, cfg: UserConfig):
        data = {
            "angle_mode": cfg.angle_mode,
            "decimal_precision": cfg.decimal_precision,
            "theme": cfg.theme,
            "layout_preset": cfg.layout_preset,
            "history_enabled": cfg.history_enabled
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

# UI - Minimal Tkinter Desktop UI
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None  # UI will be disabled if Tkinter unavailable

class SimpleCalculatorUI:
    def __init__(self, engine: CalculationEngine, config: UserConfig):
        self.engine = engine
        self.config = config
        self.root = None  # type: Optional[tk.Tk]
        self.expr_var = None  # type: Optional[tk.StringVar]
        self.result_var = None  # type: Optional[tk.StringVar]
        self.expression: str = ""
        self.memory_cache = MemoryStore()
        self.history_store = HistoryStore()
        self.create_ui()

    def create_ui(self):
        if tk is None:
            return
        self.root = tk.Tk()
        self.root.title("SafeCalc - Scientific Calculator (Demo)")
        # Expression entry
        self.expr_var = tk.StringVar(value=self.expression)
        expr_entry = tk.Entry(self.root, textvariable=self.expr_var, font=("Consolas", 16), width=40)
        expr_entry.grid(row=0, column=0, columnspan=6, padx=8, pady=8, sticky="we")

        # Result display
        self.result_var = tk.StringVar(value="")
        result_label = tk.Label(self.root, textvariable=self.result_var, font=("Consolas", 16), anchor="e", bg="white", relief="sunken", width=40)
        result_label.grid(row=1, column=0, columnspan=6, padx=8, pady=4, sticky="we")

        # Angle mode
        mode_frame = tk.LabelFrame(self.root, text="Angle Mode")
        mode_frame.grid(row=2, column=0, columnspan=6, padx=8, pady=4, sticky="we")
        self.angle_mode = tk.StringVar(value=self.config.angle_mode)
        rb_rad = ttk.Radiobutton(mode_frame, text="Radians", value="radians", variable=self.angle_mode, command=self._mode_changed)
        rb_deg = ttk.Radiobutton(mode_frame, text="Degrees", value="degrees", variable=self.angle_mode, command=self._mode_changed)
        rb_rad.pack(side="left", padx=6, pady=6)
        rb_deg.pack(side="left", padx=6, pady=6)

        # Keypad - simple
        keypad_frame = tk.Frame(self.root)
        keypad_frame.grid(row=3, column=0, columnspan=6, padx=8, pady=4)

        keys = [
            ['7','8','9','/','('],
            ['4','5','6','*',')'],
            ['1','2','3','-','pi'],
            ['0','.','^','+','e'],
        ]
        for r, row in enumerate(keys):
            for c, key in enumerate(row):
                b = tk.Button(keypad_frame, text=key, width=6, height=2,
                              command=lambda k=key: self._button_press(k))
                b.grid(row=r, column=c, padx=2, pady=2)

        # Function keys row
        func_frame = tk.Frame(self.root)
        func_frame.grid(row=4, column=0, columnspan=6, padx=8, pady=4)
        func_keys = ['sin','cos','tan','sqrt','log','ln','abs','fact','M+','M-','MR','MC','C','CE','=', '⌫']
        for i, key in enumerate(func_keys):
            b = tk.Button(func_frame, text=key, width=6, height=2,
                          command=lambda k=key: self._button_press(k))
            b.grid(row=0, column=i, padx=2, pady=2)

        # Status bar
        status = tk.Label(self.root, text="SafeEval: No dynamic code; uses safe parser", anchor="w")
        status.grid(row=5, column=0, columnspan=6, sticky="we", padx=8, pady=4)

        # Keyboard support
        self.root.bind("<Return>", lambda e: self._evaluate())
        self.root.bind("<BackSpace>", lambda e: self._backspace())

    def _mode_changed(self):
        mode = self.angle_mode.get()
        self.engine.set_angle_mode(mode)
        self._refresh_result()

    def _button_press(self, label: str):
        # Clear current entry
        if label == "CE":
            self.expression = ""
            self._update_expr_display()
            return
        # Clear all / reset
        if label == "C":
            self.expression = ""
            self.result_var.set("")
            self._update_expr_display()
            return
        if label == "=":
            self._evaluate()
            return
        if label == "⌫":
            self._backspace()
            return
        if label in ("pi","e"):
            self.expression += label
            self._update_expr_display()
            return
        if label in ("sin","cos","tan","sqrt","log","ln","abs","fact","cbrt"):
            # append function call with opening paren
            self.expression += label + "("
            self._update_expr_display()
            return
        if label in ("M+","M-","MR","MC"):
            if label == "MR":
                val = self.engine.memory_recall()
                self.expression += str(val)
            elif label == "MC":
                self.engine.memory_clear()
            elif label == "M+":
                self.engine.memory_add(self._parse_last_result())
            elif label == "M-":
                self.engine.memory_sub(self._parse_last_result())
            self._update_expr_display()
            return
        # Default: append (numbers, operators, etc.)
        self.expression += label
        self._update_expr_display()

    def _parse_last_result(self) -> float:
        try:
            val = float(self.result_var.get())
            return val
        except Exception:
            return self.engine.get_last_result()

    def _backspace(self):
        self.expression = self.expression[:-1]
        self._update_expr_display()

    def _update_expr_display(self):
        if self.expr_var is not None:
            self.expr_var.set(self.expression)

    def _refresh_result(self):
        # Try to show best guess
        try:
            if self.expression.strip():
                val = self.engine.evaluate_expression(self.expression)
                self.result_var.set(self._format_number(val))
            else:
                self.result_var.set("")
        except Exception as e:
            self.result_var.set("")

    def _evaluate(self):
        if not self.expression.strip():
            return
        try:
            result = self.engine.evaluate_expression(self.expression)
            self.result_var.set(self._format_number(result))
            self.expression = str(result)
            self._update_expr_display()
        except CalculatorError as e:
            self.result_var.set(f"Error: {str(e)}")
        except Exception as e:
            self.result_var.set("Error")

    def _format_number(self, val: float) -> str:
        precision = self.engine.precision if hasattr(self.engine, "precision") else 6
        fmt = f"{{:.{precision}f}}"
        try:
            s = fmt.format(val)
            # Trim trailing zeros
            s = s.rstrip('0').rstrip('.')
            if s == "-0":
                s = "0"
            return s
        except Exception:
            return str(val)

    def run(self):
        if self.root is None:
            self.create_ui()
        if self.root is not None:
            self.root.mainloop()

# Public API to run the app (production entrypoints)
def load_config() -> UserConfig:
    cfg_mgr = ConfigManager()
    return cfg_mgr.data

def save_config(cfg: UserConfig):
    cfg_mgr = ConfigManager()
    cfg_mgr.save_config(cfg)

# Simple usage example (CLI-friendly)
def main():
    engine = CalculationEngine(angle_mode="radians", decimal_precision=6)

    # Example expressions to test
    tests = [
        "2+2",
        "3*4+5",
        "sin(0)",
        "cos(0)",
        "tan(45)",
        "pi",
        "e",
        "sqrt(16)",
        "log(1000)",
        "ln(2.7182818285)",
        "3^2",
        "2**3",
        "-2^2",  # should be -(2^2) = -4
        "fact(5)",
    ]
    for t in tests:
        try:
            r = engine.evaluate_expression(t)
            print(f"{t} = {r:.6g}")
        except CalculatorError as ce:
            print(f"{t} -> Error: {ce}")

if __name__ == "__main__":
    # If Tkinter is available, launch a minimal UI demo; otherwise run CLI demo
    if tk is not None:
        cfg = load_config()
        eng = CalculationEngine(angle_mode=cfg.angle_mode, decimal_precision=cfg.decimal_precision)
        ui = SimpleCalculatorUI(eng, cfg)
        ui.run()
    else:
        main()