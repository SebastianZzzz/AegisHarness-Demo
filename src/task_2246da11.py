import math
from typing import List, Tuple, Optional, Dict

# Exceptions
class EvaluateError(Exception):
    pass

# Public API data formats
# EvaluationResult: {'ok': bool, 'value': float|None, 'error': str|None, 'metadata': dict|None}

# Global defaults
_DEFAULT_CONTEXT = {
    'angle_mode': 'radians',  # 'radians' or 'degrees'
    'precision': 12,          # digits after decimal for final result
    'max_depth': 100,           # recursion depth guard
    'max_tokens': 4096,          # max tokens for an expression
    'allow_nan': False,          # whether to allow NaN in outputs
    'allow_infinite': False,     # whether to allow infinite results
}

# Registry and constants
def _build_default_functions():
    # Each function receives (args: List[float], ctx: dict) and returns float
    def sin_fn(a, ctx):
        v = a
        if ctx['angle_mode'] == 'degrees':
            v = math.radians(v)
        return math.sin(v)

    def cos_fn(a, ctx):
        v = a
        if ctx['angle_mode'] == 'degrees':
            v = math.radians(v)
        return math.cos(v)

    def tan_fn(a, ctx):
        v = a
        if ctx['angle_mode'] == 'degrees':
            v = math.radians(v)
        return math.tan(v)

    def asin_fn(a, ctx):
        v = a
        res = math.asin(v)
        if ctx['angle_mode'] == 'degrees':
            res = math.degrees(res)
        return res

    def acos_fn(a, ctx):
        v = a
        res = math.acos(v)
        if ctx['angle_mode'] == 'degrees':
            res = math.degrees(res)
        return res

    def atan_fn(a, ctx):
        v = a
        res = math.atan(v)
        if ctx['angle_mode'] == 'degrees':
            res = math.degrees(res)
        return res

    def log_fn(*args, ctx):
        if len(args) == 1:
            x = args[0]
            if x <= 0:
                raise ValueError("log domain error")
            return math.log10(x)
        elif len(args) == 2:
            x, base = args
            if x <= 0 or base <= 0 or base == 1:
                raise ValueError("log domain error")
            return math.log(x, base)
        else:
            raise ValueError("log expects 1 or 2 arguments")

    def ln_fn(a, ctx):
        if a <= 0:
            raise ValueError("ln domain error")
        return math.log(a)

    def sqrt_fn(a, ctx):
        if a < 0:
            raise ValueError("sqrt domain error")
        return math.sqrt(a)

    def cbrt_fn(a, ctx):
        # cube root handles negative numbers
        if a >= 0:
            return a ** (1.0/3.0)
        else:
            return -((-a) ** (1.0/3.0))

    def abs_fn(a, ctx):
        return abs(a)

    def floor_fn(a, ctx):
        return math.floor(a)

    def ceil_fn(a, ctx):
        return math.ceil(a)

    def round_fn(a, n=None, ctx=None):
        if n is None:
            return round(a)
        else:
            return round(a, int(n))

    def exp_fn(a, ctx):
        return math.exp(a)

    def erf_fn(a, ctx):
        return math.erf(a)

    def gamma_fn(a, ctx):
        return math.gamma(a)

    def factorial_fn(a, ctx):
        if a < 0 or int(a) != a:
            raise ValueError("factorial domain error")
        return math.factorial(int(a))

    def ncr_fn(n, r, ctx):
        if int(n) != n or int(r) != r:
            raise ValueError("nCr requires integer arguments")
        n_i, r_i = int(n), int(r)
        if n_i < 0 or r_i < 0 or r_i > n_i:
            raise ValueError("nCr domain error")
        return math.comb(n_i, r_i)

    def max_fn(args, ctx):
        if len(args) == 0:
            raise ValueError("max requires at least one argument")
        return max(args)

    def min_fn(args, ctx):
        if len(args) == 0:
            raise ValueError("min requires at least one argument")
        return min(args)

    def pow_fn(a, b, ctx):
        # handle negative base with non-integer exponent as domain error
        if a < 0 and abs(b - round(b)) > 1e-12:
            raise ValueError("complex result")
        return a ** b

    def sign_fn(a, ctx):
        if a > 0:
            return 1
        if a < 0:
            return -1
        return 0

    def sqrt2_fn(a, ctx):
        return sqrt_fn(a, ctx)

    return {
        'sin': lambda args, ctx: sin_fn(args[0], ctx),
        'cos': lambda args, ctx: cos_fn(args[0], ctx),
        'tan': lambda args, ctx: tan_fn(args[0], ctx),
        'asin': lambda args, ctx: asin_fn(args[0], ctx),
        'acos': lambda args, ctx: acos_fn(args[0], ctx),
        'atan': lambda args, ctx: atan_fn(args[0], ctx),
        'log': lambda args, ctx: log_fn(*args, ctx=ctx),
        'ln': lambda args, ctx: ln_fn(args[0], ctx),
        'sqrt': lambda args, ctx: sqrt_fn(args[0], ctx),
        'cbrt': lambda args, ctx: cbrt_fn(args[0], ctx),
        'abs': lambda args, ctx: abs_fn(args[0], ctx),
        'floor': lambda args, ctx: floor_fn(args[0], ctx),
        'ceil': lambda args, ctx: ceil_fn(args[0], ctx),
        'round': lambda args, ctx: round_fn(args[0], args[1] if len(args) > 1 else None, ctx=ctx),
        'exp': lambda args, ctx: exp_fn(args[0], ctx),
        'erf': lambda args, ctx: erf_fn(args[0], ctx),
        'gamma': lambda args, ctx: gamma_fn(args[0], ctx),
        'factorial': lambda args, ctx: factorial_fn(args[0], ctx),
        'ncr': lambda args, ctx: ncr_fn(args[0], args[1], ctx),
        'max': lambda args, ctx: max_fn(args, ctx),
        'min': lambda args, ctx: min_fn(args, ctx),
        'pow': lambda args, ctx: pow_fn(args[0], args[1], ctx),
        'sign': lambda args, ctx: sign_fn(args[0], ctx),
        'sqrt2': lambda args, ctx: sqrt2_fn(args[0], ctx),
        'sqrt_': lambda args, ctx: sqrt_fn(args[0], ctx),  # alias if needed
    }

def _build_default_constants():
    return {
        'pi': math.pi,
        'e': math.e,
    }

_FUNCTION_REGISTRY = _build_default_functions()
_CONSTANT_REGISTERY = _build_default_constants()
# Normalize function keys to lowercase for lookup
_FUNCTION_REGISTRY = {k.lower(): v for k, v in _FUNCTION_REGISTRY.items()}
_CONSTANTS = {k.lower(): v for k, v in _CONSTANT_REGISTERY.items()}

# Token types
class Token:
    def __init__(self, ttype: str, value):
        self.type = ttype  # 'NUMBER', 'IDENT', 'LPAREN', 'RPAREN', 'COMMA', 'OP', 'EOF'
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

# Lexer
class Lexer:
    def __init__(self, text: str, max_tokens: int):
        self.text = text
        self.pos = 0
        self.length = len(text)
        self.max_tokens = max_tokens
        self.tokens: List[Token] = []
        self._tokenize()

    def _peek(self, offset: int = 0) -> Optional[str]:
        idx = self.pos + offset
        if idx < self.length:
            return self.text[idx]
        return None

    def _advance(self) -> Optional[str]:
        ch = self._peek()
        if ch is not None:
            self.pos += 1
        return ch

    def _tokenize(self):
        while self.pos < self.length:
            ch = self._peek()
            if ch.isspace():
                self._advance()
                continue

            # Numbers (including leading dot, e.g., .5)
            if (ch.isdigit()) or (ch == '.' and self.pos + 1 < self.length and self._peek(1).isdigit()):
                num = self._read_number()
                self.tokens.append(Token('NUMBER', float(num)))
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue

            # Identifiers (functions/constants)
            if ch.isalpha() or ch == '_':
                ident = self._read_identifier()
                self.tokens.append(Token('IDENT', ident))
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue

            if ch == '+':
                self.tokens.append(Token('OP', '+'))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == '-':
                self.tokens.append(Token('OP', '-'))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == '*':
                self.tokens.append(Token('OP', '*'))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == '/':
                self.tokens.append(Token('OP', '/'))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == '^':
                self.tokens.append(Token('OP', '^'))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == '%':
                self.tokens.append(Token('OP', '%'))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == '(':
                self.tokens.append(Token('LPAREN', '('))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == ')':
                self.tokens.append(Token('RPAREN', ')'))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue
            if ch == ',':
                self.tokens.append(Token('COMMA', ','))
                self.pos += 1
                if len(self.tokens) > self.max_tokens:
                    raise EvaluateError("Expression too long or too many tokens")
                continue

            # Unknown character
            raise EvaluateError(f"Invalid character encountered: {ch}")

        self.tokens.append(Token('EOF', None))
        if len(self.tokens) > self.max_tokens:
            raise EvaluateError("Expression too long or too many tokens")

    def _read_number(self) -> str:
        # Handle leading dot: .5
        if self._peek() == '.':
            s = '0.'
            self.pos += 1
            while self._peek() is not None and self._peek().isdigit():
                s += self._advance()
            # Optional exponent
            if self._peek() in ('e', 'E'):
                s += self._advance()
                if self._peek() in ('+', '-'):
                    s += self._advance()
                if self._peek() is None or not self._peek().isdigit():
                    raise EvaluateError("Invalid numeric literal")
                while self._peek() is not None and self._peek().isdigit():
                    s += self._advance()
            return s

        s = ''
        # integral part
        while self._peek() is not None and self._peek().isdigit():
            s += self._advance()
        # fractional part
        if self._peek() == '.':
            s += self._advance()
            while self._peek() is not None and self._peek().isdigit():
                s += self._advance()
        # exponent
        if self._peek() in ('e', 'E'):
            s += self._advance()
            if self._peek() in ('+', '-'):
                s += self._advance()
            if self._peek() is None or not self._peek().isdigit():
                raise EvaluateError("Invalid numeric literal")
            while self._peek() is not None and self._peek().isdigit():
                s += self._advance()
        if s == '':
            raise EvaluateError("Invalid numeric literal")
        return s

    def _read_identifier(self) -> str:
        s = ''
        while self._peek() is not None and (self._peek().isalnum() or self._peek() == '_'):
            s += self._advance()
        return s

# Parser
class Parser:
    def __init__(self, tokens: List[Token], engine_ctx: dict, max_depth: int):
        self.tokens = tokens
        self.pos = 0
        self.ctx = engine_ctx
        self.max_depth = max_depth

    def _peek(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token('EOF', None)

    def _consume(self) -> Token:
        token = self._peek()
        self.pos += 1
        return token

    def _expect(self, typ: str, value: Optional[str] = None):
        t = self._peek()
        if t.type != typ or (value is not None and t.value != value):
            raise EvaluateError(f"Expected {typ} {value}, got {t.type} {t.value}")
        self._consume()
        return t

    def parse(self) -> float:
        val = self._parse_expression(0)
        if self._peek().type != 'EOF':
            raise EvaluateError("Unexpected token after end of expression")
        return val

    def _parse_expression(self, depth: int) -> float:
        if depth > self.max_depth:
            raise EvaluateError("Expression too deep")
        left = self._parse_term(depth + 1)
        while True:
            t = self._peek()
            if t.type == 'OP' and t.value in ('+', '-'):
                self._consume()
                right = self._parse_term(depth + 1)
                if t.value == '+':
                    left = left + right
                else:
                    left = left - right
            else:
                break
        return left

    def _parse_term(self, depth: int) -> float:
        if depth > self.max_depth:
            raise EvaluateError("Expression too deep")
        left = self._parse_power(depth + 1)
        while True:
            t = self._peek()
            if t.type == 'OP' and t.value in ('*', '/', '%'):
                self._consume()
                right = self._parse_power(depth + 1)
                if t.value == '*':
                    left = left * right
                elif t.value == '/':
                    if right == 0:
                        raise EvaluateError("Division by zero")
                    left = left / right
                else:  # modulo
                    try:
                        left = left % right
                    except Exception:
                        raise EvaluateError("Invalid modulo operation (div by zero or non-integer)")
            else:
                break
        return left

    def _parse_power(self, depth: int) -> float:
        if depth > self.max_depth:
            raise EvaluateError("Expression too deep")
        left = self._parse_unary(depth + 1)
        t = self._peek()
        if t.type == 'OP' and t.value == '^':
            self._consume()
            right = self._parse_power(depth + 1)  # right-associative
            left = left ** right
        return left

    def _parse_unary(self, depth: int) -> float:
        if depth > self.max_depth:
            raise EvaluateError("Expression too deep")
        t = self._peek()
        if t.type == 'OP' and t.value in ('+', '-'):
            self._consume()
            val = self._parse_unary(depth + 1)
            return val if t.value == '+' else -val
        else:
            return self._parse_primary(depth + 1)

    def _parse_primary(self, depth: int) -> float:
        if depth > self.max_depth:
            raise EvaluateError("Expression too deep")
        t = self._peek()
        if t.type == 'NUMBER':
            self._consume()
            return t.value
        if t.type == 'IDENT':
            name = t.value
            self._consume()
            next_t = self._peek()
            if next_t.type == 'LPAREN':
                self._consume()  # consume '('
                args = []
                if self._peek().type == 'RPAREN':
                    self._consume()  # empty args
                else:
                    while True:
                        arg = self._parse_expression(depth + 1)
                        args.append(arg)
                        tok = self._peek()
                        if tok.type == 'COMMA':
                            self._consume()
                            continue
                        elif tok.type == 'RPAREN':
                            self._consume()
                            break
                        else:
                            raise EvaluateError("Expected ',' or ')' in argument list")
                return self._call_function(name, args)
            else:
                # Constant
                key = name.lower()
                if key in _CONSTANTS:
                    return _CONSTANTS[key]
                else:
                    raise EvaluateError(f"Unknown identifier: {name}")
        if t.type == 'LPAREN':
            self._consume()
            val = self._parse_expression(depth + 1)
            if self._peek().type != 'RPAREN':
                raise EvaluateError("Missing closing parenthesis")
            self._consume()
            return val
        raise EvaluateError("Unexpected token")

    def _call_function(self, name: str, args: List[float]) -> float:
        key = name.lower()
        if key not in _FUNCTION_REGISTRY:
            raise EvaluateError(f"Unknown function: {name}")
        fn = _FUNCTION_REGISTRY[key]

        try:
            res = fn(args, self.ctx)
        except EvaluateError:
            raise
        except Exception as e:
            raise EvaluateError(f"Function '{name}' error: {e}")
        if isinstance(res, (int, float)):
            return float(res)
        else:
            raise EvaluateError(f"Function '{name}' did not return a numeric value")

# Engine and public API
class CalculationEngine:
    def __init__(self, context: Optional[Dict] = None):
        self.ctx = dict(_DEFAULT_CONTEXT)
        if context:
            self.ctx.update(context)
        # ensure lowercase maps
        self._precision = self.ctx.get('precision', 12)

    def set_context(self, option: Dict) -> None:
        if not isinstance(option, dict):
            raise ValueError("Context option must be a dict")
        self.ctx.update(option)
        self._precision = self.ctx.get('precision', 12)

    def reset_context(self) -> None:
        self.ctx = dict(_DEFAULT_CONTEXT)
        self._precision = self.ctx.get('precision', 12)

    def evaluate(self, expression: str) -> Dict:
        if not isinstance(expression, str):
            return {'ok': False, 'error': 'Expression must be a string', 'value': None, 'metadata': None}
        try:
            lexer = Lexer(expression, max_tokens=self.ctx.get('max_tokens', 4096))
            tokens = lexer.tokens
            parser = Parser(tokens, self.ctx, self.ctx.get('max_depth', 100))
            value = parser.parse()
            # Final rounding
            precision = self.ctx.get('precision', 12)

            # Enforce finite constraints
            is_finite = math.isfinite(value)
            is_nan = math.isnan(value)

            if is_nan or not is_finite:
                if is_nan:
                    if not self.ctx.get('allow_nan', False):
                        raise EvaluateError("Result is NaN")
                if math.isinf(value) and not self.ctx.get('allow_infinite', False):
                    raise EvaluateError("Result is infinite")

            if is_finite and not is_nan:
                value = round(value, precision)
            return {'ok': True, 'value': value, 'error': None, 'metadata': {'expression': expression}}
        except EvaluateError as e:
            return {'ok': False, 'value': None, 'error': str(e), 'metadata': {'expression': expression}}
        except Exception as e:
            # Unexpected
            return {'ok': False, 'value': None, 'error': 'Internal error: ' + str(e), 'metadata': {'expression': expression}}

# Public API
def evaluateExpression(expression: str, context: Optional[Dict] = None) -> Dict:
    eng = CalculationEngine(context)
    return eng.evaluate(expression)

def getSupportedFunctions() -> List[str]:
    # return sorted unique function names
    names = sorted(set(_FUNCTION_REGISTRY.keys()))
    return names

def getConstants() -> List[str]:
    return sorted(set(_CONSTANTS.keys()))

def setContextOption(option: Dict) -> None:
    global _GLOBAL_ENGINE
    if '_GLOBAL_ENGINE' not in globals():
        _GLOBAL_ENGINE = CalculationEngine()
    _GLOBAL_ENGINE.set_context(option)

def resetContextOption() -> None:
    global _GLOBAL_ENGINE
    if '_GLOBAL_ENGINE' in globals():
        _GLOBAL_ENGINE.reset_context()

# Optional: expose a simple singleton for ease of use
_GLOBAL_ENGINE = CalculationEngine()

# If user uses module-level helper
def evaluate(expression: str) -> Dict:
    return _GLOBAL_ENGINE.evaluate(expression)

# Expose a small API surface at module level
__all__ = [
    'evaluateExpression',
    'getSupportedFunctions',
    'getConstants',
    'setContextOption',
    'resetContextOption',
    'evaluate',
    'EvaluateError'
]