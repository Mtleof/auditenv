"""Safe decimal calculator: arithmetic only, no eval/exec, no names, no attribute access.

Grammar:
    expr    := term (("+" | "-") term)*
    term    := unary (("*" | "/") unary)*
    unary   := ("+" | "-") unary | primary
    primary := NUMBER | "(" expr ")" | "round" "(" expr "," INTEGER ")"
round() uses ROUND_HALF_EVEN, places in [0, 12].
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_EVEN

MAX_LEN = 256
MAX_TOKENS = 128
MAX_DEPTH = 32
MAX_DIGITS = 40
MAX_ABS = Decimal(10) ** 30

_TOKEN_RE = re.compile(r"\s*(?:(\d+(?:\.\d+)?|\.\d+)|(round)|([+\-*/(),]))")


class CalculatorError(Exception):
    """Raised with a short, fixed error code (never echoes user input)."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _tokenize(expr: str) -> list[tuple[str, str]]:
    pos, out = 0, []
    expr = expr.rstrip()
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m or m.end() == pos:
            raise CalculatorError("syntax")
        num, name, op = m.groups()
        if num is not None:
            if len(num.replace(".", "")) > MAX_DIGITS:
                raise CalculatorError("too_long")
            out.append(("num", num))
        elif name is not None:
            out.append(("name", name))
        else:
            out.append(("op", op))
        pos = m.end()
        if len(out) > MAX_TOKENS:
            raise CalculatorError("too_long")
    return out


class _Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0
        self.depth = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, kind=None, val=None):
        k, v = self.peek()
        if k is None or (kind and k != kind) or (val and v != val):
            raise CalculatorError("syntax")
        self.i += 1
        return v

    def check(self, x: Decimal) -> Decimal:
        if not x.is_finite() or abs(x) > MAX_ABS:
            raise CalculatorError("overflow")
        return x

    def expr(self) -> Decimal:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalculatorError("too_deep")
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()
            r = self.term()
            v = self.check(v + r if op == "+" else v - r)
        self.depth -= 1
        return v

    def term(self) -> Decimal:
        v = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()
            r = self.unary()
            if op == "*":
                v = self.check(v * r)
            else:
                if r == 0:
                    raise CalculatorError("division_by_zero")
                v = self.check(v / r)
        return v

    def unary(self) -> Decimal:
        if self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise CalculatorError("too_deep")
            v = self.unary()
            self.depth -= 1
            return v if op == "+" else -v
        return self.primary()

    def primary(self) -> Decimal:
        k, v = self.peek()
        if k == "num":
            self.take()
            return Decimal(v)
        if k == "op" and v == "(":
            self.take()
            r = self.expr()
            self.take("op", ")")
            return r
        if k == "name" and v == "round":
            self.take()
            self.take("op", "(")
            x = self.expr()
            self.take("op", ",")
            places = self.take("num")
            if not places.isdigit() or int(places) > 12:
                raise CalculatorError("syntax")
            self.take("op", ")")
            return self.check(x.quantize(Decimal(1).scaleb(-int(places)), rounding=ROUND_HALF_EVEN))
        raise CalculatorError("syntax")


def evaluate(expression) -> str:
    """Evaluate an arithmetic expression and return the result as a decimal string."""
    if not isinstance(expression, str):
        raise CalculatorError("syntax")
    if len(expression) > MAX_LEN:
        raise CalculatorError("too_long")
    tokens = _tokenize(expression)
    if not tokens:
        raise CalculatorError("syntax")
    with localcontext() as ctx:
        ctx.prec = 40
        try:
            p = _Parser(tokens)
            v = p.expr()
            if p.i != len(tokens):
                raise CalculatorError("syntax")
            v = p.check(v)
        except (InvalidOperation, ArithmeticError):
            raise CalculatorError("overflow") from None
        if v == v.to_integral_value():
            return str(v.quantize(Decimal(1)))
        return format(v.normalize(), "f")
