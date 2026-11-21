"""Money helpers. Amounts are always int minor units; FX rates are decimal strings."""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_HALF_EVEN, ROUND_HALF_UP

CURRENCY_EXPONENTS: dict[str, int] = {
    "USD": 2, "EUR": 2, "GBP": 2, "CHF": 2, "SEK": 2, "JPY": 0,
}

ROUNDING_MODES = {"ROUND_HALF_EVEN": ROUND_HALF_EVEN, "ROUND_HALF_UP": ROUND_HALF_UP}


def convert_minor(amount_minor: int, src_exp: int, rate: str, dst_exp: int,
                  rounding: str = "ROUND_HALF_EVEN") -> int:
    """Convert integer minor units of a source currency into minor units of a target currency.

    value_target = amount_minor / 10**src_exp * rate, expressed in target minor units and
    rounded once with the declared rounding mode.
    """
    if not isinstance(amount_minor, int) or isinstance(amount_minor, bool):
        raise TypeError("amount_minor must be int")
    if not isinstance(rate, str):
        raise TypeError("rate must be a decimal string")
    with localcontext() as ctx:
        ctx.prec = 60
        q = Decimal(amount_minor) * Decimal(rate) * (Decimal(10) ** (dst_exp - src_exp))
        return int(q.quantize(Decimal(1), rounding=ROUNDING_MODES[rounding]))
