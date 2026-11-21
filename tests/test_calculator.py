import pytest

from auditenv.tools.calculator import CalculatorError, evaluate


@pytest.mark.parametrize("expr,val", [
    ("1+2*3", "7"), ("(1+2)*3", "9"), ("10/4", "2.5"), ("-3--2", "-1"), ("+5", "5"),
    ("123456 * 1.084512", "133889.513472"), ("round(10/3, 2)", "3.33"), ("round(2.5, 0)", "2"),
    ("0.1+0.2", "0.3"), (" 2 * ( 3 + 4 ) ", "14"),
])
def test_arithmetic(expr, val):
    assert evaluate(expr) == val


@pytest.mark.parametrize("expr", [
    "__import__('os')", "1;2", "a+1", "2**8", "open('x')", "1+", "()", "", "1 2", "(1", "1e5",
    "lambda: 1", "[1]", "round(1)", "round(1, 99)", "x.y", "1 if 1 else 2", "eval('1')",
])
def test_rejects_non_arithmetic(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_limits():
    with pytest.raises(CalculatorError) as e:
        evaluate("1+" * 200 + "1")
    assert e.value.code == "too_long"
    with pytest.raises(CalculatorError) as e:
        evaluate("1/0")
    assert e.value.code == "division_by_zero"
    with pytest.raises(CalculatorError):
        evaluate("(" * 40 + "1" + ")" * 40)
    with pytest.raises(CalculatorError):
        evaluate("9" * 39 + "*" + "9" * 39)
    with pytest.raises(CalculatorError):
        evaluate(12)
