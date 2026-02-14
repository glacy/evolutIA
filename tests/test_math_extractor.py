
import pytest
from evolutia.utils.math_extractor import (
    extract_math_expressions,
    extract_variables,
    count_math_operations,
    estimate_complexity
)

def test_extract_math_expressions():
    content = r"""
    This is inline math: $x^2 + y^2 = z^2$.
    This is another inline: \(a + b\).

    This is display math:
    $$
    \int_0^\infty e^{-x} dx
    $$

    And this is a math block:
    :::{math}
    \sum_{n=0}^\infty \frac{1}{n!}
    :::
    """
    expressions = extract_math_expressions(content)
    assert len(expressions) == 4
    assert r"x^2 + y^2 = z^2" in expressions
    assert r"a + b" in expressions
    assert r"\int_0^\infty e^{-x} dx" in expressions
    assert r"\sum_{n=0}^\infty \frac{1}{n!}" in expressions

def test_extract_variables():
    expressions = [
        r"E = mc^2",
        r"\vec{F} = m\vec{a}",
        r"\alpha + \beta = \gamma"
    ]
    variables = extract_variables(expressions)
    expected_vars = {'E', 'm', 'c', 'F', 'a', 'alpha', 'beta', 'gamma'}
    assert variables == expected_vars

def test_count_math_operations():
    expr = r"\int \frac{d}{dx} \sum \vec{v} \begin{pmatrix} 1 \\ 0 \end{pmatrix} \sin(x)"
    ops = count_math_operations(expr)
    assert ops['integrals'] == 1
    assert ops['derivatives'] == 1
    assert ops['sums'] == 1
    assert ops['vectors'] == 1
    assert ops['matrices'] == 1
    assert ops['functions'] == 1

def test_estimate_complexity():
    simple_expr = [r"x + y = z"]
    complex_expr = [r"\int_{-\infty}^\infty e^{-x^2} dx = \sqrt{\pi}"]

    simple_score = estimate_complexity(simple_expr)
    complex_score = estimate_complexity(complex_expr)

    assert complex_score > simple_score


def test_count_math_operations_edge_cases():
    test_cases = [
        (r"\vec{\nabla}", {'vectors': 1, 'derivatives': 1}),
        (r"\int\int", {'integrals': 2}),
        (r"\frac{d}{dx}\int", {'derivatives': 1, 'integrals': 1}),
        (r"\begin{pmatrix}\int\end{pmatrix}", {'matrices': 1, 'integrals': 1}),
        (r"\sin\cos\tan", {'functions': 3}),
        (r"\vec\mathbf", {'vectors': 2}),
    ]

    for expr, expected_counts in test_cases:
        ops = count_math_operations(expr)
        for key, count in expected_counts.items():
            assert ops[key] == count, f"Failed for {expr}: expected {count} {key}, got {ops[key]}"
