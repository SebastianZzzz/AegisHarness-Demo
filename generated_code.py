from typing import Optional, Tuple

def _validate_n(n: int) -> None:
    """
    Validate that n is a non-negative integer (not a boolean).
    """
    if isinstance(n, bool):
        raise TypeError("N must be a non-boolean integer, got bool")
    if not isinstance(n, int):
        raise TypeError(f"N must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"N must be a non-negative integer, got {n}")

def _validate_mod(mod: Optional[int]) -> None:
    """
    Validate that mod, if provided, is a positive integer (not a boolean).
    """
    if mod is None:
        return
    if isinstance(mod, bool):
        raise TypeError("mod must be a non-boolean integer, got bool")
    if not isinstance(mod, int):
        raise TypeError(f"mod must be an int, got {type(mod).__name__}")
    if mod <= 0:
        raise ValueError(f"mod must be a positive integer, got {mod}")

def _fib_doubling(n: int, mod: Optional[int] = None) -> Tuple[int, int]:
    """
    Compute F(n) and F(n+1) using the fast-doubling method.

    Returns:
        A tuple (F(n), F(n+1))
    """
    if n == 0:
        return (0, 1)
    a, b = _fib_doubling(n >> 1, mod)  # F(k), F(k+1) where k = n // 2
    if mod is None:
        c = a * (2 * b - a)  # F(2k)
        d = a * a + b * b      # F(2k + 1)
    else:
        t = (2 * b - a) % mod
        c = (a * t) % mod        # F(2k) mod m
        d = (a * a + b * b) % mod  # F(2k + 1) mod m
    if (n & 1) == 0:
        return (c, d)
    else:
        if mod is None:
            return (d, c + d)
        else:
            return (d, (c + d) % mod)

def fibonacci(n: int, mod: Optional[int] = None) -> int:
    """
    Compute the N-th Fibonacci number using the fast-doubling method.

    Specifications:
    - F(0) = 0, F(1) = 1
    - Time complexity: O(log N)
    - Space complexity: O(log N) recursion depth
    - If mod is provided, returns F(n) modulo mod.

    Parameters:
    - n: non-negative integer index
    - mod: optional positive integer modulus to compute F(n) mod mod

    Returns:
    - F(n) as int if mod is None
    - F(n) mod mod if mod is provided

    Raises:
    - TypeError if n is not an int or is a boolean
    - ValueError if n < 0 or mod <= 0
    """
    _validate_n(n)
    _validate_mod(mod)
    result, _ = _fib_doubling(n, mod)
    return result


if __name__ == "__main__":
    # Lightweight self-checks to validate correctness and inputs
    import unittest

    class TestFibonacci(unittest.TestCase):
        def test_base_cases(self):
            self.assertEqual(fibonacci(0), 0)
            self.assertEqual(fibonacci(1), 1)

        def test_small_values(self):
            self.assertEqual(fibonacci(2), 1)
            self.assertEqual(fibonacci(3), 2)
            self.assertEqual(fibonacci(5), 5)
            self.assertEqual(fibonacci(10), 55)

        def test_against_iterative(self):
            def iterative(n: int) -> int:
                a, b = 0, 1
                for _ in range(n):
                    a, b = b, a + b
                return a
            for n in range(0, 50):
                self.assertEqual(fibonacci(n), iterative(n))

        def test_input_validation(self):
            with self.assertRaises(TypeError):
                fibonacci(3.5)
            with self.assertRaises(TypeError):
                fibonacci("10")
            with self.assertRaises(TypeError):
                fibonacci(True)
            with self.assertRaises(ValueError):
                fibonacci(-1)
            with self.assertRaises(ValueError):
                fibonacci(0, mod=0)
            with self.assertRaises(TypeError):
                fibonacci(5, mod=3.14)
            with self.assertRaises(TypeError):
                fibonacci(5, mod="7")
            with self.assertRaises(ValueError):
                fibonacci(5, mod=-7)

        def test_large_n_with_mod(self):
            n = 1_000_000
            mod = 1_000_000_007
            res1 = fibonacci(n, mod=mod)
            res2 = fibonacci(n, mod=mod)
            self.assertEqual(res1, res2)
            self.assertIsInstance(res1, int)
            self.assertTrue(0 <= res1 < mod)

    unittest.main()