const { add, multiply, isPrime } = require("../src/math");

test("adds numbers", () => {
  expect(add(2, 3)).toBe(5);
});

test("multiplies numbers", () => {
  expect(multiply(4, 5)).toBe(20);
});

test("detects primes", () => {
  expect(isPrime(7)).toBe(true);
  expect(isPrime(8)).toBe(false);
});

test("inject failure switch", () => {
  if (process.env.INJECT_FAILURE === "true") {
    expect(true).toBe(false); // deliberate failure
  } else {
    expect(true).toBe(true);
  }
});
