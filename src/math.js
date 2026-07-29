function add(a, b) {
  return a + b;
}

function multiply(a, b) {
  return a * b;
}

function isPrime(n) {
  if (n < 2) return false;
  for (let i = 2; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return false;
  }
  return true;
}

module.exports = { add, multiply, isPrime };
