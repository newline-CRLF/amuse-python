def sieve_of_eratosthenes(n):
    """エラトステネスの篩を用いて、n以下の素数をリストで返す（より高速な実装）"""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = bytearray([0, 0])
    sqrt_n = int(n**0.5) + 1
    for i in range(2, sqrt_n):
        if sieve[i]:
            sieve[i*i:n+1:i] = bytearray([0]) * len(sieve[i*i:n+1:i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Example usage
if __name__ == "__main__":
    n = 100
    primes = sieve_of_eratosthenes(n)
    print(f"2から{n}までの素数: {primes}")