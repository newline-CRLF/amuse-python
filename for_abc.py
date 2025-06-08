import sys
from collections import Counter

def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print(0)
        return
    it = iter(data)
    L = int(next(it))
    N = int(next(it))
    # read d_1 through d_{N-1}, possibly across multiple lines
    ds = [int(next(it)) for _ in range(N-1)]

    # compute positions on the circle
    pos = [0] * N
    for i in range(1, N):
        pos[i] = (pos[i-1] + ds[i-1]) % L

    # if circumference not divisible by 3, no equilateral triangles possible
    if L % 3 != 0:
        print(0)
        return

    t = L // 3
    c = Counter(pos)
    ans = 0
    # count triples (v, v+t, v+2t) for v in [0, t-1]
    for v in range(t):
        ans += c[v] * c[v + t] * c[v + 2 * t]
    print(ans)

if __name__ == '__main__':
    main()
