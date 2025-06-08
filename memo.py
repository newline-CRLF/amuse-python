import sys
input = sys.stdin.readline
def solve():
    N = int(input())
    S = input().strip()
    dp = [[float('inf')] * 3 for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(N):
        char_val = int(S[i])
        cost_to_be_0 = (1 if char_val == 1 else 0)
        if dp[i][0] != float('inf'):
            dp[i+1][0] = dp[i][0] + cost_to_be_0
        cost_to_be_1 = (1 if char_val == 0 else 0)
        val1 = float('inf')
        if dp[i][0] != float('inf'):
            val1 = dp[i][0] + cost_to_be_1
        val2 = float('inf')
        if dp[i][1] != float('inf'):
            val2 = dp[i][1] + cost_to_be_1
        dp[i+1][1] = min(val1, val2)
        cost_to_be_0 = (1 if char_val == 1 else 0)
        val1 = float('inf')
        if dp[i][1] != float('inf'):
            val1 = dp[i][1] + cost_to_be_0
        val2 = float('inf')
        if dp[i][2] != float('inf'):
            val2 = dp[i][2] + cost_to_be_0
        dp[i+1][2] = min(val1, val2)
    ans = min(dp[N][0], dp[N][1], dp[N][2])
    print(ans)
T = int(input())
for _ in range(T):
    solve()