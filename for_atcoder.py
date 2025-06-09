import sys
import threading
def main():
    import math
    input = sys.stdin.readline

    # --- 入力 ---
    N, K, H, T, D = map(int, input().split())
    own_colors = [tuple(map(float, input().split())) for _ in range(K)]
    target_colors = [tuple(map(float, input().split())) for _ in range(H)]

    # --- モード判定（今回は fixed のみ）---
    mode = 'fixed'

    # --- ウェル面積 A の決定 ---
    A = max(4, min(16, (16 * T) // 5000))
    # A に近い矩形 f1×f2 を選ぶ
    f1 = int(math.sqrt(A))
    while f1 > 0:
        f2 = (A + f1 - 1) // f1
        if f1 * f2 >= A:
            break
        f1 -= 1

    # --- 初期仕切り（全て仕切りON=1）---
    v = [[1]*(N-1) for _ in range(N)]
    h = [[1]*N       for _ in range(N-1)]

    # --- fixed モード：ブロックごとに内部仕切りを外す ---
    blocks = []
    for bi in range(0, N - f1 + 1, f1):
        for bj in range(0, N - f2 + 1, f2):
            # 内部の縦仕切りを下げる
            for di in range(f1):
                for dj in range(f2-1):
                    v[bi+di][bj+dj] = 0
            # 内部の横仕切りを下げる
            for di in range(f1-1):
                for dj in range(f2):
                    h[bi+di][bj+dj] = 0
            blocks.append((bi, bj))
    # （余りセルはそのまま1マスウェルとして残る）

    # --- 仕切り出力 ---
    for i in range(N):
        print(" ".join(str(x) for x in v[i]))
    for i in range(N-1):
        print(" ".join(str(x) for x in h[i]))

    # --- ビームサーチの準備 ---
    from heapq import heappush, heappop
    BEAM_WIDTH = 32
    MAX_DEPTH = 5
    def beam_search(colors, target):
        # (err, used, c_sum, m_sum, y_sum, path)
        initial = (math.dist((0,0,0), target), 0, 0.0, 0.0, 0.0, [])
        beam = [initial]
        for _ in range(MAX_DEPTH):
            nb = []
            for err, used, c, m, y, path in beam:
                for idx, (C0,M0,Y0) in enumerate(colors):
                    tot = used + 1
                    nc = (c*used + C0)/tot
                    nm = (m*used + M0)/tot
                    ny = (y*used + Y0)/tot
                    nerr = math.sqrt((nc-target[0])**2 + (nm-target[1])**2 + (ny-target[2])**2)
                    heappush(nb, (nerr, tot, nc, nm, ny, path+[idx]))
                    if len(nb) > BEAM_WIDTH:
                        heappop(nb)
            beam = nb
        # 最良ノードを返す
        return min(beam, key=lambda x: x[0])

    # --- 操作列生成 ---
    ops = []
    max_ops = int(T * 0.9)
    blk_cnt = len(blocks)
    for t_i, tgt in enumerate(target_colors):
        if len(ops) >= max_ops:
            break
        # ビームサーチで投入シーケンス獲得
        err, used, *_ , seq = beam_search(own_colors, tgt)
        # ブロック選択
        bi, bj = blocks[t_i % blk_cnt]
        # 操作1: 各チューブ投入
        for tube_idx in seq:
            ops.append((1, bi, bj, tube_idx))
        # 操作2: 取り出し
        ops.append((2, bi, bj))
        # 操作3: 残りを一回捨て (簡易)
        ops.append((3, bi, bj))

    # --- 操作出力 ---
    for op in ops:
        if op[0] == 1:
            _, i, j, k = op
            print(f"1 {i} {j} {k}")
        elif op[0] == 2:
            _, i, j = op
            print(f"2 {i} {j}")
        elif op[0] == 3:
            _, i, j = op
            print(f"3 {i} {j}")

if __name__ == "__main__":
    threading.Thread(target=main).start()
