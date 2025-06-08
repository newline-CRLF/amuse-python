def solve():
    # 入力を読み取り
    L, N = map(int, input().split())
    d = list(map(int, input().split()))
    
    # 各点の位置を計算（点1を基準位置0とする）
    positions = [0]  # 点1の位置
    current_pos = 0
    
    for i in range(N - 1):
        current_pos = (current_pos + d[i]) % L
        positions.append(current_pos)
    
    # 正三角形の条件：3点が円周を3等分する
    # L が 3 で割り切れない場合、正三角形は作れない
    if L % 3 != 0:
        print(0)
        return
    
    # 正三角形を作るための間隔
    interval = L // 3
    
    # 各位置に何個の点があるかをカウント
    position_count = {}
    for pos in positions:
        position_count[pos] = position_count.get(pos, 0) + 1
    
    # 正三角形を作る点の組み合わせを数える
    count = 0
    
    # すべての可能な正三角形の基準点を試す
    checked = set()
    
    for pos in position_count:
        if pos in checked:
            continue
            
        # この位置を基準とした正三角形の3つの頂点位置
        pos1 = pos
        pos2 = (pos + interval) % L
        pos3 = (pos + 2 * interval) % L
        
        # 3つの位置すべてに点が存在するかチェック
        if pos1 in position_count and pos2 in position_count and pos3 in position_count:
            # 3つの位置が異なる場合
            if pos1 != pos2 and pos2 != pos3 and pos1 != pos3:
                # 各位置の点の数の積が組み合わせ数
                count += position_count[pos1] * position_count[pos2] * position_count[pos3]
                
                # 重複を避けるため、チェック済みとしてマーク
                checked.add(pos1)
                checked.add(pos2)
                checked.add(pos3)
    
    print(count)

# 実行
solve()