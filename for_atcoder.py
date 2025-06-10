import sys
import math
from typing import List, Tuple
import threading

# 浮動小数点数の比較用定数
EPS = 1e-9
TAKE_THRESHOLD = 1.0 - 1e-6

class PaintMixer:
    def __init__(self):
        self.N = 0
        self.K = 0
        self.H = 0
        self.T = 0
        self.D = 0
        
        self.tubes: List[Tuple[float, float, float]] = []
        self.targets: List[Tuple[float, float, float]] = []
        
        self.v_walls: List[List[int]] = []
        self.h_walls: List[List[int]] = []
        self.wells = {} 
        self.cell_to_well = {} 
        
        self.well_configs: List[Tuple[int, int, int, int, int]] = []
        self.total_wells = 0
        
        self.operations: List[str] = []
        self.turns_used = 0
        self.paint_taken_count = 0
        
        self.well_usage: List[bool] = []
        self.well_last_target_idx: List[int] = []

        # --- パラメータ ---
        self.BEAM_WIDTH = 32      # ビームサーチの幅
        self.MAX_DEPTH = 5        # ビームサーチの深さ
        self.ERR_FALLBACK = 1e-2  # フォールバック投入閾値

    def read_input(self):
        self.N, self.K, self.H, self.T, self.D = map(int, sys.stdin.readline().split())
        for _ in range(self.K):
            c, m, y = map(float, sys.stdin.readline().split())
            self.tubes.append((c, m, y))
        for _ in range(self.H):
            c, m, y = map(float, sys.stdin.readline().split())
            self.targets.append((c, m, y))

    def initialize_palette(self):
        N, T = self.N, self.T
        # ウェル面積 A の決定
        A = max(4, min(16, (16 * T) // 5000))
        f1 = int(math.sqrt(A))
        while f1 > 0:
            f2 = (A + f1 - 1) // f1
            if f1 * f2 >= A:
                break
            f1 -= 1

        # 仕切り全 ON
        self.v_walls = [[1]*(N-1) for _ in range(N)]
        self.h_walls = [[1]*N       for _ in range(N-1)]

        # ブロックごとに内部仕切りを下ろしてウェルを作る
        current_id = 0
        for bi in range(0, N - f1 + 1, f1):
            for bj in range(0, N - f2 + 1, f2):
                # 縦仕切り下ろし
                for di in range(f1):
                    for dj in range(f2-1):
                        self.v_walls[bi+di][bj+dj] = 0
                # 横仕切り下ろし
                for di in range(f1-1):
                    for dj in range(f2):
                        self.h_walls[bi+di][bj+dj] = 0
                # ウェル設定
                self.well_configs.append((current_id, bi, bi+f1, bj, bj+f2))
                current_id += 1

        self.total_wells = current_id
        self.well_usage = [False] * self.total_wells
        self.well_last_target_idx = [-1] * self.total_wells

        # 各ウェルにセル割当
        for well_id, si, ei, sj, ej in self.well_configs:
            cells = set()
            for i in range(si, ei):
                for j in range(sj, ej):
                    cells.add((i, j))
                    self.cell_to_well[(i, j)] = well_id
            self.wells[well_id] = {
                'cells': cells,
                'color': (0.0, 0.0, 0.0),
                'amount': 0.0
            }

    def output_initial_state(self):
        for row in self.v_walls:
            print(" ".join(str(x) for x in row))
        for row in self.h_walls:
            print(" ".join(str(x) for x in row))
        sys.stdout.flush()

    def _add_op(self, op_str: str, op_type: int) -> bool:
        if self.turns_used >= self.T:
            return False
        self.operations.append(op_str)
        self.turns_used += 1
        if op_type == 2:
            self.paint_taken_count += 1
        return True

    def color_distance(self, c1, c2) -> float:
        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

    def mix_colors(self, c1, v1, c2, v2):
        if v1 + v2 < EPS:
            return (0.0, 0.0, 0.0)
        return (
            (c1[0]*v1 + c2[0]*v2) / (v1+v2),
            (c1[1]*v1 + c2[1]*v2) / (v1+v2),
            (c1[2]*v1 + c2[2]*v2) / (v1+v2),
        )

    def clear_well(self, well_id):
        well = self.wells[well_id]
        if well['amount'] > EPS:
            i, j = next(iter(well['cells']))
            if self._add_op(f"3 {i} {j}", 3):
                well['amount'] = 0.0
                well['color'] = (0.0, 0.0, 0.0)
                return True
        return False

    def add_paint(self, well_id, tube_idx):
        well = self.wells[well_id]
        i, j = next(iter(well['cells']))
        cap = float(len(well['cells']))
        if well['amount'] >= cap - EPS:
            return False
        add_amt = min(1.0, cap - well['amount'])
        if not self._add_op(f"1 {i} {j} {tube_idx}", 1):
            return False
        well['color'] = self.mix_colors(well['color'], well['amount'], self.tubes[tube_idx], add_amt)
        well['amount'] += add_amt
        return True

    def extract_paint(self, well_id):
        well = self.wells[well_id]
        i, j = next(iter(well['cells']))
        if well['amount'] < TAKE_THRESHOLD:
            return False
        if not self._add_op(f"2 {i} {j}", 2):
            return False
        well['amount'] -= 1.0
        if well['amount'] < EPS:
            well['amount'] = 0.0
            well['color'] = (0.0, 0.0, 0.0)
        return True

    def find_best_mix_sequence(self, well_id, target_color, budget, init_color, init_amt):
        max_cap = float(len(self.wells[well_id]['cells']))
        depth = min(budget, self.MAX_DEPTH)
        beam = [(self.color_distance(init_color, target_color), init_amt, init_color, [])]

        for _ in range(depth):
            next_cands = []
            for err, amt, col, seq in beam:
                if amt >= max_cap - EPS:
                    next_cands.append((err, amt, col, seq))
                    continue
                for tidx in range(self.K):
                    add_amt = min(1.0, max_cap - amt)
                    new_col = self.mix_colors(col, amt, self.tubes[tidx], add_amt)
                    new_amt = amt + add_amt
                    new_seq = seq + [tidx]
                    new_err = self.color_distance(new_col, target_color)
                    next_cands.append((new_err, new_amt, new_col, new_seq))
                if err > self.ERR_FALLBACK or not seq:
                    best_t, best_d = None, float('inf')
                    for tidx in range(self.K):
                        trial_amt = min(1.0, max_cap - amt)
                        trial_col = self.mix_colors(col, amt, self.tubes[tidx], trial_amt)
                        d = self.color_distance(trial_col, target_color)
                        if d < best_d:
                            best_d, best_t = d, tidx
                    if best_t is not None:
                        fb_amt = min(1.0, max_cap - amt)
                        fb_col = self.mix_colors(col, amt, self.tubes[best_t], fb_amt)
                        next_cands.append((best_d, amt+fb_amt, fb_col, seq+[best_t]))
            next_cands.sort(key=lambda x: x[0])
            beam = next_cands[:self.BEAM_WIDTH]

        return beam[0][3] if beam else []

    def solve(self):
        self.read_input()
        self.initialize_palette()
        self.output_initial_state()

        ops_limit = int(self.T * 0.9)
        # blocks を (si, sj) のリストとして正しく生成
        blocks = [(si, sj) for (_, si, ei, sj, ej) in self.well_configs]
        bcnt = len(blocks)

        for idx, tgt in enumerate(self.targets):
            # 残ターンと残ターゲットを考慮して予算算出
            remain = self.H - self.paint_taken_count
            left_turns = self.T - self.turns_used
            if left_turns <= 0 or remain <= 0:
                break
            budget = 1 + (left_turns - remain) // remain
            budget = max(1, min(budget, left_turns))

            bi, bj = blocks[idx % bcnt]
            well_id = self.cell_to_well[(bi, bj)]

            # 必要ならクリア
            if self.wells[well_id]['amount'] > EPS and budget >= 2:
                self.clear_well(well_id)
                budget -= 1

            # 混色シーケンス取得
            seq = self.find_best_mix_sequence(
                well_id, tgt, budget-1,
                self.wells[well_id]['color'], self.wells[well_id]['amount']
            )
            # 投入
            for t in seq:
                if budget <= 1:
                    break
                if self.add_paint(well_id, t):
                    budget -= 1
                else:
                    break
            # 取り出し
            if budget >= 1:
                self.extract_paint(well_id)

        # 緊急取り出し
        well_idx = 0
        while self.paint_taken_count < self.H and self.turns_used < self.T:
            well_id = well_idx % self.total_wells
            if self.wells[well_id]['amount'] < TAKE_THRESHOLD:
                self.add_paint(well_id, 0)
            self.extract_paint(well_id)
            well_idx += 1

        for op in self.operations:
            print(op)
        sys.stdout.flush()

def main():
    mixer = PaintMixer()
    mixer.solve()

if __name__ == "__main__":
    threading.Thread(target=main).start()
