import sys
import math
from typing import List, Tuple, Set

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
        
        self.well_size = 4
        self.wells_per_row = 0
        self.total_wells = 0
        
        self.operations: List[str] = []
        self.turns_used = 0
        self.paint_taken_count = 0
        
        self.well_usage: List[bool] = []  # 現在のターゲットで使用中ならTrue
        self.well_last_target_idx: List[int] = []  # 最後にこのウェルを使用したターゲットのインデックス
        
        self.EPS = EPS
        self.TAKE_THRESHOLD = TAKE_THRESHOLD
        self.REUSE_DIST_THRESHOLD = 0.15  # ウェル再利用のための固定閾値

    def read_input(self):
        line = sys.stdin.readline().split()
        self.N, self.K, self.H, self.T, self.D = map(int, line)
        
        self.wells_per_row = self.N // self.well_size
        self.total_wells = self.wells_per_row ** 2
        self.well_usage = [False] * self.total_wells
        self.well_last_target_idx = [-1] * self.total_wells  # 初期化
        
        for _ in range(self.K):
            c, m, y = map(float, sys.stdin.readline().split())
            self.tubes.append((c, m, y))
        
        for _ in range(self.H):
            c, m, y = map(float, sys.stdin.readline().split())
            self.targets.append((c, m, y))
    
    def get_well_bounds(self, well_id: int) -> Tuple[int, int, int, int]:
        well_row = well_id // self.wells_per_row
        well_col = well_id % self.wells_per_row
        start_i = well_row * self.well_size
        end_i = start_i + self.well_size
        start_j = well_col * self.well_size
        end_j = start_j + self.well_size
        return start_i, end_i, start_j, end_j
    
    def initialize_palette(self):
        self.v_walls = [[0 for _ in range(self.N - 1)] for _ in range(self.N)]
        self.h_walls = [[0 for _ in range(self.N)] for _ in range(self.N - 1)]
        
        for well_id in range(self.total_wells):
            start_i, end_i, start_j, end_j = self.get_well_bounds(well_id)
            well_cells = set()
            for r_idx in range(start_i, end_i):
                for c_idx in range(start_j, end_j):
                    if r_idx < self.N and c_idx < self.N:
                        well_cells.add((r_idx, c_idx))
                        self.cell_to_well[(r_idx, c_idx)] = well_id
            
            self.wells[well_id] = {'cells': well_cells, 'color': (0.0, 0.0, 0.0), 'amount': 0.0}
            
            for r_idx in range(start_i, end_i):
                for c_idx in range(start_j, end_j):
                    if c_idx + 1 < self.N and (c_idx + 1) % self.well_size == 0:
                        self.v_walls[r_idx][c_idx] = 1
            for r_idx in range(start_i, end_i):
                # 水平壁については、ウェル内のセルの下端（境界）のみを考慮する
                if r_idx + 1 < self.N and (r_idx + 1) % self.well_size == 0: 
                    # これはウェルの境界行
                    for c_idx_h in range(start_j, end_j):
                        if c_idx_h < self.N:
                            self.h_walls[r_idx][c_idx_h] = 1

    def output_initial_state(self):
        for r_idx in range(self.N):
            print(' '.join(map(str, self.v_walls[r_idx])))
        for r_idx in range(self.N - 1):
            print(' '.join(map(str, self.h_walls[r_idx])))
        sys.stdout.flush()

    def _add_op(self, op_str: str, op_type: int) -> bool:
        if self.turns_used >= self.T:
            return False
        self.operations.append(op_str)
        self.turns_used += 1
        if op_type == 2:  # ペイント抽出 (操作2)
            self.paint_taken_count += 1
        return True

    def color_distance(self, c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> float:
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)

    def mix_colors(self, c1: Tuple[float, float, float], v1: float, 
                   c2: Tuple[float, float, float], v2: float) -> Tuple[float, float, float]:
        if abs(v1) < self.EPS and abs(v2) < self.EPS:
            return (0.0, 0.0, 0.0)
        if abs(v1) < self.EPS:
            return c2
        if abs(v2) < self.EPS:
            return c1
        total_v = v1 + v2
        if abs(total_v) < self.EPS:
            return (0.0, 0.0, 0.0)
        return (
            (c1[0] * v1 + c2[0] * v2) / total_v,
            (c1[1] * v1 + c2[1] * v2) / total_v,
            (c1[2] * v1 + c2[2] * v2) / total_v,
        )

    def clear_well(self, well_id: int) -> bool:
        well = self.wells[well_id]
        if well['amount'] > self.EPS:
            rep_cell = next(iter(well['cells']))
            r_idx, c_idx = rep_cell
            if self._add_op(f"3 {r_idx} {c_idx}", 3):
                well['amount'] = 0.0
                well['color'] = (0.0, 0.0, 0.0)
                return True
        return False

    def add_paint(self, well_id: int, tube_idx: int) -> bool:
        well = self.wells[well_id]
        rep_cell = next(iter(well['cells']))
        r_idx, c_idx = rep_cell
        
        max_well_capacity = float(self.K)
        current_amount = well['amount']
        
        if current_amount >= max_well_capacity - self.EPS:
            return False
        
        actual_add_amount = min(1.0, max_well_capacity - current_amount)
        if actual_add_amount < self.EPS:
            return False

        if not self._add_op(f"1 {r_idx} {c_idx} {tube_idx}", 1):
            return False

        tube_color = self.tubes[tube_idx]
        if current_amount < self.EPS:
            well['color'] = tube_color
            well['amount'] = actual_add_amount
        else:
            well['color'] = self.mix_colors(well['color'], current_amount, tube_color, actual_add_amount)
            well['amount'] += actual_add_amount
        return True

    def extract_paint(self, well_id: int) -> bool:
        well = self.wells[well_id]
        rep_cell = next(iter(well['cells']))
        r_idx, c_idx = rep_cell
        
        if well['amount'] < self.TAKE_THRESHOLD:
            return False
        
        if not self._add_op(f"2 {r_idx} {c_idx}", 2):
            return False

        if well['amount'] >= 1.0:
            well['amount'] -= 1.0
        else:
            well['amount'] = 0.0
        
        if well['amount'] < self.EPS:
            well['color'] = (0.0, 0.0, 0.0)
            well['amount'] = 0.0
        return True
    
    def find_available_well_for_target(self, target_idx: int) -> Tuple[int, str]:
        target_color = self.targets[target_idx]
        start_well_search_idx = target_idx % self.total_wells  # 巡回検索の開始インデックス

        # 優先順位1: 完全に空で未使用のウェル
        for i in range(self.total_wells):
            well_id = (start_well_search_idx + i) % self.total_wells
            if not self.well_usage[well_id] and self.wells[well_id]['amount'] < self.EPS:
                return well_id, "empty"
        
        # 優先順位2: 再利用可能なウェル（十分なペイント量があり、目標色に近く、使用中でない）
        best_reusable_well_id = -1
        min_dist_for_reusable = float('inf')
        for i in range(self.total_wells):
            well_id = (start_well_search_idx + i) % self.total_wells
            if not self.well_usage[well_id] and self.wells[well_id]['amount'] >= self.TAKE_THRESHOLD:
                dist = self.color_distance(self.wells[well_id]['color'], target_color)
                if dist < self.REUSE_DIST_THRESHOLD and dist < min_dist_for_reusable:
                    min_dist_for_reusable = dist
                    best_reusable_well_id = well_id
        
        if best_reusable_well_id != -1:
            return best_reusable_well_id, "reuse"

        # 優先順位3: ウェルをクリアする。最も最近使用されていないか、単純に巡回順で選ぶ。
        chosen_clear_well_id = start_well_search_idx  # 全てが「悪い」または使用中の場合のデフォルト
        
        # 使用中でないウェルを探す
        can_find_non_used_to_clear = False
        for i in range(self.total_wells):
            well_id = (start_well_search_idx + i) % self.total_wells
            if not self.well_usage[well_id]:
                chosen_clear_well_id = well_id
                can_find_non_used_to_clear = True
                break
        
        if can_find_non_used_to_clear:
            return chosen_clear_well_id, "clear_and_use"  # 現在使用中でないウェルをクリアして使用
        else:
            # すべてのウェルが使用中、または再利用できなかった場合は巡回順でクリアする
            return start_well_search_idx, "force_clear_and_use"

    def allocate_op1_tubes_greedy(self, well_id: int, target_color: Tuple[float, float, float], num_op1_budget: int) -> List[int]:
        """貪欲法による操作1のチューブ追加シーケンスを決定するシミュレーション"""
        sim_current_color = self.wells[well_id]['color']
        sim_current_amount = self.wells[well_id]['amount']
        max_well_capacity = float(self.K)
        tubes_to_add_sequence = []

        for _ in range(num_op1_budget):
            if sim_current_amount >= max_well_capacity - self.EPS:
                break 

            best_tube_idx_to_add = -1
            min_resulting_dist = float('inf')

            for t_idx, tube_c_data_candidate in enumerate(self.tubes):
                # このチューブを1g追加するシミュレーション
                sim_add_amount = min(1.0, max_well_capacity - sim_current_amount)
                if sim_add_amount < self.EPS:
                    continue 

                mixed_c_sim = self.mix_colors(sim_current_color, sim_current_amount, 
                                              tube_c_data_candidate, sim_add_amount)
                dist = self.color_distance(mixed_c_sim, target_color)
                if dist < min_resulting_dist:
                    min_resulting_dist = dist
                    best_tube_idx_to_add = t_idx
            
            if best_tube_idx_to_add != -1:
                tubes_to_add_sequence.append(best_tube_idx_to_add)
                # シミュレーション状態を更新
                sim_add_amount_actual = min(1.0, max_well_capacity - sim_current_amount)
                sim_current_color = self.mix_colors(sim_current_color, sim_current_amount, 
                                                     self.tubes[best_tube_idx_to_add], sim_add_amount_actual)
                sim_current_amount += sim_add_amount_actual
            else:
                break  # 適切なチューブが見つからなかった場合
        return tubes_to_add_sequence

    def create_target_color(self, target_idx: int):
        target_color = self.targets[target_idx]

        # --- ターン予算の計画 ---
        remaining_targets_to_make = self.H - self.paint_taken_count 
        turns_left_total = self.T - self.turns_used
        
        if turns_left_total <= 0 or remaining_targets_to_make <= 0:
            return 

        budget_this_target_ops = 1  # 操作2用に最低1ターン必要
        
        available_for_non_op2 = turns_left_total - remaining_targets_to_make
        if available_for_non_op2 > 0 and remaining_targets_to_make > 0:
            budget_this_target_ops += math.floor(available_for_non_op2 / remaining_targets_to_make)
        
        if target_idx == self.H - 1:  # 最後のターゲットは残りターンすべて使用可能
            budget_this_target_ops = turns_left_total
        
        budget_this_target_ops = max(1, budget_this_target_ops)
        budget_this_target_ops = min(budget_this_target_ops, turns_left_total)

        # --- ウェルの選択と準備 ---
        well_id, well_state_type = self.find_available_well_for_target(target_idx)
        turns_spent_locally = 0 

        if well_state_type in ["clear_and_use", "force_clear_and_use"]:
            # クリア後の最小ターン数: ペイント量が十分なら1（操作2）、不足なら2（操作1 + 操作2）
            min_ops_after_clear = 1
            if self.wells[well_id]['amount'] < self.TAKE_THRESHOLD or well_state_type == "force_clear_and_use":
                min_ops_after_clear += 1  # 空または不適なウェルの場合、操作1が必要
            if budget_this_target_ops - turns_spent_locally >= (1 + min_ops_after_clear):  # 1はクリア操作の分
                if self.clear_well(well_id):
                    turns_spent_locally += 1
                else:
                    return  # クリア失敗（例：ターン不足）
            else:
                # クリアおよび最小操作を行う予算が足りない場合
                return 
        
        self.well_usage[well_id] = True  # このターゲット用にウェルを使用中とマーク

        # --- ペイント追加操作（操作1） ---
        num_op1_allowed = budget_this_target_ops - turns_spent_locally - 1  # 操作2のために1ターンを確保
        num_op1_allowed = max(0, num_op1_allowed)  # 予算が厳しい場合は0になる可能性あり

        # 簡略化した貪欲法によるチューブ追加のシミュレーションを使用
        tubes_to_add = self.allocate_op1_tubes_greedy(well_id, target_color, num_op1_allowed)
        
        for tube_idx_to_add in tubes_to_add:
            if self.add_paint(well_id, tube_idx_to_add):
                turns_spent_locally += 1
            else:
                break 
        
        # --- ペイント抽出（操作2） ---
        can_perform_op2 = False
        # 操作2を行うために最低1ターン必要
        if budget_this_target_ops - turns_spent_locally >= 1:
            # ペイント量が閾値未満の場合、抽出前に緊急でペイント追加（1追加＋1抽出＝計2ターン必要）
            if self.wells[well_id]['amount'] < self.TAKE_THRESHOLD and \
               budget_this_target_ops - turns_spent_locally >= 2 and \
               self.wells[well_id]['amount'] < float(self.K) - self.EPS:
                if self.add_paint(well_id, 0):  # 緊急追加（tube0）を実施
                    turns_spent_locally += 1
            # 緊急追加後に再度抽出を試みる
            if budget_this_target_ops - turns_spent_locally >= 1:
                if self.extract_paint(well_id):
                    turns_spent_locally += 1  # _add_op内で既にカウント済み
                    can_perform_op2 = True
        
        # このターゲットに対するウェルの使用状態を解除する
        self.well_usage[well_id] = False 
        self.well_last_target_idx[well_id] = target_idx  # 最後に使用したターゲットのインデックスを更新
        
        # 操作2が成功しなかった場合は、solve()内の緊急処理で対応

    def solve(self):
        self.read_input()
        self.initialize_palette()
        self.output_initial_state()
        
        for target_idx_loop in range(self.H):
            if self.turns_used + (self.H - self.paint_taken_count) > self.T:
                break 
            self.create_target_color(target_idx_loop)
        
        # H個のペイント抽出操作に対する緊急処理
        emergency_well_idx_counter = 0
        while self.paint_taken_count < self.H and self.turns_used < self.T:
            current_well_id_emergency = emergency_well_idx_counter % self.total_wells
            
            # この緊急操作に必要なターン数を決定
            turns_for_this_emergency = 1  # 操作2分
            if self.wells[current_well_id_emergency]['amount'] < self.TAKE_THRESHOLD:
                turns_for_this_emergency += 1  # 操作1（追加）の分
            if self.wells[current_well_id_emergency]['amount'] > self.EPS and \
               self.wells[current_well_id_emergency]['amount'] < self.TAKE_THRESHOLD:
                turns_for_this_emergency += 1  # 一部ペイントがあるが不足している場合、操作3（クリア）の分
            elif self.wells[current_well_id_emergency]['amount'] <= self.EPS and \
                 self.wells[current_well_id_emergency]['amount'] < self.TAKE_THRESHOLD:
                pass  # 空で追加＋抽出の分は既に含まれている
            
            if self.T - self.turns_used < turns_for_this_emergency:
                break 

            if self.wells[current_well_id_emergency]['amount'] > self.EPS and \
               self.wells[current_well_id_emergency]['amount'] < self.TAKE_THRESHOLD:
                self.clear_well(current_well_id_emergency)
            
            if self.wells[current_well_id_emergency]['amount'] < self.TAKE_THRESHOLD:
                self.add_paint(current_well_id_emergency, 0)
            
            self.extract_paint(current_well_id_emergency)
            
            # ペイントがなくなった場合、ウェルの使用状態を解除する
            if self.wells[current_well_id_emergency]['amount'] < self.EPS:
                self.well_usage[current_well_id_emergency] = False  # 未使用とマーク
            self.well_last_target_idx[current_well_id_emergency] = self.H  # 緊急処理で使用されたことを記録
            emergency_well_idx_counter += 1

        for op_str_final in self.operations:
            print(op_str_final)
        sys.stdout.flush()

if __name__ == "__main__":
    solver = PaintMixer()
    solver.solve()