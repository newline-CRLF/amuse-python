import sys
from typing import List, Tuple, Set
import math

class PaintMixer:
    def __init__(self):
        self.N = 0  # パレットサイズ
        self.K = 0  # チューブ数
        self.H = 0  # 目標色数
        self.T = 0  # 最大ターン数
        self.D = 0  # コスト
        
        self.tubes = []  # チューブの色 [(C, M, Y), ...]
        self.targets = []  # 目標色 [(C, M, Y), ...]
        
        # パレット状態
        self.v_walls = []  # 縦の仕切り
        self.h_walls = []  # 横の仕切り
        self.wells = {}  # ウェル情報 {well_id: {'cells': set, 'color': (C,M,Y), 'amount': float}}
        self.cell_to_well = {}  # セル→ウェルID
        
        # 固定分割設定 (4x4の小ウェルに分割)
        self.well_size = 4  # 各ウェルのサイズ
        self.wells_per_row = self.N // self.well_size if self.N else 5  # 20//4 = 5
        self.total_wells = self.wells_per_row ** 2  # 5x5 = 25個のウェル
        
        self.operations = []  # 操作履歴
        self.current_target = 0  # 現在作成中の目標色インデックス
        self.well_usage = []  # 各ウェルの使用状況
        
    def read_input(self):
        line = input().split()
        self.N, self.K, self.H, self.T, self.D = map(int, line)
        
        # 固定分割設定を更新
        self.well_size = 4
        self.wells_per_row = self.N // self.well_size
        self.total_wells = self.wells_per_row ** 2
        self.well_usage = [False] * self.total_wells  # False = 空き
        
        # チューブ色読み込み
        for _ in range(self.K):
            c, m, y = map(float, input().split())
            self.tubes.append((c, m, y))
        
        # 目標色読み込み
        for _ in range(self.H):
            c, m, y = map(float, input().split())
            self.targets.append((c, m, y))
    
    def get_well_bounds(self, well_id: int) -> Tuple[int, int, int, int]:
        """ウェルIDから境界座標を取得 (start_i, end_i, start_j, end_j)"""
        well_row = well_id // self.wells_per_row
        well_col = well_id % self.wells_per_row
        
        start_i = well_row * self.well_size
        end_i = start_i + self.well_size
        start_j = well_col * self.well_size
        end_j = start_j + self.well_size
        
        return start_i, end_i, start_j, end_j
    
    def initialize_palette(self):
        """パレットを4x4の小ウェルに分割"""
        # 仕切りを初期化
        self.v_walls = [[0 for _ in range(self.N-1)] for _ in range(self.N)]
        self.h_walls = [[0 for _ in range(self.N)] for _ in range(self.N-1)]
        
        # 各小ウェルを作成
        for well_id in range(self.total_wells):
            start_i, end_i, start_j, end_j = self.get_well_bounds(well_id)
            
            well_cells = set()
            for i in range(start_i, end_i):
                for j in range(start_j, end_j):
                    if i < self.N and j < self.N:  # 境界チェック
                        well_cells.add((i, j))
                        self.cell_to_well[(i, j)] = well_id
            
            self.wells[well_id] = {
                'cells': well_cells,
                'color': (0.0, 0.0, 0.0),
                'amount': 0.0
            }
            
            # ウェル境界に仕切りを設置
            for i in range(start_i, end_i):
                for j in range(start_j, end_j):
                    if i < self.N and j < self.N:
                        # 右側の仕切り（ウェル境界の場合）
                        if j + 1 < self.N and (j + 1) % self.well_size == 0:
                            self.v_walls[i][j] = 1
                        
                        # 下側の仕切り（ウェル境界の場合）  
                        if i + 1 < self.N and (i + 1) % self.well_size == 0:
                            self.h_walls[i][j] = 1
    
    def output_initial_state(self):
        """初期状態の仕切り配置を出力"""
        # 縦の仕切り
        for i in range(self.N):
            print(' '.join(map(str, self.v_walls[i])))
        
        # 横の仕切り
        for i in range(self.N-1):
            print(' '.join(map(str, self.h_walls[i])))
    
    def color_distance(self, color1: Tuple[float, float, float], 
                      color2: Tuple[float, float, float]) -> float:
        """2色間の距離を計算"""
        c1, m1, y1 = color1
        c2, m2, y2 = color2
        return math.sqrt((c1-c2)**2 + (m1-m2)**2 + (y1-y2)**2)
    
    def find_best_tube_combination(self, target_color: Tuple[float, float, float]) -> List[Tuple[int, int]]:
        """目標色に最も近い色を作るチューブの組み合わせを見つける (tube_idx, count)"""
        best_combination = []
        best_distance = float('inf')
        
        # 単色での最適解
        for tube_idx in range(self.K):
            distance = self.color_distance(self.tubes[tube_idx], target_color)
            if distance < best_distance:
                best_distance = distance
                best_combination = [(tube_idx, 1)]
        
        # 2色混合での改善を試す（整数比率で）
        for tube1 in range(self.K):
            for tube2 in range(tube1 + 1, self.K):
                for ratio1 in range(1, 6):  # 1:4 から 5:1 まで
                    ratio2 = 6 - ratio1
                    
                    c1, m1, y1 = self.tubes[tube1]
                    c2, m2, y2 = self.tubes[tube2]
                    
                    total = ratio1 + ratio2
                    mixed_color = (
                        (c1 * ratio1 + c2 * ratio2) / total,
                        (m1 * ratio1 + m2 * ratio2) / total,
                        (y1 * ratio1 + y2 * ratio2) / total
                    )
                    
                    distance = self.color_distance(mixed_color, target_color)
                    if distance < best_distance:
                        best_distance = distance
                        best_combination = [(tube1, ratio1), (tube2, ratio2)]
        
        return best_combination
    
    def find_available_well(self) -> int:
        """使用可能なウェルを見つける"""
        for well_id in range(self.total_wells):
            if not self.well_usage[well_id]:
                return well_id
        
        # 空きがない場合は適当なウェルをクリアして使用
        well_id = self.current_target % self.total_wells
        self.clear_well(well_id)
        return well_id
    
    def clear_well(self, well_id: int):
        """ウェルをクリア"""
        well = self.wells[well_id]
        if well['amount'] > 0:
            # 代表セルから廃棄
            representative_cell = next(iter(well['cells']))
            i, j = representative_cell
            self.operations.append(f"3 {i} {j}")
            
            well['amount'] = 0.0
            well['color'] = (0.0, 0.0, 0.0)
        
        self.well_usage[well_id] = False
    
    def add_paint(self, well_id: int, tube_idx: int):
        """ウェルに絵の具を追加"""
        well = self.wells[well_id]
        representative_cell = next(iter(well['cells']))
        i, j = representative_cell
        
        # 容量チェック
        max_capacity = len(well['cells'])
        current_amount = well['amount']
        
        if current_amount >= max_capacity:
            return  # 容量オーバー
        
        # 追加量を決定
        add_amount = min(1.0, max_capacity - current_amount)
        
        if add_amount > 0:
            # 色を混合
            if current_amount == 0:
                well['color'] = self.tubes[tube_idx]
                well['amount'] = add_amount
            else:
                old_c, old_m, old_y = well['color']
                new_c, new_m, new_y = self.tubes[tube_idx]
                
                total_amount = current_amount + add_amount
                well['color'] = (
                    (old_c * current_amount + new_c * add_amount) / total_amount,
                    (old_m * current_amount + new_m * add_amount) / total_amount,
                    (old_y * current_amount + new_y * add_amount) / total_amount
                )
                well['amount'] = total_amount
        
        # 操作を記録
        self.operations.append(f"1 {i} {j} {tube_idx}")
    
    def extract_paint(self, well_id: int) -> bool:
        """ウェルから絵の具を取り出す"""
        well = self.wells[well_id]
        representative_cell = next(iter(well['cells']))
        i, j = representative_cell
        
        # 1グラム以上あるかチェック
        if well['amount'] < 1.0 - 1e-6:
            return False
        
        # 取り出し
        if well['amount'] >= 1.0:
            well['amount'] -= 1.0
        else:
            well['amount'] = 0.0
        
        # 操作を記録
        self.operations.append(f"2 {i} {j}")
        return True
    
    def create_target_color(self, target_idx: int):
        """目標色を作成"""
        target_color = self.targets[target_idx]
        
        # 使用可能なウェルを取得
        well_id = self.find_available_well()
        self.well_usage[well_id] = True
        
        # 最適なチューブ組み合わせを取得
        combination = self.find_best_tube_combination(target_color)
        
        # 組み合わせに基づいて絵の具を追加
        for tube_idx, count in combination:
            for _ in range(count):
                self.add_paint(well_id, tube_idx)
        
        # 1グラム取り出し
        if not self.extract_paint(well_id):
            # 取り出せない場合は適当に追加して取り出し
            self.add_paint(well_id, 0)
            self.extract_paint(well_id)
        
        # ウェルの使用状況を更新（まだ絵の具が残っていれば使用中のまま）
        if self.wells[well_id]['amount'] < 1e-9:
            self.well_usage[well_id] = False
    
    def solve(self):
        """メイン解法"""
        self.read_input()
        self.initialize_palette()
        self.output_initial_state()
        
        # 各目標色を順番に作成
        for target_idx in range(self.H):
            self.current_target = target_idx
            self.create_target_color(target_idx)
        
        # 操作を出力
        for op in self.operations:
            print(op)

if __name__ == "__main__":
    solver = PaintMixer()
    solver.solve()