import sys
import math
from typing import List, Tuple, Set

# Constants for floating point comparisons
EPS = 1e-9 
TAKE_THRESHOLD = 1.0 - 1e-6 

class PaintMixer:
    def __init__(self):
        self.N = 0
        self.K = 0  # Number of tube paints
        self.H = 0
        self.T = 0
        self.D = 0
        
        self.tubes: List[Tuple[float, float, float]] = []
        self.targets: List[Tuple[float, float, float]] = []
        
        self.v_walls: List[List[int]] = []
        self.h_walls: List[List[int]] = []
        self.wells = {} 
        self.cell_to_well = {} 
        
        # New: Store well configurations (size and bounds)
        self.well_configs: List[Tuple[int, int, int, int, int]] = [] # (well_id, start_i, end_i, start_j, end_j)
        self.total_wells = 0
        
        self.operations: List[str] = []
        self.turns_used = 0
        self.paint_taken_count = 0
        
        self.well_usage: List[bool] = [] 
        self.well_last_target_idx: List[int] = [] 

        self.EPS = EPS
        self.TAKE_THRESHOLD = TAKE_THRESHOLD
        self.REUSE_DIST_THRESHOLD = 0.15 # Fixed threshold for reusing wells

    def read_input(self):
        line = sys.stdin.readline().split()
        self.N, self.K, self.H, self.T, self.D = map(int, line)
        
        # Initialize well usage and last_target_idx arrays after total_wells is determined
        # This will be done in initialize_palette now.
        
        for _ in range(self.K):
            c, m, y = map(float, sys.stdin.readline().split())
            self.tubes.append((c, m, y))
        
        for _ in range(self.H):
            c, m, y = map(float, sys.stdin.readline().split())
            self.targets.append(tuple(map(float, sys.stdin.readline().split())))
    
    # get_well_bounds will no longer be fixed to well_size. Instead, it will retrieve from well_configs
    def get_well_bounds(self, well_id: int) -> Tuple[int, int, int, int]:
        for config in self.well_configs:
            if config[0] == well_id:
                return config[1], config[2], config[3], config[4]
        raise ValueError(f"Well ID {well_id} not found in configurations.")

    def initialize_palette(self):
        self.v_walls = [[0 for _ in range(self.N - 1)] for _ in range(self.N)]
        self.h_walls = [[0 for _ in range(self.N)] for _ in range(self.N - 1)]
        
        current_well_id = 0

        # Region A: (0,0) to (11,11) -> 3x3 wells
        region_A_size = 3
        for r_offset in range(0, 12, region_A_size):
            for c_offset in range(0, 12, region_A_size):
                start_i, end_i = r_offset, r_offset + region_A_size
                start_j, end_j = c_offset, c_offset + region_A_size
                self.well_configs.append((current_well_id, start_i, end_i, start_j, end_j))
                current_well_id += 1

        # Region B: (0,12) to (11,19) -> 4x4 wells
        region_B_size = 4
        for r_offset in range(0, 12, region_B_size):
            for c_offset in range(12, 20, region_B_size):
                start_i, end_i = r_offset, r_offset + region_B_size
                start_j, end_j = c_offset, c_offset + region_B_size
                self.well_configs.append((current_well_id, start_i, end_i, start_j, end_j))
                current_well_id += 1
        
        # Region C: (12,0) to (19,19) -> 4x4 wells
        region_C_size = 4
        for r_offset in range(12, 20, region_C_size):
            for c_offset in range(0, 20, region_C_size):
                start_i, end_i = r_offset, r_offset + region_C_size
                start_j, end_j = c_offset, c_offset + region_C_size
                self.well_configs.append((current_well_id, start_i, end_i, start_j, end_j))
                current_well_id += 1
        
        self.total_wells = current_well_id
        self.well_usage = [False] * self.total_wells
        self.well_last_target_idx = [-1] * self.total_wells

        # Build wells and place walls based on configurations
        for well_id, start_i, end_i, start_j, end_j in self.well_configs:
            well_cells = set()
            for r_idx in range(start_i, end_i):
                for c_idx in range(start_j, end_j):
                    well_cells.add((r_idx, c_idx))
                    self.cell_to_well[(r_idx, c_idx)] = well_id
            
            self.wells[well_id] = {'cells': well_cells, 'color': (0.0, 0.0, 0.0), 'amount': 0.0}
            
            # Place walls at well boundaries
            # Vertical walls to the right
            for r_idx in range(start_i, end_i):
                if end_j < self.N: 
                    self.v_walls[r_idx][end_j - 1] = 1 

            # Horizontal walls below
            for c_idx in range(start_j, end_j):
                if end_i < self.N: 
                    self.h_walls[end_i - 1][c_idx] = 1 

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
        if op_type == 2: # TAKE_PAINT
            self.paint_taken_count += 1
        return True

    def color_distance(self, c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> float:
        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

    def mix_colors(self, c1: Tuple[float, float, float], v1: float, 
                   c2: Tuple[float, float, float], v2: float) -> Tuple[float, float, float]:
        if abs(v1) < self.EPS and abs(v2) < self.EPS: return (0.0, 0.0, 0.0)
        if abs(v1) < self.EPS: return c2
        if abs(v2) < self.EPS: return c1
        total_v = v1 + v2
        if abs(total_v) < self.EPS: return (0.0, 0.0, 0.0)
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
        
        # Crucial change: max_well_capacity is now based on number of cells
        max_well_capacity = float(len(well['cells'])) 
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
        start_well_search_idx = target_idx % self.total_wells 

        # Priority 1: Completely empty and unused well
        for i in range(self.total_wells):
            well_id = (start_well_search_idx + i) % self.total_wells
            if not self.well_usage[well_id] and self.wells[well_id]['amount'] < self.EPS:
                return well_id, "empty"
        
        # Priority 2: Reusable well (enough paint, close to target, not in use)
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

        # Priority 3: Clear a well. Choose one that was used least recently (LRU)
        chosen_clear_well_id = -1
        least_recent_target_idx = self.H + 1 
        
        for i in range(self.total_wells):
            well_id = (start_well_search_idx + i) % self.total_wells
            if not self.well_usage[well_id]: # Not currently active
                if self.well_last_target_idx[well_id] < least_recent_target_idx:
                    least_recent_target_idx = self.well_last_target_idx[well_id]
                    chosen_clear_well_id = well_id
        
        if chosen_clear_well_id != -1:
             return chosen_clear_well_id, "clear_and_use" 
        else:
             # Fallback: All wells are currently marked as in_use (should be rare)
             return start_well_search_idx, "force_clear_and_use"

    def allocate_op1_tubes_greedy(self, well_id: int, target_color: Tuple[float, float, float], num_op1_budget: int, 
                                  sim_current_color: Tuple[float, float, float], sim_current_amount: float) -> List[int]:
        
        max_well_capacity = float(len(self.wells[well_id]['cells'])) # Capacity by cell count
        tubes_to_add_sequence = []

        for _ in range(num_op1_budget):
            if sim_current_amount >= max_well_capacity - self.EPS:
                break 

            best_tube_idx_to_add = -1
            min_resulting_dist = float('inf')

            for t_idx, tube_c_data_candidate in enumerate(self.tubes):
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
                sim_add_amount_actual = min(1.0, max_well_capacity - sim_current_amount)
                sim_current_color = self.mix_colors(sim_current_color, sim_current_amount, 
                                                     self.tubes[best_tube_idx_to_add], sim_add_amount_actual)
                sim_current_amount += sim_add_amount_actual
            else:
                break 
        return tubes_to_add_sequence


    def create_target_color(self, target_idx: int):
        target_color = self.targets[target_idx]

        # --- Turn Budget Planning ---
        remaining_targets_to_make = self.H - self.paint_taken_count 
        turns_left_total = self.T - self.turns_used
        
        if turns_left_total <= 0 or remaining_targets_to_make <= 0:
            return 

        budget_this_target_ops = 1 # Min 1 turn for Op2
        
        available_for_non_op2 = turns_left_total - remaining_targets_to_make
        if available_for_non_op2 > 0 and remaining_targets_to_make > 0:
            budget_this_target_ops += math.floor(available_for_non_op2 / remaining_targets_to_make)
        
        if target_idx == self.H - 1: # Last target can use all remaining turns
            budget_this_target_ops = turns_left_total
        
        budget_this_target_ops = max(1, budget_this_target_ops) 
        budget_this_target_ops = min(budget_this_target_ops, turns_left_total) 

        # --- Well Selection and Preparation ---
        well_id, well_state_type = self.find_available_well_for_target(target_idx)
        turns_spent_locally = 0 

        if well_state_type in ["clear_and_use", "force_clear_and_use"]:
            min_ops_after_clear = 1 # For take
            if self.wells[well_id]['amount'] < self.TAKE_THRESHOLD or well_state_type == "force_clear_and_use": 
                min_ops_after_clear +=1 # Need at least one Op1 after clearing an empty/bad well

            if budget_this_target_ops - turns_spent_locally >= (1 + min_ops_after_clear): # 1 for clear
                if self.clear_well(well_id):
                    turns_spent_locally += 1
                else: return 
            else:
                return 
        
        self.well_usage[well_id] = True 

        # --- Paint Addition Operations (Op1) ---
        num_op1_allowed = budget_this_target_ops - turns_spent_locally - 1 # -1 for Op2
        num_op1_allowed = max(0, num_op1_allowed) 

        tubes_to_add = self.allocate_op1_tubes_greedy(well_id, target_color, num_op1_allowed,
                                                      self.wells[well_id]['color'], self.wells[well_id]['amount'])
        
        for tube_idx_to_add in tubes_to_add:
            if self.add_paint(well_id, tube_idx_to_add):
                turns_spent_locally += 1
            else: 
                break 
        
        # --- Paint Extraction (Op2) ---
        if budget_this_target_ops - turns_spent_locally >= 1: # At least 1 turn for Op2
            if self.wells[well_id]['amount'] < self.TAKE_THRESHOLD:
                 if budget_this_target_ops - turns_spent_locally >= 2: # Need 1 more turn for add
                    if self.wells[well_id]['amount'] < float(len(self.wells[well_id]['cells'])) - self.EPS: # Capacity check
                        if self.add_paint(well_id, 0): # Emergency add with tube 0
                            turns_spent_locally += 1
            
            if budget_this_target_ops - turns_spent_locally >= 1:
                 if self.extract_paint(well_id):
                    turns_spent_locally +=1 
                 else:
                    pass 
        else:
            pass 

        self.well_usage[well_id] = False 
        self.well_last_target_idx[well_id] = target_idx 

    def solve(self):
        self.read_input()
        self.initialize_palette()
        self.output_initial_state()
        
        for target_idx_loop in range(self.H):
            if self.turns_used + (self.H - self.paint_taken_count) > self.T:
                 break 
            self.create_target_color(target_idx_loop)
        
        emergency_well_idx_counter = 0
        while self.paint_taken_count < self.H and self.turns_used < self.T:
            current_well_id_emergency = emergency_well_idx_counter % self.total_wells
            
            turns_for_this_emergency = 1 
            needs_add = self.wells[current_well_id_emergency]['amount'] < self.TAKE_THRESHOLD
            needs_clear = self.wells[current_well_id_emergency]['amount'] > self.EPS and needs_add 

            if needs_add: turns_for_this_emergency += 1 
            if needs_clear: turns_for_this_emergency += 1 
            
            if self.T - self.turns_used < turns_for_this_emergency:
                break 

            if needs_clear:
                self.clear_well(current_well_id_emergency)
            
            if self.wells[current_well_id_emergency]['amount'] < self.TAKE_THRESHOLD:
                 self.add_paint(current_well_id_emergency, 0) 
            
            self.extract_paint(current_well_id_emergency)
            
            if self.wells[current_well_id_emergency]['amount'] < self.EPS:
                 self.well_usage[current_well_id_emergency] = False 
            
            self.well_last_target_idx[current_well_id_emergency] = self.H # Mark as used by emergency
            emergency_well_idx_counter += 1

        for op_str_final in self.operations:
            print(op_str_final)
        sys.stdout.flush()

if __name__ == "__main__":
    solver = PaintMixer()
    solver.solve()