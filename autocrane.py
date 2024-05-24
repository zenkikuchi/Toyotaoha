import numpy as np 
from collections import deque
import random
'''
N = int(input())
A = [list(map(int, input().split())) for l in range(N)]
A = np.array(A)
'''
# Read input from input.txt
with open('input.txt', 'r') as file:
    content = file.readlines()

# Parse the content
N = int(content[0].strip())
A = [list(map(int, line.strip().split())) for line in content[1:N+1]]
A = np.array(A)


# 現在控えているコンテナ,盤面に出ているものを含む
inputContainers = A

# 排出されたコンテナ 排出されたら-1 にする
carriedOut = np.arange(25).reshape(5,5)

# 盤面初期化 常にコンテナのnumberか　−１をとる
grid_ground = np.full((N,N),-1)
grid_ground[:, 0] = inputContainers[:, 0]  # inputContainersの一列目

# 盤面初期化 輸送中のgrid 常にコンテナのnumberか　−１をとる
grid_transporting = np.full((N,N),-1)

# 戦略的持ち出しコンテナ
targetContainers = []

# crane クラスを定義
class Crane:
    def __init__(self, craneType, craneId, start_pos, grid_ground, grid_transporting, field):
        self.craneType = craneType  # クレーンの種類
        self.craneId = craneId # unique id 
        self.position = start_pos  # クレーンの初期位置
        self.holding_container = False  # コンテナを保持しているかどうか
        self.commands = []  # クレーンの操作コマンドのリスト
        self.grid_ground = grid_ground
        self.grid_transporting = grid_transporting
        self.route = deque()
        self.field = field
        self.holding_container_number = -1

    def addCommand(self, command):
        self.commands.append(command)

    def move(self, direction):
        old_position = self.position
        if direction == 'U':
            self.position = (self.position[0] - 1, self.position[1])
        elif direction == 'D':
            self.position = (self.position[0] + 1, self.position[1])
        elif direction == 'L':
            self.position = (self.position[0], self.position[1] - 1)
        elif direction == 'R':
            self.position = (self.position[0], self.position[1] + 1)
        self.addCommand(direction)

        #　コンテナをつかんでいる場合はgrid_transportingを更新
        if self.holding_container:
            containernum = self.holding_container_number
            self.grid_transporting[self.position] = containernum
            if self.grid_transporting[old_position] == containernum:
                self.grid_transporting[old_position] = -1

    def pick_up(self):
        self.holding_container = True
        number =  self.grid_ground[self.position[0]][self.position[1]]
        self.holding_container_number = number
        self.grid_transporting[self.position[0]][self.position[1]] = number
        self.grid_ground[self.position[0]][self.position[1]] = -1
        
        self.addCommand('P')

    def put_down(self):
        self.holding_container = False
        number =  self.grid_transporting[self.position[0]][self.position[1]]
        self.holding_container_number = -1
        self.grid_ground[self.position[0]][self.position[1]] = number
        self.grid_transporting[self.position[0]][self.position[1]] = -1
        self.addCommand('Q')

    def stay(self):
        self.addCommand('.')

    def explode(self):
        self.addCommand('B')
        self.position = (-1, -1)  # effectively removes the crane from the grid
    
    def set_route(self, route, max_jump=100):
        if route:
            if self.position != route[0]:
                additional_route, _ , _ = find_route(self.position, route[0], max_jump=max_jump)
                if additional_route:
                    self.route = deque(additional_route[1:]) 
                    self.route.append('P')
                    self.route.append(route[1:])
                else:
                    # route = false の時stay
                    self.route.append('.')
            else:
                self.route = deque(route)
            
            self.route.append('Q')  # put down command at the end
        else:
            # route = false の時stay
            self.route.append('.')

    def nothing_to_do(self):
        possible_moves = field[self.position[0], self.position[1]]
        if isinstance(possible_moves, list):
            possible_moves = random.choice(possible_moves)
        
        direction = random.choice(possible_moves)
        self.move(direction)
    
    def follow_route(self):
        direction_map = {
            (-1, 0): 'U',
            (1, 0): 'D',
            (0, 1): 'R',
            (0, -1): 'L'
        }
        
        if self.route:
            step = self.route.popleft()
            if isinstance(step, tuple):
                direction = (step[0] - self.position[0], step[1] - self.position[1])
                if direction in direction_map:
                    old_position = self.position
                    self.move(direction_map[direction])
                    if self.holding_container:
                        # コンテナの値を移動先に移す
                        self.grid[self.position] = self.grid[old_position]
                        self.grid[old_position] = -1
                else:
                    #stay
                    self.stay()

            elif step == 'P':
                self.pick_up()
            elif step == 'Q':
                self.put_down()
            elif step == '.':
                self.stay()
            else: 
                print("illigal move!!")
        else:
            self.nothing_to_do()

    
        
# 場の定義
field = np.array([
    ['D','L','L','L','L'],
    ['DR','DR','DR','DR','U'],
    ['DR','DR','DR','DR','U'],
    ['DR','DR','DR','DR','U'],
    ['R','R','R','R','U']
])

# crane 初期化
crane10 = Crane('L', 10, (0, 0), grid_ground,grid_transporting, field)
crane1 = Crane('S', 1, (1, 0), grid_ground,grid_transporting, field)
crane2 = Crane('S', 2, (2, 0), grid_ground,grid_transporting, field)
crane3 = Crane('S', 3, (3, 0), grid_ground,grid_transporting, field)
crane4 = Crane('S', 4, (4, 0), grid_ground,grid_transporting, field)
cranes = [crane10, crane1, crane2, crane3, crane4]

# あるmod５配列のコンテナをすべて引き出す時に出てくる不要なコンテナ数
def pullrowCost(inputContainers, row):
    removingContainer = np.full((N,N), 0)
    for i in range(5):
        num = row*5+i
        currentlocation = np.where(inputContainers == num)
        if currentlocation[0].size > 0:
            for j in range(currentlocation[1][0]):
                if removingContainer[currentlocation[0][0]][j] == 0 and (inputContainers[currentlocation[0][0]][j] % 5) != row: #フラグ確認
                    removingContainer[currentlocation[0][0]][j] = 1
    
    return np.sum(removingContainer)



# 各方向の移動ベクトルの定義
directions = {
    'U': (-1, 0),
    'D': (1, 0),
    'R': (0, 1),
    'L': (0, -1),
    'UR': [(-1, 0), (0, 1)],  # 上または右
    'DR': [(1, 0), (0, 1)],  # 下または右
    'DL': [(1, 0), (0, -1)] # 下または左
}

# obstaclesの座標 初期化
obstacles = (grid_ground != -1)
'''
obstacles = np.array([
[False, False, False, False, False],
[False, False, False, False, False],
[False, True, False, True, False],
[False, False, False, True, False],
[False, True, False, False, False]
])'''

# 幅優先探索でルートを見つける
# スタートとエンドの定義（例: (0,0)から(4,4)まで）start = (1,2 ) end = (2, 4)
# max_jumpは飛び越えを認めるかどうかの引数
# return →path, jamped_cells

def find_route(start, end, max_depth=20, max_jump=0):
    queue = deque([(start, [start], 0, 0, [])])  # (現在の座標, これまでの経路, 深さ, 飛び越えた障害物の数, 飛び越えたマスのリスト)
    visited = set()
    
    while queue:
        (current, path, depth, jumped_obstacles, jumped_cells) = queue.popleft()
        if current == end:
            return path, jumped_cells, len(path) - 1  # 移動回数は経路の長さから1引いた値
        
        if depth >= max_depth:
            continue
        
        if current in visited:
            continue
        
        visited.add(current)
        i, j = current
        
        if field[i][j] in directions:
            moves = directions[field[i][j]]
            if not isinstance(moves, list):
                moves = [moves]
            
            for move in moves:
                ni, nj = i + move[0], j + move[1]
                
                if 0 <= ni < 4  and 0 <= nj < 4:
                    if not obstacles[ni][nj]:
                        queue.append(((ni, nj), path + [(ni, nj)], depth + 1, jumped_obstacles, jumped_cells))
                    elif obstacles[ni][nj] and jumped_obstacles < max_jump:
                        ni2, nj2 = ni + move[0], nj + move[1]
                        if 0 <= ni2 < 4 and 0 <= nj2 < 4 and not obstacles[ni2][nj2]:
                            queue.append(((ni2, nj2), path + [(ni, nj), (ni2, nj2)], depth + 1, jumped_obstacles + 1, jumped_cells + [(ni, nj)]))
    
    return None, None, 0

# 保管場所の定義
warehouse = np.array([
    [np.inf,11, 10, 9, np.inf],
    [np.inf, 0, 1, 2, np.inf],
    [np.inf, 3, 5, 6, np.inf],
    [np.inf, 4, 7 ,8, np.inf],
    [np.inf, 12, 13, 14, np.inf]
])
def find_nth_smallest_empty_warehouse_location(grid_ground, warehouse,n=0):
    empty_locations = []
    
    for i in range(len(warehouse)):
        for j in range(len(warehouse[i])):
            if grid_ground[i][j] == -1:
                empty_locations.append((warehouse[i][j], (i, j)))
    
    empty_locations.sort()
    
    if n < len(empty_locations):
        return empty_locations[n][1]
    else:
        return None

# 盤面ですぐに出荷できるものはあるか判定
def get_first_non_negative_one(row):
    for element in row:
        if element != -1:
            return element
    return None

def detectConnection(grid_ground, carriedOut):
    numbers = []
    for row in carriedOut:
        num = get_first_non_negative_one(row)
        if num is not None and num not in numbers:
            coordinates = np.argwhere(grid_ground == num)
            if coordinates.size > 0:
                numbers.append(num)
    if numbers:
        return numbers
    else:
        return False

# numを入れたら目指すべき座標を返す
def destination(containernum, carriedOut):
    for row in carriedOut:
        num = get_first_non_negative_one(row)
        if num == containernum:
            return [[row],[4]]
    return 


# クレーン間の優先順位決定
def decidePriority(crane1,crane2):
    if crane1.holding_container:
        return crane1.craneId
    elif crane2.holding_container:
        return crane2.craneId
    else:
        max(crane1.craneId, crane2.craneId)

def location(grid,num):
    return np.argwhere(grid == num)

def distination(grid,num,carriedOut):
    return

# 空いているクレーンを検索。id を返す
def getEmptycraneId(cranes):
    ids = [crane.craneId for crane in cranes if not crane.holding_container]
    ids.sort()
    return ids

# 出荷できないとき取り合えず取り出すコンテナ番号のリストを更新する関数
def pull_container(inputContainers, targetContainers):
    if not targetContainers:
        min_cost = float('inf')
        selected_row = -1
        for row in range(5):
            cost = pullrowCost(inputContainers, row)
            if cost < min_cost:
                min_cost = cost
                selected_row = row
        
        targetContainers = [(-1,-1,selected_row, inputContainers[selected_row][i]) for i in range(5) if inputContainers[selected_row][i] != -1]

    new_target_containers = []
    for row, container_num in targetContainers:
        locations = np.argwhere(inputContainers == container_num)
        if locations.size > 0:
            for loc in locations:
                i, j = loc
                left_non_negatives = np.count_nonzero(inputContainers[i][:j] != -1)
                new_target_containers.append((i, j, left_non_negatives, container_num))
                
    return new_target_containers


# クレーンルート修正
def adjust_containers(cranes):
    # Get the next position for each crane based on its route
    next_positions = {}
    for crane in cranes:
        if crane.route:
            next_step = crane.route[0]
            if isinstance(next_step, tuple):
                next_position = next_step
            else:
                next_position = crane.position
            next_positions[crane.craneId] = next_position
    
    # Identify conflicts
    position_to_crane_ids = {}
    for crane_id, position in next_positions.items():
        if position in position_to_crane_ids:
            position_to_crane_ids[position].append(crane_id)
        else:
            position_to_crane_ids[position] = [crane_id]
    
    # Resolve conflicts
    for position, crane_ids in position_to_crane_ids.items():
        if len(crane_ids) > 1:
            # There is a conflict, decide priority
            prioritized_crane_id = crane_ids[0]
            for crane_id in crane_ids[1:]:
                prioritized_crane_id = decidePriority(cranes[prioritized_crane_id], cranes[crane_id])
            
            # Adjust routes for non-prioritized cranes
            for crane_id in crane_ids:
                if crane_id != prioritized_crane_id:
                    cranes[crane_id].route.appendleft('.')
                    
                    # Check if other cranes are planning to move to the non-prioritized crane's current position
                    non_prioritized_position = cranes[crane_id].position
                    for other_crane_id, other_crane_next_position in next_positions.items():
                        if other_crane_next_position == non_prioritized_position:
                            cranes[other_crane_id].route.appendleft('.')
                            next_positions[other_crane_id] = other_crane_next_position

    
def execute_turn(grid_ground, inputContainers, carriedOut, cranes, turn):
    # Step 1: Update the grid for incoming containers
    for i in range(5):
        for j in range(5):
            if inputContainers[i][j] != -1 and grid_ground[i][0] == -1:
                grid_ground[i][0] = inputContainers[i][j]
                inputContainers[i][j] = -1
        
    # Step 2: クレーんの動きの決定
    # Step 2-1 クレーンの挙動の決定
    # 空いているクレーンを検索
    empty_crane_ids = getEmptycraneId(cranes)
    empty_smallcrane_ids = []
    is_large_crane_available = False

    for crane_id in empty_crane_ids:
        if crane_id != 0:
            empty_smallcrane_ids.append(crane_id)
        else:
            is_large_crane_available = True

    
    # 出荷できるコンテナを検索 ある：numberのリスト なし：False
    number_of_containers = detectConnection(grid_ground, carriedOut)
    if number_of_containers:
        # 出荷できるコンテナあり
        no_jump_containers = []
        jump_containers = []
        recessed_containers = []

        # 出荷ルート検索　jumpなし；no_jump_containers jump1回；jump_containers =　jamp一回以上：recessed_containers
        for num in number_of_containers:
            loc = location(grid_ground, num)
            dis = destination(grid_ground, num)
            
            route, jumped_cells, _ = find_route((loc[0][0], loc[0][1]), (dis[0][0], dis[0][1]), max_jump=1)
        
            if jumped_cells[0].size == 0:
                no_jump_containers.append((num, route))
            elif len(jumped_cells) == 1:
                jump_containers.append((num, jumped_cells, route))
            else:
                recessed_containers.append((num, jumped_cells, route))

        # 空いているクレーンに対して no_jump_containersを優先的に処理
        for i in range(min(len(no_jump_containers), len(empty_crane_ids))):
            num, route = no_jump_containers[i]
            assigned_crane = cranes[empty_crane_ids[i]]
            if assigned_crane.craneType == 'L':
                is_large_crane_available = False
            assigned_crane.set_route(route)

        # 残りのクレーンに対して jump_containers コンテナを処理
        remaining_crane_ids = empty_crane_ids[len(no_jump_containers):]
        for i in range(min(len(jump_containers), len(remaining_crane_ids))):
            num, jumped_cells, route = jump_containers[i]
            assigned_crane = cranes[remaining_crane_ids[i]]
            for k in range(3):
                routetoremove, jumped_cellstoremove, _ = find_route(jumped_cells, find_nth_smallest_empty_warehouse_location(grid_ground, warehouse, k), max_jump=0)
                if jumped_cellstoremove[0].size == 0:
                    # 優先順位３以内の保管場所にjump するコンテナがコンテナを跨ぐことなく保管場所に到達できる場合
                    assigned_crane.set_route(routetoremove[1:])
            else:
                if is_large_crane_available:
                    is_large_crane_available = False
                    crane10.set_route(route)

        # 奥まったコンテナに対して
        if recessed_containers.size > 0:
            if is_large_crane_available:
                    is_large_crane_available = False
                    crane10.set_route(route)
    
    else:
        # 出荷できるコンテナなし
        global targetContainers
        targetContainers = pull_container(inputContainers, targetContainers)

        if targetContainers:
            for crane_id in empty_crane_ids:
                if not targetContainers:
                    break
                assigned_crane = cranes.pop(0)
                row, j, left_non_negatives, container_num = targetContainers.pop(0)

                if left_non_negatives == 0 and grid_ground[row][0] == container_num:
                    loc = (row, 0)
                    destination_loc = (find_nth_smallest_empty_warehouse_location(grid_ground, warehouse, 0))
                    route, _, _ = find_route(loc, destination_loc)
                    assigned_crane.set_route(route)

    # Step 2-2　クレーンルート調整
    adjust_containers(cranes)
    
    # Step 2-3 クレーン動かす
    for crane in cranes:
        crane.follow_route()

    # Step 3: Remove containers from the grid if they are at the exit points
    for i in range(len(grid_ground)):
        if grid_ground[i][4] != -1:
            carriedOut[i][grid_ground[i][4] % 5] = -1
            grid_ground[i][4] = -1

def detectTermination(grid_ground):
    return np.all(grid_ground == -1)

for i in range(10000):
    execute_turn(grid_ground, inputContainers, carriedOut, cranes, i)
    if detectTermination(grid_ground):
        break

outputs = [crane.commands for crane in cranes]

for output in outputs:
    print(''.join(output))