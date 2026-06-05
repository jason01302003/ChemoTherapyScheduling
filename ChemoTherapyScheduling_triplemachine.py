import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import time
import random
from gurobipy import *

# 參數
P = 20            # 病人數
L = 28            # 相對治療日數
T = 40            # 規劃天數
N_nurses = 1       # 護士人數
prob = 0.3
# 設定時間，並切分為slots
slot_length = 15
work_start = 9 * 60
max_clock = 8 * 60
# 定義午休時段
lunch_start_min = 12 * 60          # 12:00
lunch_end_min = 13 * 60          # 13:00

# 轉換成 slot 索引 -> 也就是 slot 12, 13, 14, 15 是禁止護士工作的時段
lunch_start_slot = (lunch_start_min - work_start) // slot_length  # = 12
lunch_end_slot = (lunch_end_min - work_start) // slot_length  # = 16

# 總共的 slot 數量
n_slots = max_clock // slot_length

random.seed(17)
fname = rf"C:\Users\jason\Desktop\OTA_paper\ChemoTherapyScheduling\SCP\SCP" + \
    str(P)+"-"+str(1)+".txt"
path = fname
f = open(path, 'w')

# 生成測試資料
V = []
Last_Position = []

# V[i][j]為 病人 i 在療程開始後第 j 天是否需要治療
for i in range(P):
    treatment_list = []
    last = -1
    for j in range(L):
        if random.random() < prob:
            treatment_list.append(1)
            last = j
        else:
            treatment_list.append(0)
    V.append(treatment_list)
    Last_Position.append(last) #紀錄病人 i 最後一次治療出現在療程開始後第幾天
# 隨機每日最大容量，要隨P調整，模擬實際醫院每天的治療容量可能不同(門診、排班)
K = [random.randint(5 * N_nurses, 10 * N_nurses) for _ in range(T)]

# Pattern生成和對照
Pattern = []

for i in range(P):
    pattern_list = [
        random.choices([1, 2], weights=[0.6, 0.4])[
            0],                     # task1
        random.choices([1, 2, 3, 4, 5, 6, 7],
                       weights=[0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05])[0],  # task2
        random.choices([1, 2], weights=[0.7, 0.3])[
            0],                     # task3
        random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[
            0],             # task4
        random.choices([1, 2], weights=[0.7, 0.3])[
            0]                      # task5
    ]
    Pattern.append(pattern_list)

# task1: 打針，task2: 打點滴，task3: 拔針，task4:患者休息， task5:批價，出院手續
task_time_map = {
    "task1": {"1": 15, "2": 30},
    "task2": {"1": 150, "2": 165, "3": 180, "4": 195, "5": 210, "6": 225, "7": 240},
    "task3": {"1": 15, "2": 30},
    "task4": {"1": 15, "2": 30, "3": 45},
    "task5": {"1": 15, "2": 30}
}

tasks = ["task1", "task2", "task3", "task4", "task5"]

# 輸出生成資料
print(f"Daily QTY:\n", file=f)
for j in range(T):
    print(K[j], end="", file=f)
print("\n", file=f)

for i in range(P):
    print(f"Patient {i:2d}: ", end="", file=f) #輸出每位病人的治療需求序列 V[i]

    for j in range(L):
        print(V[i][j], end="", file=f)
    print("\n", file=f)
    print(f"Pattern  {i:2d}: ", end="", file=f)

    for j in range(5):
        print(Pattern[i][j], end="", file=f)
    print("\n", file=f)

# 預處理

# 判斷護士工作是否與午休衝突的函數
# 午休時段，護士不能工作，所以如果病人的 task1、task3、task5 的任何部分落在午休時段，就不允許從該 slot 開始治療
def nurse_tasks_overlap_lunch(q, dslots, lunch_start_slot, lunch_end_slot):
    t1_start = q
    t3_start = q + dslots[0] + dslots[1]
    t5_start = q + dslots[0] + dslots[1] + dslots[2] + dslots[3]

    # 只檢查 task1、task3、task5
    for start, length in [(t1_start, dslots[0]),
                          (t3_start, dslots[2]),
                          (t5_start, dslots[4])]:
        for s in range(start, start + length):
            if lunch_start_slot <= s < lunch_end_slot:
                return True
    return False


durations_minutes = [] #病人 i 的 5 個 task 各需要幾分鐘
durations_slots = [] #病人 i 的 5 個 task 各需要幾個 slot
total_slots = [] #病人 i 一次治療總共需要幾個 slot
feasible_start_slots = {} #病人 i 可以從哪些 slot 開始治療

#根據病人的 Pattern 找出 各個task 的分鐘數。
for i in range(P):
    dmins = [task_time_map["task1"][str(Pattern[i][0])],
             task_time_map["task2"][str(Pattern[i][1])],
             task_time_map["task3"][str(Pattern[i][2])],
             task_time_map["task4"][str(Pattern[i][3])],
             task_time_map["task5"][str(Pattern[i][4])]]
    durations_minutes.append(dmins)
    dslots = []

    #將分鐘數轉換成需要的 slot 數量，並取整數（向上取整，確保有足夠的時間）
    for d in dmins:
        dslots.append((d + slot_length - 1)//slot_length)
    durations_slots.append(dslots)
    total_slots.append(sum(dslots))


    # 找病人 i "最晚"可以從哪些 slot 開始，且整個治療能在上班時間內完成。
    feasible_start_slots[i] = [
        q for q in range(0, n_slots)
        if q + sum(dslots) <= n_slots
        and not nurse_tasks_overlap_lunch(q, dslots, lunch_start_slot, lunch_end_slot)
    ]

# 紀錄病人在那些slot會需要護士
nurse_occ = {}
for i in range(P):
    dslots = durations_slots[i]
    for q in feasible_start_slots[i]:
        # 明確計算每個 task 的起始 slot
        t1_start = q
        t3_start = q + dslots[0] + dslots[1]
        t5_start = q + dslots[0] + dslots[1] + dslots[2] + dslots[3]

        occ = set()
        for s in range(t1_start, t1_start + dslots[0]):
            occ.add(s)  # task1
        for s in range(t3_start, t3_start + dslots[2]):
            occ.add(s)  # task3
        for s in range(t5_start, t5_start + dslots[4]):
            occ.add(s)  # task5

        occ = {s for s in occ if 0 <= s < n_slots}
        nurse_occ[(i, q)] = occ

# 建模
CTS = Model("ChemoTherapyScheduling")
CTS.setParam('OutputFlag', 1) #輸出求解過程
CTS.setParam('TimeLimit', 1800)
CTS.setParam('MIPGap', 0.01) #允許 1% gap。也就是不一定要找到絕對最佳解，只要夠接近即可。
CTS.setParam('Heuristics', 0.5)
CTS.setParam('Threads', 6)

# 定義決策變數 X、Y & MaxSize
X = CTS.addVars(P, T, vtype=GRB.BINARY, name="X") #病人 i 在第 t 天開始療程
Y = CTS.addVars(P, T, vtype=GRB.BINARY, name="Y") #病人 i 在第 t 天有治療
MaxSize = CTS.addVar(vtype=GRB.INTEGER, lb=0, name="MaxSize")  # 每天最多病人數

# 定義決策變數 Z, Z[i,t,q],病人 i 在第 t 天治療，且從 slot q 開始治療
Z = {}
for i in range(P):
    for t in range(T):
        for q in feasible_start_slots[i]:
            for n in range(N_nurses):
                Z[(i, t, q, n)] = CTS.addVar(vtype=GRB.BINARY,
                                             name=f"Z_{i}_{t}_{q}_{n}")  # 病人 i 在 t 天從 slot q 開始治療，並且由nurse n 照顧
CTS.update()


# 限制式
# 約束 (A) 只開始一次
for i in range(P):
    if Last_Position[i] == -1:
        for t in range(T):
            CTS.addConstr(X[i, t] == 0)  # 病人 i 在第 t 天開始療程
        continue
    # 計算最晚可以開始療程的日期
    max_offset = Last_Position[i]
    max_start = T - 1 - max_offset
    CTS.addConstr(quicksum(X[i, s] for s in range(
        max_start + 1)) == 1, f"Start_Once_{i}")
    
     # 新增:超過最晚開始日的部分沒有被固定為 0，會讓模型在求解過程中多一些彈性，可能會找到更好的解。
    for s in range(max_start + 1, T):
        CTS.addConstr(X[i, s] == 0, f"Invalid_Start_{i}_{s}")

# 約束 (B) Y >= X 若已經開始治療，後面幾天要跟著治療
# 由開始日 X 推出治療日 Y =>用途: 把「療程相對日期」轉換成「實際排程日期」。
for i in range(P):
    if Last_Position[i] == -1:
        for t in range(T):
            CTS.addConstr(Y[i, t] == 0)  # 病人 i 在第 t 天有治療
        continue
    max_offset = Last_Position[i]
    max_start = T - 1 - max_offset
    
    for s in range(max_start + 1):
        for j, val in enumerate(V[i]):
            if val == 1:
                day = s + j
                if day < T:
                    CTS.addConstr(Y[i, day] >= X[i, s])

# 約束 (C) 每日最大容量限制
for t in range(T):
    CTS.addConstr(quicksum(Y[i, t] for i in range(P))
                  <= MaxSize,   f"Max_Daily_Size_{t}")
    CTS.addConstr(quicksum(Y[i, t] for i in range(P)) <=
                  K[t],      f"Actual_Daily_Size_{t}")

# 約束 (D) Z and Y link 如果有治療，則一定要選一個開始時間
for i in range(P):
    for t in range(T):
        zs = [Z[(i, t, q, n)]
              for q in feasible_start_slots[i]
              for n in range(N_nurses)]
        if zs:
            CTS.addConstr(quicksum(zs) == Y[i, t])
        else:
            CTS.addConstr(Y[i, t] == 0)

# 衝突懲罰，護士衝突量 = 同一時段需要同一位護士的病人數 - 1 (如果有衝突才會大於0)
# Conflict[t,s] 代表在第 t 天的第s個 slot，有多少病人需要護士，扣掉1之後就是衝突量。
Conflict = {}
for t in range(T):
    for s in range(n_slots):
        Conflict[(t, s)] = CTS.addVar(
            vtype=GRB.CONTINUOUS, lb=0, name=f"Conflict_{t}_{s}")  # 該時段護理師衝突量

# 計算每個時段的衝突量 → 如果同一時段有多於 1 位病人需要護士，則產生衝突
for t in range(T):
    for s in range(n_slots):
        terms = []
        for i in range(P):
            for q in feasible_start_slots[i]:
                if s in nurse_occ[(i, q)]:
                    for n in range(N_nurses):
                        terms.append(Z[(i, t, q, n)])
        if terms:
            CTS.addConstr(Conflict[(t, s)] >= quicksum(
                terms) - N_nurses)  # 改為 N_nurses
            CTS.addConstr(Conflict[(t, s)] >= 0)

# 每位護士的個別衝突（同一護士同時服務 2 人才懲罰）
NurseConflict = {}
for t in range(T):
    for s in range(n_slots):
        for n in range(N_nurses):
            NurseConflict[(t, s, n)] = CTS.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"NC_{t}_{s}_{n}")
            
# 午休時間護士不能工作
for t in range(T):
    for s in range(n_slots):
        for n in range(N_nurses):
            terms = [Z[(i, t, q, n)]
                     for i in range(P)
                     for q in feasible_start_slots[i]
                     if s in nurse_occ[(i, q)]]
            if terms:
                CTS.addConstr(NurseConflict[(t, s, n)] >= quicksum(terms) - 1)
                CTS.addConstr(NurseConflict[(t, s, n)] >= 0)

# 午休時間護士不能工作
for t in range(T):
    for s in range(lunch_start_slot, lunch_end_slot):
        for n in range(N_nurses):
            terms = [Z[(i, t, q, n)]
                     for i in range(P)
                     for q in feasible_start_slots[i]
                     if s in nurse_occ[(i, q)]]
            if terms:
                CTS.addConstr(quicksum(terms) == 0,
                              f"NurseLunch_{t}_{s}_{n}")

# 目標函數
nurse_penalty = 4   # 同一護士同時服務 2 人的懲罰（設較高)
total_penalty = 2   # 護士都忙還有第 3 人的懲罰

CTS.setObjective(
    MaxSize
    + nurse_penalty * quicksum(NurseConflict.values())
    + total_penalty * quicksum(Conflict.values()),
    GRB.MINIMIZE
)

# 求解
starttime = time.time()
CTS.optimize()
endtime = time.time()
print("Run time: %2.5s seconds" % (endtime - starttime))
print("Run time: %2.5s seconds" % (endtime - starttime), file=f)

# 確認求解狀態
status_map = {
    GRB.OPTIMAL:    "最優解",
    GRB.TIME_LIMIT: "達到時間限制",
    GRB.INFEASIBLE: "不可行",
    GRB.UNBOUNDED:  "無界",
}
status_msg = status_map.get(CTS.status, f"狀態碼 {CTS.status}")
print(f"求解狀態：{status_msg}，找到解的數量：{CTS.SolCount}")
print(f"求解狀態：{status_msg}，找到解的數量：{CTS.SolCount}", file=f)

if CTS.SolCount == 0:
    print("未找到可行解，程式終止")
    print("未找到可行解，程式終止", file=f)
    f.close()

# 統計衝突資訊
conflict_slots = 0
total_conflict_amount = 0
conflict_day_slots = {}

for t in range(T):
    for s in range(n_slots):
        val = Conflict[(t, s)].X
        if val > 0.5:
            conflict_slots += 1
            total_conflict_amount += val

            if t not in conflict_day_slots:
                conflict_day_slots[t] = 0
            conflict_day_slots[t] += 1

conflict_day_text = ", ".join(
    f"{day}*{slot_count}"
    for day, slot_count in sorted(conflict_day_slots.items())
)

if not conflict_day_text:
    conflict_day_text = "None"

# 每護士個別衝突
nurse_conflict_slots = 0
for t in range(T):
    for s in range(n_slots):
        for n in range(N_nurses):
            if NurseConflict[(t, s, n)].X > 0.5:
                nurse_conflict_slots += 1

# 兩護士都忙的衝突
conflict_slots = 0
total_conflict_amount = 0
for t in range(T):
    for s in range(n_slots):
        val = Conflict[(t, s)].X
        if val > 0.5:
            conflict_slots += 1
            total_conflict_amount += val

print(f"同護士衝突 slot 數：{nurse_conflict_slots}")
print(f"三護士滿載衝突 slot 數：{conflict_slots}")
print(f"三護士滿載衝突總人次：{total_conflict_amount:.0f}")
print(f"衝突天數:{conflict_day_text}")

# 同步寫入 txt 檔
print(f"同護士衝突 slot 數：{nurse_conflict_slots}", file=f)
print(f"三護士滿載衝突 slot 數：{conflict_slots}", file=f)
print(f"三護士滿載衝突總人次：{total_conflict_amount:.0f}", file=f)
print(f"衝突天數:{conflict_day_text}", file=f)

# 生成排程
schedule = {}
for i in range(P):
    schedule[i] = []
    start_day = None
    for s in range(T):
        if X[i, s].X > 0.5:
            start_day = s
            break
    if start_day is None:
        continue
    for t in range(T):
        if Y[i, t].X > 0.5:
            chosen_q = None
            chosen_n = None
            for q in feasible_start_slots[i]:
                for n in range(N_nurses):
                    if Z[(i, t, q, n)].X > 0.5:
                        chosen_q = q
                        chosen_n = n
                        break
                if chosen_q is not None:
                    break
            if chosen_q is None:
                chosen_q = feasible_start_slots[i][0]
                chosen_n = 0
            start_min = work_start + chosen_q * slot_length
            day_tasks = []
            for k_idx, k in enumerate(tasks):
                dur = task_time_map[k][str(Pattern[i][k_idx])]
                end_min = start_min + dur
                day_tasks.append([t, k, start_min, end_min, chosen_n])
                start_min = end_min
            schedule[i].append(day_tasks)

# 印出 schedule
for i in range(P):
    print(f"Patient {i:2d}:", end="", file=f)
    print("\n", end="", file=f)
    print(f"Patient {i:2d}:")
    print("\n")
    for day_tasks in schedule[i]:
        for day, task, s_min, e_min, nurse_id in day_tasks:
            s_hour = s_min // 60
            s_m = s_min % 60
            e_hour = e_min // 60
            e_m = e_min % 60
            print(
                f"  Day {day}: {task} {s_hour:02d}:{s_m:02d} - {e_hour:02d}:{e_m:02d}", end="", file=f)
            print("\n", end="", file=f)
            print(
                f"  Day {day}: {task} {s_hour:02d}:{s_m:02d} - {e_hour:02d}:{e_m:02d}")
    print()
f.close()

# 輸出甘特圖
# 雙護士顏色配置：同 task 同色系，深淺區分護士
task_colors = {
    # Nurse 0 藍色為主色系
    0: {
        "task1": "#BDE3FF",  # 淺藍
        "task2": "#FFB6C1",  # 粉紅（不需護士，兩位護士同色）
        "task3": "#5DADE2",  # 亮藍
        "task4": "#BD6162",  # 亮紅（不需護士，兩位護士同色）
        "task5": "#1F77B4",  # 深藍
    },
    # Nurse 1 綠色為主色系
    1: {
        "task1": "#A5F8A8",  # 淺綠
        "task2": "#FFB6C1",  # 粉紅（同上，不需護士）
        "task3": "#2EC153",  # 中深綠
        "task4": "#BD6162",  # 亮紅（同上，不需護士）
        "task5": "#0A6B22",  # 極深綠
    },
    # Nurse 2 黃色為主色系
    2: {          # 新增
        "task1": "#F9E79F",  # 淺黃
        "task2": "#FFB6C1",
        "task3": "#F1C40F",  # 橙
        "task4": "#BD6162",
        "task5": "#9A7D0A",  # 深棕
    }
}

output_folder = r"C:\Users\jason\Desktop\OTA_paper\ChemoTherapyScheduling\gantt_days_20_1"
os.makedirs(output_folder, exist_ok=True)

for day in range(T):
    fig, ax = plt.subplots(figsize=(12, 6))
    patient_indices = []
    y_labels = []

    # 只收集當天有治療的病人
    for i in range(P):
        for day_tasks in schedule[i]:
            if day_tasks[0][0] == day:
                patient_indices.append(i)
                y_labels.append(f"Patient {i}")
                break

    if not patient_indices:
        plt.close(fig)
        continue

    # 繪製甘特圖
    nurse_colors = {0: "skyblue", 1: "lightgreen"}  # 護士 0 藍色，護士 1 綠色

    # 繪製甘特圖
    nurse_colors = {0: "skyblue", 1: "lightgreen"}  # 護士 0 藍色，護士 1 綠色

    for pos, i in enumerate(patient_indices):
        for day_tasks in schedule[i]:
            if day_tasks[0][0] == day:

                for _, task, s_min, e_min, nurse_id in day_tasks:

                    current = s_min
                    while current < e_min:
                        next_t = min(current + slot_length, e_min)
                        t_slot = (current - work_start) // slot_length

                        has_conflict = False
                        if 0 <= t_slot < n_slots and task in ["task1", "task3", "task5"]:
                            # 檢查這位護士自己的衝突
                            nurse_conflict = NurseConflict[(
                                day, t_slot, nurse_id)].X > 0.5
                            # 檢查雙護士滿載衝突
                            total_conflict = Conflict[(day, t_slot)].X > 0.5

                            if nurse_conflict or total_conflict:
                                has_conflict = True

                        # 有衝突標橘色
                        if has_conflict:
                            color = "orange"
                        else:
                            color = task_colors[nurse_id][task]

                        ax.barh(y=pos, width=next_t - current,
                                left=current, height=0.4,
                                color=color, edgecolor="black")
                        current = next_t

    # 午休區域標記（畫在所有病人的y上）
    lunch_start_min_abs = work_start + lunch_start_slot * slot_length  # 轉回分鐘
    lunch_end_min_abs = work_start + lunch_end_slot * slot_length

    ax.axvspan(
        lunch_start_min_abs, lunch_end_min_abs,
        color="lightgray", alpha=0.5, label="Lunch Break"
    )
    # 設定甘特圖座標軸與圖例
    # y 軸
    ax.set_yticks(range(len(patient_indices)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()

    # x 軸格式
    ticks = list(range(work_start, work_start + max_clock + 1, slot_length))
    labels = [
        f"{(t - work_start)//60 + 9}:{(t - work_start)%60:02d}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)

    # 圖例
    patches = [
        mpatches.Patch(color="#BDE3FF", label="Nurse 0 - task1"),
        mpatches.Patch(color="#5DADE2", label="Nurse 0 - task3"),
        mpatches.Patch(color="#1F77B4", label="Nurse 0 - task5"),
        mpatches.Patch(color="#A5F8A8", label="Nurse 1 - task1"),
        mpatches.Patch(color="#2EC153", label="Nurse 1 - task3"),
        mpatches.Patch(color="#0A6B22", label="Nurse 1 - task5"),
        mpatches.Patch(color="#F9E79F", label="Nurse 2 - task1"),
        mpatches.Patch(color="#F1C40F", label="Nurse 2 - task3"),
        mpatches.Patch(color="#9A7D0A", label="Nurse 2 - task5"),
        mpatches.Patch(color="#FFB6C1", label="task2 (no nurse)"),
        mpatches.Patch(color="#BD6162", label="task4 (no nurse)"),
        mpatches.Patch(color="orange",   label="Nurse conflict"),
        mpatches.Patch(color="lightgray", label="Lunch Break")
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(f"Chemotherapy Schedule - Day {day}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Patients")

    plt.tight_layout()
    filename = os.path.join(output_folder, f"Day_{day:02d}.png")
    plt.savefig(filename)
    plt.close(fig)

print(f"甘特圖自動生成並存至 {output_folder} 資料夾")



# =========================
# 輸出總覽甘特圖：Day00 ~ Day39
# 每個 subplot 是一天
# X 軸為時間，Y 軸為 Patient1 ~ Patient20
# =========================

overview_filename = os.path.join(output_folder, "Overview_All_Days.png")

cols = 4
rows = (T + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 4), sharex=True, sharey=True)
axes = axes.flatten()

for day in range(T):
    ax = axes[day]

    # 畫每位病人在當天的排程
    for i in range(P):
        for day_tasks in schedule[i]:
            if day_tasks[0][0] == day:

                y_pos = P - 1 - i  # 讓 Patient 1 在上方，Patient 20 在下方

                for _, task, s_min, e_min, nurse_id in day_tasks:

                    current = s_min
                    while current < e_min:
                        next_t = min(current + slot_length, e_min)

                        # 計算目前時間屬於哪一個 slot
                        t_slot = (current - work_start) // slot_length

                        # 判斷是否有護士衝突
                        has_conflict = False
                        if 0 <= t_slot < n_slots:
                            if Conflict[(day, t_slot)].X > 0.5:
                                has_conflict = True

                        # 若是護士 task 且有衝突，標成橘色
                        if has_conflict and task in ["task1", "task3", "task5"]:
                            color = "orange"
                        else:
                            color = task_colors[nurse_id][task]

                        ax.barh(
                            y=y_pos,
                            width=next_t - current,
                            left=current,
                            height=0.55,
                            color=color,
                            edgecolor="black",
                            linewidth=0.3
                        )

                        current = next_t

                break

    ax.set_title(f"Day {day:02d}", fontsize=10)

    # X 軸時間
    ticks = list(range(work_start, work_start + max_clock + 1, 60))
    labels = [f"{t//60:02d}:{t%60:02d}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)

    # Y 軸病人
    ax.set_yticks(range(P))
    ax.set_yticklabels([f"Patient {P - 1 - i}" for i in range(P)], fontsize=8)

    ax.set_xlim(work_start, work_start + max_clock)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

# 多餘的 subplot 關掉
for idx in range(T, len(axes)):
    axes[idx].axis("off")

# 讓 Patient 1 在上方
# for ax in axes[:T]:
#     ax.invert_yaxis()

# 圖例
legend_patches = [
    mpatches.Patch(color=task_colors[0]["task1"], label="task1 (Nurse 0)"),
    mpatches.Patch(color=task_colors[0]["task2"], label="task2 (no nurse)"),
    mpatches.Patch(color=task_colors[0]["task3"], label="task3 (Nurse 0)"),
    mpatches.Patch(color=task_colors[0]["task4"], label="task4 (no nurse)"),
    mpatches.Patch(color=task_colors[0]["task5"], label="task5 (Nurse 0)"),
    mpatches.Patch(color="orange", label="Nurse conflict")
]

fig.legend(
    handles=legend_patches,
    loc="upper center",
    ncol=6,
    bbox_to_anchor=(0.43, 1.01)
)

# 新增:在總覽圖上方加入衝突統計資訊
summary_text = (
    f"conflict slot QTY: {conflict_slots}\n"
    f"conflict total amount: {total_conflict_amount:.0f}\n"
    f"conflict day: {conflict_day_text}"
)

fig.text(
    0.74, 1.01,
    summary_text,
    ha="left",
    va="top",
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="red",
        linewidth=1.8
    )
)

fig.suptitle("Chemotherapy Schedule Overview - All Days", fontsize=16, y=1.04) #y=1.04 是讓標題稍微往上移動，避免和圖例重疊

plt.tight_layout()
plt.savefig(overview_filename, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"總覽甘特圖已輸出至 {overview_filename}")
