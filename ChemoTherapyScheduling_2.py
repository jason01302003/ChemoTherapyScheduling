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
prob = 0.3
slot_length = 15
work_start = 9*60
max_clock = 7*60
n_slots = max_clock // slot_length

random.seed(17)
rnd_txt = random.randint(1, 9)
fname = rf"D:\桌面\OTA_paper\chemo\SCP"+str(P)+"-"+str(2)+".txt"
path = fname
f = open(path, 'w')

# 生成測試資料
V = []
Last_Position = []
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
    Last_Position.append(last)
# 隨機每日最大容量
K = [random.randint(3, 10) for _ in range(T)]
# Pattern生成和對照
Pattern = []
for i in range(P):
    pattern_list = [
        random.randint(1, 2),
        random.randint(1, 7),
        random.randint(1, 2),
        random.randint(1, 3),
        random.randint(1, 2)
    ]
    Pattern.append(pattern_list)
task_time_map = {
    "task1": {"1": 15, "2": 30},
    "task2": {"1": 150, "2": 165, "3": 180, "4": 195, "5": 210, "6": 225, "7": 240},
    "task3": {"1": 15, "2": 30},
    "task4": {"1": 15, "2": 30, "3": 45},
    "task5": {"1": 15, "2": 30}
}
tasks = ["task1", "task2", "task3", "task4", "task5"]

# 輸出生成資料

for j in range(L):
    print(K[j], end="", file=f)
print("\n", file=f)
for i in range(P):
    print(f"Patient {i:2d}: ", end="", file=f)
    for j in range(L):
        print(V[i][j], end="", file=f)
    print("\n", file=f)
    print(f"Pattern  {i:2d}: ", end="", file=f)
    for j in range(5):
        print(Pattern[i][j], end="", file=f)
    print("\n", file=f)

# 預處理
durations_minutes = []
durations_slots = []
total_slots = []
feasible_start_slots = {}

for i in range(P):
    dmins = [task_time_map["task1"][str(Pattern[i][0])],
             task_time_map["task2"][str(Pattern[i][1])],
             task_time_map["task3"][str(Pattern[i][2])],
             task_time_map["task4"][str(Pattern[i][3])],
             task_time_map["task5"][str(Pattern[i][4])]]
    durations_minutes.append(dmins)
    dslots = []
    for d in dmins:
        dslots.append((d + slot_length - 1)//slot_length)
    durations_slots.append(dslots)
    total_slots.append(sum(dslots))
    feasible_start_slots[i] = [q for q in range(
        0, n_slots) if q + sum(dslots) <= n_slots]

nurse_occ = {}
for i in range(P):
    dslots = durations_slots[i]
    for q in feasible_start_slots[i]:
        occ = set()
        offset = q
        for s in range(offset, offset + dslots[0]):
            occ.add(s)
        offset += dslots[0] + dslots[1]
        for s in range(offset, offset + dslots[2]):
            occ.add(s)
        offset += dslots[2] + dslots[3]
        for s in range(offset, offset + dslots[4]):
            occ.add(s)
        occ = set([s for s in occ if 0 <= s < n_slots])
        nurse_occ[(i, q)] = occ

# 建模
CTS = Model("ChemoTherapyScheduling")
CTS.setParam('OutputFlag', 1)
CTS.setParam('TimeLimit', 300)
CTS.setParam('MIPGap', 0.01)
CTS.setParam('Threads', 6)

X = CTS.addVars(P, T, vtype=GRB.BINARY, name="X")
Y = CTS.addVars(P, T, vtype=GRB.BINARY, name="Y")
MaxSize = CTS.addVar(vtype=GRB.INTEGER, lb=0, name="MaxSize")

Z = {}
for i in range(P):
    for t in range(T):
        for q in feasible_start_slots[i]:
            Z[(i, t, q)] = CTS.addVar(vtype=GRB.BINARY, name=f"Z_{i}_{t}_{q}")
CTS.update()

# 約束 (A) 只開始一次
for i in range(P):
    if Last_Position[i] == -1:
        for t in range(T):
            CTS.addConstr(X[i, t] == 0)
        continue
    max_offset = Last_Position[i]
    max_start = T - 1 - max_offset
    CTS.addConstrs(
        (quicksum(X[i, s] for s in range(max_start + 1)) == 1 for i in range(P)), "Start Once")

# 約束 (B) Y >= X
for i in range(P):
    if Last_Position[i] == -1:
        for t in range(T):
            CTS.addConstr(Y[i, t] == 0)
        continue
    max_offset = Last_Position[i]
    max_start = T - 1 - max_offset
    for s in range(max_start + 1):
        for j, val in enumerate(V[i]):
            if val == 1:
                day = s + j
                if day < T:
                    CTS.addConstr(Y[i, day] >= X[i, s])

# 約束 (C) 最大容量
for t in range(T):
    CTS.addConstr(quicksum(Y[i, t] for i in range(P))
                  <= MaxSize, "Maximum Daily Size")
    CTS.addConstr(quicksum(Y[i, t]
                  for i in range(P)) <= K[t], "Actual Daily Size")

# 約束 (D) Z and Y link
for i in range(P):
    for t in range(T):
        zs = [Z[(i, t, q)] for q in feasible_start_slots[i]]
        if zs:
            CTS.addConstr(quicksum(zs) == Y[i, t])
        else:
            CTS.addConstr(Y[i, t] == 0)

# 衝突懲罰
Conflict = {}
for t in range(T):
    for s in range(n_slots):
        Conflict[(t, s)] = CTS.addVar(
            vtype=GRB.CONTINUOUS, lb=0, name=f"Conflict_{t}_{s}")

for t in range(T):
    for s in range(n_slots):
        terms = []
        for i in range(P):
            for q in feasible_start_slots[i]:
                if s in nurse_occ[(i, q)]:
                    terms.append(Z[(i, t, q)])
        if terms:
            CTS.addConstr(Conflict[(t, s)] >= quicksum(terms) - 1)
            CTS.addConstr(Conflict[(t, s)] >= 0)

# 目標函數
penalty_weight = 100
CTS.setObjective(MaxSize + penalty_weight *
                 quicksum(Conflict.values()), GRB.MINIMIZE)

# 求解
starttime = time.time()
CTS.optimize()
endtime = time.time()
print("Run time: %2.5s seconds" % (endtime - starttime))
print("Run time: %2.5s seconds" % (endtime - starttime), file=f)

# 生成排程
schedule = {}
for i in range(P):
    schedule[i] = []
    start_day = None
    for s in range(T):
        if X[i, s].X != 0:
            start_day = s
            break
    if start_day is None:
        continue
    for t in range(T):
        if Y[i, t].X != 0:
            chosen_q = None
            for q in feasible_start_slots[i]:
                if Z[(i, t, q)].X != 0:
                    chosen_q = q
                    break
            if chosen_q is None:
                chosen_q = feasible_start_slots[i][0] if feasible_start_slots[i] else 0
            start_min = work_start + chosen_q * slot_length
            day_tasks = []
            for k_idx, k in enumerate(tasks):
                dur = task_time_map[k][str(Pattern[i][k_idx])]
                end_min = start_min + dur
                day_tasks.append([t, k, start_min, end_min])
                start_min = end_min
            schedule[i].append(day_tasks)
# 印出 schedule
for i in range(P):
    print(f"Patient {i:2d}:", end="", file=f)
    print("\n", end="", file=f)
    print(f"Patient {i:2d}:")
    print("\n")
    for day_tasks in schedule[i]:
        for day, task, s_min, e_min in day_tasks:
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
task_colors = {
    "task1": "skyblue",
    "task3": "skyblue",
    "task5": "skyblue",
    "task2": "red",
    "task4": "red"
}

output_folder = r"D:\桌面\OTA_paper\ChemoTherapyScheduling\gantt_days_2"
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
    for pos, i in enumerate(patient_indices):
        for day_tasks in schedule[i]:
            if day_tasks[0][0] == day:

                for _, task, s_min, e_min in day_tasks:

                    # 從 s_min 開始，每 15 分鐘畫一段 bar
                    current = s_min
                    while current < e_min:
                        next_t = min(current + slot_length, e_min)

                        # 計算 slot index
                        t_slot = (current - work_start) // slot_length

                        # 判斷衝突
                        has_conflict = False
                        if 0 <= t_slot < n_slots:
                            if Conflict[(day, t_slot)].X > 0.5:
                                has_conflict = True

                        # 顏色邏輯
                        if has_conflict and task in ["task1", "task3", "task5"]:
                            color = "orange"
                        else:
                            color = task_colors[task]

                        # 畫出 15 分鐘 bar
                        ax.barh(
                            y=pos,
                            width=next_t - current,
                            left=current,
                            height=0.4,
                            color=color,
                            edgecolor="black"
                        )

                        current = next_t

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
    patches = [mpatches.Patch(color=color, label=task)
               for task, color in task_colors.items()]
    patches.append(mpatches.Patch(color="orange", label="Nurse conflict"))
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(f"Chemotherapy Schedule - Day {day}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Patients")

    plt.tight_layout()
    filename = os.path.join(output_folder, f"Day_{day:02d}.png")
    plt.savefig(filename)
    plt.close(fig)

print(f"甘特圖自動生成並存至 {output_folder} 資料夾")
