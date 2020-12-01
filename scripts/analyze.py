with open('timing.txt', 'r') as f:
    lines = f.readlines()

records = []
for line in lines:
    if line.startswith('@'):
        records.append(line[1:].strip().split(','))

warp_ids = set([int(r[0]) for r in records])
print("number of warps: %d" % len(warp_ids))
'''
time_points = []
for r in records:
    time_points.append(int(r[1]))
    time_points.append(int(r[2]))
time_points = sorted(time_points)

unique_time_points = [time_points[0]]
for i in range(1, len(time_points)):
    if time_points[i] != time_points[i - 1]:
        unique_time_points.append(time_points[i])

max_timestamp = unique_time_points[-1]
nr_intervals = 10000
interval_len = (max_timestamp + nr_intervals - 1) // nr_intervals
active_warps_count = [0.0] * nr_intervals
for r in records:
    t1, t2 = int(r[1]), int(r[2])

    left  = t1 // interval_len
    right = (t2 + interval_len - 1) // interval_len
    for i in range(left, right):
        begin = interval_len * i
        end = begin + interval_len
        active_warps_count[i] += (min(end, t2) - max(begin, t1)) / interval_len
'''
total_time_costs = {}
num_tasks = {}
time_costs = []
for r in records:
    warp_id, t1, t2 = int(r[0]), int(r[1]), int(r[2])
    time_costs.append(t2 - t1)
    #
    tot = total_time_costs.get(warp_id, 0)
    tot += t2 - t1
    total_time_costs[warp_id] = tot
    #
    num_tasks[warp_id] = num_tasks.get(warp_id, 0) + 1

import json
with open('time_costs.json', 'w') as f:
    json.dump({'task_time': time_costs, 'total_time': total_time_costs, 'num_tasks': num_tasks}, f)

from IPython import embed
#embed()
