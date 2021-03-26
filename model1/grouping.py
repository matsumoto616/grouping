import numpy as np
import pulp
import itertools

number_of_people = 6
number_of_group = 3
number_of_times = 3

people_indices = list(range(number_of_people))
group_indices = list(range(number_of_group))
time_indices = list(range(number_of_times))
combi_indices = list(itertools.combinations(people_indices,2))

num_of_people_in_group = np.empty(number_of_group)
q, mod = divmod(number_of_people,number_of_group)
for j in group_indices:
    if j < mod:
        num_of_people_in_group[j] = q + 1
    else:
        num_of_people_in_group[j] = q

prob = pulp.LpProblem(sense=pulp.LpMaximize)

x = pulp.LpVariable.dicts(
    'x',
    (people_indices,group_indices,time_indices),
    0,
    1,
    pulp.LpBinary
)

y = pulp.LpVariable.dicts(
    'y',
    (combi_indices,group_indices,time_indices),
    0,
    1,
    pulp.LpBinary
)

z = pulp.LpVariable.dicts(
    'z',
    (combi_indices),
    0,
    1,
    pulp.LpBinary
)

w = pulp.LpVariable('w',0,number_of_people-1,pulp.LpInteger)

# どこかのグループに必ず入る
for t in time_indices:
    for i in people_indices:
        prob += pulp.lpSum([x[i][j][t] for j in group_indices]) == 1

# グループ人数はgiven
for j in group_indices:
    for t in time_indices:
        prob += pulp.lpSum([x[i][j][t] for i in people_indices]) \
                 == num_of_people_in_group[j]

# yはi1とi2がt回目にどちらもグループjなら1，そうでないとき0
for t in time_indices:
    for j in group_indices:
        for (i1,i2) in combi_indices:
            prob += 2*y[(i1,i2)][j][t] <= x[i1][j][t] + x[i2][j][t]
            prob += x[i1][j][t] + x[i2][j][t] <= y[(i1,i2)][j][t] + 1

# zはi1とi2が1回でも同じグループになったら1，そうでないとき0
for (i1,i2) in combi_indices:
    prob += z[(i1,i2)] <= pulp.lpSum([y[(i1,i2)][j][t] 
                for (j,t) in itertools.product(group_indices,time_indices)])
                
# w
for i1 in people_indices:
    w_tmp = []
    for i2 in people_indices:
        if i1 < i2:
            w_tmp.append(z[(i1,i2)])
        elif i1 == i2:
            continue
        else:
            w_tmp.append(z[(i2,i1)])
    prob += w <= pulp.lpSum(w_tmp)

prob += w

# 1回目は自動で決定
count_global = 0
for j in group_indices:
    count_local = 0
    for i in people_indices[count_global:]:
        if count_local < num_of_people_in_group[j]:
            x[i][j][0].setInitialValue(1)
            x[i][j][0].fixValue()
        else:
            x[i][j][0].setInitialValue(0)
            x[i][j][0].fixValue()
        count_local += 1
    count_global += count_local

# 2回目以降も0番の人はグループ0に配属
for t in time_indices[1:]:
    x[0][0][t].setInitialValue(1)
    x[0][0][t].fixValue()

prob.solve(pulp.PULP_CBC_CMD(path=None,
                            keepFiles=1,
                            mip=1,
                            msg=1,
                            cuts=None,
                            presolve=None,
                            strong=None,
                            options=[],
                            fracGap=None,
                            maxSeconds=None,
                            threads=None))

for t in time_indices:
    for i in people_indices:
        for j in group_indices:
            if x[i][j][t].value() != 0:
                print(f'{x[i][j][t]} = {x[i][j][t].value()}')

# for (i1,i2) in combi_indices:
#     print(f'{z[(i1,i2)]} = {z[(i1,i2)].value()}')

# print(f'{w} = {w.value()}')