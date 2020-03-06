#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pulp
import time

def solve_it_pulp(cost_f, cap_f, loc_f, demand_c, loc_c, d_fc, model1_flag=True):
    facility_count = len(cost_f)
    customer_count = len(demand_c)
    
    # Define Variables For Pulp
    lp = pulp.LpProblem('lp', pulp.LpMinimize)
    x_f = [pulp.LpVariable('x_f({})'.format(i), 0, 1, 'Integer') for i in range(facility_count)]
    y_fc = [[pulp.LpVariable('y_fc({}, {})'.format(i, j), 0, 1, 'Integer') for j in range(customer_count)] for i in range(facility_count)]
    y_fc = np.array(y_fc)
    s = time.time()
    
    # Objective Function
    obj1 = pulp.lpDot(cost_f, x_f)
    obj2 = pulp.lpSum([pulp.lpDot(d_fc[i, :], y_fc[i, :])] for i in range(facility_count))
    lp += obj1 + obj2
    
    # Constraint Condition 1
    for j in range(customer_count):
        lp += pulp.lpSum(y_fc[:, j]) == 1
    
    # Constraint Condition 2
    if model1_flag:
        for i in range(facility_count):
            for j in range(customer_count):
                lp += y_fc[i, j] <= x_f[i]
    else:
        for i in range(facility_count):
            lp += pulp.lpSum([y_fc[i, j] for j in range(customer_count)]) <= customer_count * x_f[i]

    f = time.time()
    print("Modeling Time:{}[s]".format(f-s))
    
    s = time.time()
    solver = pulp.PULP_CBC_CMD(maxSeconds=60, msg=1)
    result_status = lp.solve(solver)
    result = pulp.value(lp.objective)
    f = time.time()
    print("Solving Time:{}[s]".format(f-s))
    
    print(pulp.LpStatus[result_status])
    obj = pulp.value(lp.objective)
    solution = []
    for j in range(customer_count):
        for i in range(facility_count):
            if y_fc[i, j].value() > 0:
                solution.append(i)
                break
                
    solution_f = np.zeros(facility_count)
    for i in range(facility_count):
        solution_f[i] = x_f[i].value()
    return obj, solution, solution_f

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')
    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    print(facility_count, customer_count)

    # Define Data List
    cost_f = []
    cap_f = []
    loc_f = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        cost_f.append(float(parts[0]))
        cap_f.append(float(parts[1]))
        loc_f.append([float(parts[2]), float(parts[3])])

    demand_c = []
    loc_c = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        demand_c.append(float(parts[0]))
        loc_c.append([float(parts[1]), float(parts[2])])

    cost_f = np.array(cost_f)
    cap_f = np.array(cap_f)
    loc_f = np.array(loc_f)
    demand_c = np.array(demand_c)
    loc_c = np.array(loc_c)
    
    d_fc = np.zeros((facility_count, customer_count)).astype(float)
    for i in range(facility_count):
        for j in range(customer_count):
            d_fc[i, j] = np.sqrt((loc_c[j][0] - loc_f[i][0])**2 + (loc_c[j][1] - loc_f[i][1])**2)

    
    print("Model 1")
    model1_flag = True
    obj, solution, solution_f = solve_it_pulp(cost_f, cap_f, loc_f, demand_c, loc_c, d_fc, model1_flag)
    
    print("Model 2")
    model1_flag = False
    obj, solution, solution_f = solve_it_pulp(cost_f, cap_f, loc_f, demand_c, loc_c, d_fc, model1_flag)
    
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data
    
import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')