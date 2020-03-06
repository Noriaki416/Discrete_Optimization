#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
import pulp
import time

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it_pulp(cost_f, cap_f, loc_f, demand_c, loc_c, d_fc):
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
    for i in range(facility_count):
        for j in range(customer_count):
            lp += y_fc[i, j] <= x_f[i]
     
    
    # Constraint Condition 3
    #if customer_count <= 1000:
    if customer_count <= 200:
        print("Constraint Codition 3 On")
        for i in range(facility_count):
            lp += pulp.lpDot(demand_c , y_fc[i, :]) <= cap_f[i] * x_f[i]
          
    '''
    # Constraint Condition 3
    for i in range(facility_count):
        lp += pulp.lpDot(demand_c , y_fc[i, :]) <= cap_f[i] * x_f[i]
    '''
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
        #if x_f[i].value() > 0:
        #    solution_f[i] = 1
    return obj, solution, solution_f

def solve_it_pulp_again(cost_f, cap_f, loc_f, demand_c, loc_c, d_fc, solution, solution_f):
    facility_count = len(cost_f)
    customer_count = len(demand_c)
    
    # Define Variables For Pulp
    lp = pulp.LpProblem('lp', pulp.LpMinimize)
    x_f = [pulp.LpVariable('x_f({})'.format(i), 0, 1, 'Integer') for i in range(facility_count)]
    y_fc = [[pulp.LpVariable('y_fc({}, {})'.format(i, j), 0, 1, 'Integer') for j in range(customer_count)] for i in range(facility_count)]
    y_fc = np.array(y_fc)
    
    customer_ok_list_all = []
    for idx in range(facility_count):
        if solution_f[idx] == 1:
            x_f[idx] = pulp.LpVariable('x_f({})'.format(idx), 1, 1, 'Integer')
            demand_sum = np.sum(np.array(demand_c)[np.where(np.array(solution) == idx)[0]])
            if demand_sum > cap_f[idx]:
                #print(idx, cap_f[idx])
                customer_set = np.where(np.array(solution) == idx)[0]
                #print(customer_set)
                #print(demand_sum)
                d_temp = d_fc[idx, customer_set]
                arg_index = np.argsort(d_temp)
                #print(customer_set[arg_index])
                demand_temp = 0
                customer_ok_list = []
                for cus_id in customer_set[arg_index]:
                    if demand_temp + demand_c[cus_id]< cap_f[idx]:
                        customer_ok_list.append(cus_id)
                        demand_temp += demand_c[cus_id]
                #print(customer_ok_list)
                customer_ok_list_all.extend(customer_ok_list)
                #print(list(set(customer_set) - set(customer_ok_list)))
                customer_violated_list = list(set(customer_set) - set(customer_ok_list))
                for cus_id in customer_ok_list:
                    for idx_ in range(facility_count):
                        if idx_ == idx:
                            y_fc[idx, cus_id] = pulp.LpVariable('y_fc({}, {})'.format(idx, cus_id), 1, 1, 'Integer')
                        else:
                            y_fc[idx_, cus_id] = pulp.LpVariable('y_fc({}, {})'.format(idx_, cus_id), 0, 0, 'Integer')
                #print("")
                
    s = time.time()
    # Objective Function
    obj1 = pulp.lpDot(cost_f, x_f)
    obj2 = pulp.lpSum([pulp.lpDot(d_fc[i, :], y_fc[i, :])] for i in range(facility_count))
    lp += obj1 + obj2
    
    # Constraint Condition 1
    for j in range(customer_count):
        if j not in customer_ok_list_all:
            lp += pulp.lpSum(y_fc[:, j]) == 1
    
    
    # Constraint Condition 2
    for i in range(facility_count):
        for j in range(customer_count):
            lp += y_fc[i, j] <= x_f[i]
            
    # Constraint Condition 3
    for i in range(facility_count):
        lp += pulp.lpDot(demand_c , y_fc[i, :]) <= cap_f[i] * x_f[i]
    
    f = time.time()
    print("Modeling Time:{}[s]".format(f-s))
    
    s = time.time()
    solver = pulp.PULP_CBC_CMD(maxSeconds=60)
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
        #if x_f[i].value() > 0:
        #    solution_f[i] = 1
    return obj, solution, solution_f



def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')
    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    print(facility_count, customer_count)
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1]*len(customers)
    capacity_remaining = [f.capacity for f in facilities]
    
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
        
    for customer_index in np.argsort(demand_c)[::-1]:
        for facility_index in np.argsort(d_fc[:, customer_index]):
             if capacity_remaining[facility_index] >= customers[customer_index].demand:
                solution[customer_index] = facility_index
                capacity_remaining[facility_index] -= customers[customer_index].demand
                break
                
    facility_opened = list(set(solution))
    for customer_index in np.argsort(demand_c):
        for facility_index in np.argsort(d_fc[:, customer_index]):
            if solution[customer_index] != facility_index and facility_index in facility_opened:
                if capacity_remaining[facility_index] >= customers[customer_index].demand\
                    and  cost_f[solution[customer_index]] + d_fc[solution[customer_index] , customer_index] > \
                        cost_f[facility_index] + d_fc[facility_index, customer_index]:
                    capacity_remaining[solution[customer_index]] += customers[customer_index].demand
                    solution[customer_index] = facility_index
                    capacity_remaining[facility_index] -= customers[customer_index].demand
                    break
    
    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    print(len(list(set(solution))))
    max_f = len(list(set(solution)))
    
    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    
    
    if customer_count <= 1000:
        obj, solution, solution_f = solve_it_pulp(cost_f, cap_f, loc_f, demand_c, loc_c, d_fc)
        if customer_count >= 800:
            obj, solution, solution_f = solve_it_pulp_again(cost_f, cap_f, loc_f, demand_c, loc_c, d_fc, solution, solution_f)
    
    
    
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

