# @author: Niklas Schmid, Marta Fochesato, Tobias Sutter, John Lygeros, ETH Zurich, Automatic Control Laboratory
# This script belongs to the publication "Joint Chance Constrained Optimal Control via Linear Programming" and
# implements the described numerical example. The script shows how to solve joint chance constrained optimal control
# problems for MDPs, where the joint chance constraint can either represent an invariance, reachability or reach-avoid
# objective.

# Import Libraries
import numpy as np
import imageio.v3 as iio
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import sparse
from gurobipy import *
import pickle
import random
import time

# Bound values between high and low
def bound(low, high, value):
    return max(low, min(high, value))

# Generate world with safe and target sets from the attached .png files
def generateWorld():
    print("Loading track.")
    im = iio.imread(track_name)   # Loads track from png file
    avoid = np.array(im[:, :, 0]) / 255.0                 # red:   Avoid
    reach = np.array(im[:, :, 2]) / 255.0                 # blue:  Target
    x_0 = np.argwhere(np.array(im[:, :, 1]) > 0)[0]       # green: Initial State (row, column)

    # Depending on objective type, set variable so that goal is not reached yet
    if OBJECTIVE_TYPE == 'I' or OBJECTIVE_TYPE == 'RA':
        b_0 = 1
    else:
        b_0 = 0

    # The safe set is what is left over
    safe = 1 - avoid - reach

    # Uncomment the following lines to plot the environment
    #plt.figure(0)
    #plt.imshow(safe, cmap='hot', interpolation='nearest')
    #plt.show() # Plots track
    return [safe,reach,avoid, x_0, b_0]

# Implement the success indicator function delta
def successIndicator(b_idx):
    if OBJECTIVE_TYPE == 'I' or OBJECTIVE_TYPE == 'R':
        return b_idx
    elif OBJECTIVE_TYPE == 'RA' and b_idx==2:
        return 1
    else:
        return 0

# This function simulates the binary dynamics. Input: x_k, b_k; Output: b_{k+1}
def binaryDynamics(b_idx, x_idx_next, y_idx_next):
    if OBJECTIVE_TYPE == 'I':
        if b_idx == 0 or (b_idx == 1 and map_avoid[x_idx_next, y_idx_next] == 1):
            b_idx_next = 0
        else:
            b_idx_next = 1

    elif OBJECTIVE_TYPE == 'R':
        if b_idx == 1 or (b_idx == 0 and map_reach[x_idx_next, y_idx_next] == 1):
            b_idx_next = 1
        else:
            b_idx_next = 0
    else:
        if b_idx == 0 or (b_idx == 1 and map_avoid[x_idx_next, y_idx_next] == 1):
            b_idx_next = 0
        elif b_idx == 2 or (b_idx == 1 and map_reach[x_idx_next, y_idx_next] == 1):
            b_idx_next = 2
        else:
            b_idx_next = 1

    return b_idx_next

# Generate the stochastic transition kernel by exhaustive simulation
def generateDynamicsAndCosts():
    # Build the transition kernel
    print("Building Dynamics.")

    # Allocate memory
    T = np.zeros([card_X_xy,card_X_xy,card_b,card_U_speed,card_U_angle,card_X_xy,card_X_xy,card_b])
    T_vec = np.zeros([card_X_xy*card_X_xy*card_b*card_U_speed*card_U_angle, card_X_xy*card_X_xy*card_b])

    # We later build constraints for every state and action. We build vectors remembering for every state action pair
    indicator_delta_of_b = np.zeros([card_X_xy*card_X_xy*card_b*card_U_speed*card_U_angle, 1])  # the success-indicator value of the binary state
    indicator_of_safe_augmented_states = np.zeros([card_X_xy*card_X_xy*card_b, ])               # if the state was safe
    indicator_of_target_augmented_states = np.zeros([card_X_xy*card_X_xy*card_b, ])             # if the state was a target

    C = np.zeros([card_X_xy*card_X_xy*card_b*card_U_speed*card_U_angle, 1]) # Store the stage cost of every combination of states and actions

    print("Memory allocated, running simulations now.")
    for x_idx in range(card_X_xy):  # for all states in x
        print("Progress: ", np.floor(x_idx/card_X_xy*100), "%")
        for y_idx in range(card_X_xy):  # for all states in y
                for b_idx in range(card_b):  # for all stage values
                    # Remember states that are considered successfull in terms of the augmented state by setting a one in this vector
                    indicator_of_safe_augmented_states[b_idx + card_b*(y_idx + card_X_xy*x_idx)] = successIndicator(b_idx)
                    for speed in range(card_U_speed):  # for all speed-inputs
                        for angle_aimed in range(card_U_angle):  # for all angle-inputs
                            # Technically, we could ignore many cases here, e.g., for invariance problems when b_k=1 and
                            # the state is in the unsafe set, since we never transition to such states. We decided to
                            # leave this computational overhead for better readability of the code and simplicity.

                            # Safe the stage cost of the given state action combination
                            C[angle_aimed + card_U_angle * (speed + card_U_speed * (b_idx + card_b * (
                                    y_idx + card_X_xy * x_idx))), 0] = speed
                            # Same as indicator_of_safe_augmented_states, but vector is for every state and action now.
                            indicator_delta_of_b[angle_aimed + card_U_angle * (speed + card_U_speed * (b_idx + card_b * (
                                    y_idx + card_X_xy * x_idx))), 0] = successIndicator(b_idx)
                            for dist_sample in range(card_D):   # sample multiple disturbances
                                angle_true = angle_aimed + np.random.normal(0, .5) # angle headed by vehicle
                                x_idx_next = x_idx + round((speed)*np.cos(2*np.pi/card_U_angle * angle_true) + np.random.normal(0, 1))
                                x_idx_next = bound(0, card_X_xy - 1, x_idx_next) # next x
                                y_idx_next = y_idx + round((speed)*np.sin(2*np.pi/card_U_angle * angle_true) + np.random.normal(0, 1))
                                y_idx_next = bound(0, card_X_xy - 1, y_idx_next) # next y
                                b_idx_next = binaryDynamics(b_idx, x_idx_next, y_idx_next)

                                # Store the simulated dynamics, once in a high dimensional matrix, once in a 2x2 matrix.
                                T[x_idx,y_idx,b_idx,speed,angle_aimed,x_idx_next,y_idx_next,b_idx_next] += 1/card_D
                                T_vec[angle_aimed + card_U_angle * (speed + card_U_speed * (b_idx + card_b * (
                                    y_idx + card_X_xy * x_idx))), b_idx_next + card_b * (
                                    y_idx_next + card_X_xy * x_idx_next)] += 1/card_D

    print("Transition kernel generated.")
    return [C, T, T_vec, indicator_delta_of_b, indicator_of_safe_augmented_states, indicator_of_target_augmented_states]

# Solve the LP formulation in the paper. If LP="Lambda", then the optimal lambda is evaluated. Otherwise the optimal
# value functions for lambda_star_input are evaluated.
def solveLP(LP ,lambda_star_input):
    print("Constructing LP.")
    n = card_X_xy*card_X_xy*card_b
    m = card_U_speed*card_U_angle

    # At the cost of computational overhead the implementation is kept as simple as possible.
    # Technically, only the initial state at k=0 needs to be considered, and the value functions at the terminal state
    # k=N are independent of the inputs. Here we consider all states under every input for simplicity. This does not
    # change the solution.

    # The optimization variables of the LP are the values of the function J_k(x_k) at all states x_k and time-steps k.
    # The LP hence optimizes over the vector [J_0(0), J_0(1), ..., J_0(n-1), J_1(0), ..., J_N(n-1), lambda]^T.

    # Construct constraints J_k(x_k) \leq l_k(x_k, u_k) + \int_X J_{k+1}(x_{k+1}) T(dx_{k+1}|x_k,u_k)
    #                       = J_k(x_k) - \int_X J_{k+1}(x_{k+1}) T(dx_{k+1}|x_k,u_k) \leq l_k(x_k, u_k)
    x_k_pick = np.kron(np.eye(n), np.ones((m,1)))     # Picks the J_k(x_k) for all x_k at a given time-step k
    x_0toN_pick = np.kron(np.eye(N+1), x_k_pick)      # Picks the J_k(x_k) for all x_k at every time-step k
    # The following matrix realizes the left hand side of the constraint. T_vec times the decision variable realizes
    # \int_X J_{k+1}(x_{k+1}) T(dx_{k+1}|x_k,u_k). The identity in the kronecker is shifted so that T_vec is multiplied
    # with the value function values of the next time-step.
    A = x_0toN_pick - np.kron(np.eye(N+1,k=1), T_vec)
    # Build stage cost right-hand-side vector. We assume no terminal cost for our example, so we copy the stage-cost N-1
    # times and have only zeros on the right hand side for the constraints value functions at time step N.
    N_times_1_then_0 = np.concatenate((np.ones((N,1)),np.zeros((1,1))), axis=0)
    b = np.kron(N_times_1_then_0, C)

    # So far the LP does not include any lambda. Extend constraints for the variable lambda:
    #   Terminal cost is        V_N(x_N) <= l_N(x_N) + \lambda(\alpha-b_N)
    #                     <=>   V_N(x_N) - \lambda(\alpha-b_N) <= l_N(x_N)
    lagrange_cost = np.kron(1-N_times_1_then_0, -alpha + indicator_delta_of_b)
    A = np.concatenate((A, lagrange_cost), axis=1) # adding (alpha-b_N)*lambda to terminal cost

    # If LP="Lambda" --> Solve for the optimal lambda
    if LP=="Lambda":
        # We add a constraint for the values lambda may attain; lambda\geq 0.
        # Lambda is the last variable in the vector which the LP optimizes over.
        lambda_indicator = np.zeros((1, (N + 1) * n + 1))
        lambda_indicator[0, (N + 1) * n] = -1  # lambda greater than zero
        A = np.concatenate((A, lambda_indicator), axis=0)  # -lambda <= 0 i.e., lambda >= 0
        b = np.concatenate((b, np.zeros((1,1))), axis=0)   # lambda >= 0

    # Formulate objective function. If we aim to solve for the optimal lambda, just consider the initial state at k=0.
    # Otherwise, consider all states at all time-steps.
    if LP=="Lambda":
        objective = np.zeros((1, (N+1)*n + 1))  # Do not consider anything in the objective...
        objective[0,x_0_idx] = 1                # except for the initial state.
    else:
        objective = np.ones((1, (N+1)*n + 1))   # Consider everything in the objective...
        objective[0, (N+1)*n] = 0               # except for the value of lambda.

    # Create a new model
    model = Model()
    # Create variables
    vars = model.addMVar(((N+1)*n + 1,),-GRB.INFINITY)
    # Set objective function
    model.setObjective(objective@vars, GRB.MAXIMIZE)

    # Add constraints
    for k in range(n*m*(N+1)): # Add the constraints on the value function values
        if k%np.floor((n*m*(N+1) + 1)/10)==0:
            print("Progress: ", np.floor(k / np.floor((n*m*(N+1) + 1)/100)), "%")
        model.addConstr(A[k,:]@vars <= b[k,0])
    if LP=="Lambda":        # LP=="Lambda"? --> Add the constraints that lambda is geater or equal zero
        model.addConstr(A[n*m*(N+1), :] @ vars <= b[n*m*(N+1), 0])
    if not LP=="Lambda":    # LP!="Lambda"? --> Add the constraint that lambda is fixed to the predefined lambda
        zerosThenOne = np.zeros((1, (N+1)*n + 1))
        zerosThenOne[0, (N+1)*n] = 1
        model.addConstr(zerosThenOne @ vars == lambda_star_input)

    # Run LP solver
    print("Solving optimization with ", (N+1)*n + 1, " variables now.")
    model.setParam("OptimalityTol", 1e-9)
    model.optimize()

    # Print maximized profit value
    print('Maximized profit:', model.objVal)

    # Extract value functions and lambda from the solver.
    tmp = model.getVars()
    J_0N_vec = model.getAttr("X", tmp)
    J_0N_vec = np.array(J_0N_vec).reshape(-1,1) # Reshape the solution as a numpy vector

    # Extract the optimal policy by checking for tight constraints
    J_0N = np.zeros([N+1,card_X_xy,card_X_xy,card_b])   # Store value functions by their time-step and state.
    J_0N_vec_k = np.zeros([N+1,card_X_xy*card_X_xy*card_b]) # Store value functions in vector form like we optimized them in the LP
    Axmb = np.abs(A@J_0N_vec - b)                           # "Tightness of the constraints"
    # Store the optimal policy. Sets a one for an input at a given state if the input is optimal.
    pi_star = np.zeros([N+1,card_X_xy,card_X_xy,card_b,card_U_speed,card_U_angle])
    # Therefore, we iterate over all time-steps, states and inputs, check if the constraint in the LP was tight, i.e., it
    # constrained the value function, i.e., it is the input that yields the lowest cost. If the input yields a tight
    # constraint at a given state, then it is an optimal input and we write a one into the respective coordinate of pi_star.
    for k in range(N+1):
        for x_idx in range(card_X_xy):  # for all states in x
            for y_idx in range(card_X_xy):  # for all states in y
                for b_idx in range(card_b):  # for all b
                    J_0N[k, x_idx,y_idx,b_idx] = J_0N_vec[b_idx + card_b*(y_idx + card_X_xy*(x_idx + card_X_xy*k)),0]
                    J_0N_vec_k[k, b_idx + card_b*(y_idx + card_X_xy*x_idx)] = J_0N_vec[b_idx + card_b*(y_idx + card_X_xy*(x_idx + card_X_xy*k)),0]
                    foundsome=0 # Check if at least one optimal input has been found. (was used for debugging and validation).
                    for speed in range(card_U_speed):  # for all speed-inputs
                        for angle_aimed in range(card_U_angle):  # for all angle-inputs
                            # If the constraint is tight enough:
                            if Axmb[angle_aimed + card_U_angle*(speed + card_U_speed*(b_idx + card_b*(y_idx + card_X_xy*(x_idx + card_X_xy*k))))]<1e-8:
                                pi_star[k,x_idx,y_idx,b_idx,speed,angle_aimed] = 1 # mark the input as optimal
                                foundsome = 1
                    if foundsome==0 and not LP=="Lambda":
                        error=1
                        print("\n")
                        print("ERROR ||| THERE IS A STATE WHERE NO INPUT GENERATES A TIGHT CONSTRAINT! POLICY MAY BE UNINFORMATIVE!")
                        print("\n")

    # Extract lambda_star from the LP solution. If LP=!"Lambda", then it will be equivalent to the inputted lambda_star
    # due to the constructed LP constraints.
    lambda_star = J_0N_vec[card_b-1 + card_b*(card_X_xy-1 + card_X_xy*(card_X_xy-1 + card_X_xy*N)) + 1,0]

    return [J_0N, J_0N_vec, J_0N_vec_k, lambda_star, A, b, pi_star]

# Store data in a file.
def storeData(filename,a,b,c,d,e):
    with open(filename, 'wb') as file:
        tmp = [a,b,c,d,e]
        # Serialize and write the variable to the file
        pickle.dump(tmp, file)
    # Step 3: Loading Variables
    loaded_data = None

# DP Recursion to find the minimum cost policy
def minCostEvaluation():
    C_pi = np.zeros((N + 1, card_X_xy * card_X_xy * card_b))
    # Initialize with terminal cost if there is any here.
    pi_out = np.zeros((N+1,card_X_xy,card_X_xy,card_b,card_U_speed,card_U_angle))
    for k in reversed(range(N)):
        for x_idx in range(card_X_xy):
            for y_idx in range(card_X_xy):
                for b_idx in range(card_b):
                    min_cost = 1000000000000
                    for speed in range(card_U_speed):
                        for angle_aimed in range(card_U_angle):
                            transistion_vector = T_vec[angle_aimed + card_U_angle * (
                                    speed + card_U_speed * (b_idx + card_b * (
                                    y_idx + card_X_xy * x_idx))), :]
                            cost = C[angle_aimed + card_U_angle * (speed + card_U_speed * (b_idx + card_b * (
                                y_idx + card_X_xy * x_idx))), 0] + transistion_vector @ np.transpose(C_pi[k + 1, :])
                            if cost <= min_cost:
                                min_cost = cost
                                speed_star = speed
                                angle_star = angle_aimed
                    pi_out[k, x_idx, y_idx, b_idx, speed_star, angle_star] = 1          # Store input as optimal
                    C_pi[k, b_idx + card_b * (y_idx + card_X_xy * x_idx)] = min_cost
    return [pi_out, C_pi[0,x_0_idx]]

# Evaluate the minimum cost of a policy. If a policy has multiple optimal inputs advertised at a given state, choose the
# one with the lowest control cost.
def costEvaluation(pi_in):
    C_pi = np.zeros((N + 1, card_X_xy * card_X_xy * card_b))
    # Initialize with terminal cost if there is any here.
    pi_out = np.zeros((N+1,card_X_xy,card_X_xy,card_b,card_U_speed,card_U_angle))
    for k in reversed(range(N)):
        for x_idx in range(card_X_xy):
            for y_idx in range(card_X_xy):
                for b_idx in range(card_b):
                    min_cost = 1000000000000
                    for speed in range(card_U_speed):
                        for angle_aimed in range(card_U_angle):
                            transistion_vector = T_vec[angle_aimed + card_U_angle * (
                                    speed + card_U_speed * (b_idx + card_b * (
                                    y_idx + card_X_xy * x_idx))), :]
                            if pi_in[k,x_idx,y_idx,b_idx,speed,angle_aimed]==1:    # Restrict to optimal inputs of pi_in
                                cost = C[angle_aimed + card_U_angle * (speed + card_U_speed * (b_idx + card_b * (
                                    y_idx + card_X_xy * x_idx))), 0] + transistion_vector @ np.transpose(C_pi[k + 1, :])
                                if cost <= min_cost:
                                    min_cost = cost
                                    speed_star = speed
                                    angle_star = angle_aimed
                    pi_out[k, x_idx, y_idx, b_idx, speed_star, angle_star] = 1          # Store input as optimal
                    C_pi[k, b_idx + card_b * (y_idx + card_X_xy * x_idx)] = min_cost
    #print("The achieved cost is ", C_pi[0, x_0_idx])
    return [pi_out, C_pi[0,x_0_idx]]


# DP Recursion to find the safest policy
def maxSafetyEvaluation():
    V_pi = np.zeros((N+1,card_X_xy*card_X_xy*card_b))
    if OBJECTIVE_TYPE=="I":
        V_pi[N, :] = indicator_of_safe_augmented_states
    else:
        V_pi[N,:] = indicator_of_safe_augmented_states
    pi_out = np.zeros((N+1,card_X_xy,card_X_xy,card_b,card_U_speed,card_U_angle))
    for k in reversed(range(N)):
        for x_idx in range(card_X_xy):
            for y_idx in range(card_X_xy):
                for b_idx in range(card_b):
                    max_safety = 0
                    speed_star = 0
                    angle_star = 0
                    for speed in range(card_U_speed):
                        for angle_aimed in range(card_U_angle):
                            transistion_vector = T_vec[angle_aimed + card_U_angle * (
                                    speed + card_U_speed * (b_idx + card_b * (
                                    y_idx + card_X_xy * x_idx))), :]
                            value_function_of_next_time_step = np.transpose(V_pi[k+1, :])

                            # Compared to the typical recursions in the literature there is an implicit trick here: Our
                            # dynamics are generated in such a way that if the system is in an augmented state that is
                            # considered a success by the success-indicator function, then it will only transition
                            # to states that have the same binary state; and all states with this
                            # augmented state value have been assigned a safety of one at terminal time in
                            # the first lines of this function. So they keep being assigned a safety of one over the
                            # recursion. For simplicity of the code, we also assign a non-zero safety to states in the unsafe set with
                            # b_k > 0 here. However, these states are never transitioned to (and they technically do not exist).
                            # Hence, they do not change the value of the safety associated to any other state in the recursion.
                            if OBJECTIVE_TYPE=="I" or OBJECTIVE_TYPE=="RA":
                                if (b_idx == 1 and map_safe[x_idx,y_idx]==1) or b_idx==2:
                                    safety = transistion_vector @ np.transpose(V_pi[k+1, :])
                                else:
                                    safety = 0
                            if OBJECTIVE_TYPE=="R":
                                safety = transistion_vector @ np.transpose(V_pi[k+1, :])

                            if safety>max_safety:
                                max_safety = safety
                                speed_star = speed
                                angle_star = angle_aimed

                    pi_out[k, x_idx, y_idx, b_idx, speed_star, angle_star] = 1          # Store input as optimal
                    V_pi[k, b_idx + card_b * (y_idx + card_X_xy * x_idx)] = max_safety
    #print("The maximum achievable safety is ", V_pi[0,x_0_idx])
    return [pi_out, V_pi[0,x_0_idx]]

# Evaluate the maximum safety of a policy. If a policy has multiple optimal inputs advertised at a given state, choose the
# one with the highest safety.
def safetyEvaluation(pi_in):
    V_pi = np.zeros((N+1,card_X_xy*card_X_xy*card_b))
    if OBJECTIVE_TYPE=="I":
        V_pi[N, :] = indicator_of_safe_augmented_states
    else:
        V_pi[N,:] = indicator_of_safe_augmented_states
    V_pi_map = np.zeros((N+1,card_X_xy,card_X_xy))
    V_pi_map[N,:,:] = map_safe
    pi_out = np.zeros((N+1,card_X_xy,card_X_xy,card_b,card_U_speed,card_U_angle))
    for k in reversed(range(N)):
        for x_idx in range(card_X_xy):
            for y_idx in range(card_X_xy):
                for b_idx in range(card_b):
                    max_safety = 0
                    speed_star = 0
                    angle_star = 0
                    for speed in range(card_U_speed):
                        for angle_aimed in range(card_U_angle):
                                if pi_in[k,x_idx,y_idx,b_idx,speed,angle_aimed]==1: # Restrict to optimal input of pi_in
                                    transistion_vector = T_vec[angle_aimed + card_U_angle * (
                                                speed + card_U_speed * (b_idx + card_b * (
                                                y_idx + card_X_xy * x_idx))), :]

                                    if OBJECTIVE_TYPE == "I" or OBJECTIVE_TYPE == "RA":
                                        if (b_idx == 1 and map_safe[x_idx,y_idx]==1) or b_idx==2:
                                            safety = transistion_vector @ np.transpose(V_pi[k + 1, :])
                                        else:
                                            safety = 0
                                    if OBJECTIVE_TYPE == "R":
                                        safety = transistion_vector @ np.transpose(V_pi[k + 1, :])

                                    if safety>=max_safety:
                                        max_safety = safety
                                        speed_star = speed
                                        angle_star = angle_aimed
                    pi_out[k, x_idx, y_idx, b_idx, speed_star, angle_star] = 1          # Store input as optimal

                    V_pi_map[k,x_idx,y_idx] = max_safety
                    V_pi[k, b_idx + card_b * (y_idx + card_X_xy * x_idx)] = max_safety
    #print("The achieved safety is ", V_pi[0,x_0_idx])
    return [pi_out, V_pi[0,x_0_idx]]

# Run a monte carlo simulation on the continuous state and action system over trial_number trials using the policy pi
def monteCarlo(trial_number, pi, draw):
    cost = 0
    safety = 0
    trajectory_collection = np.zeros((trial_number,3,N+1))
    for trial in range(trial_number):
        cost_of_trajectory = 0
        x_k = np.zeros((3,N+1)) # the third state is b_k
        # Initial state:
        x_k[0, 0]=x_0[0]
        x_k[1, 0]=x_0[1]
        x_k[2, 0]=b_0

        for k in range(N):  # For all time-steps
            # Get the associated index of the current state to find the corresponding input advertised by pi.
            y_idx = round(x_k[1,k])
            y_idx = bound(0, card_X_xy - 1, y_idx)
            x_idx = round(x_k[0,k])
            x_idx = bound(0, card_X_xy - 1, x_idx)
            b_idx = int(x_k[2, k])

            # Extract the optimal input at the given state
            u_star = np.argwhere(pi[k,x_idx,y_idx,b_idx,:,:] == 1)
            u_star = u_star[0]  # Get the first coordinates of the arguments of prior command
            speed = u_star[0]
            angle_aimed = u_star[1]

            # Simulate continuous dynamics
            angle_true = angle_aimed + np.random.normal(0, .5)
            x_dist = np.random.normal(0, 1)
            x_k[0,k+1] = x_k[0,k] + (speed) * np.cos(2 * np.pi / card_U_angle * angle_true)  + x_dist
            x_k[0, k + 1] = bound(0, card_X_xy - 1, x_k[0, k + 1])
            y_dist = np.random.normal(0, 1)
            x_k[1,k+1] = x_k[1,k] + (speed) * np.sin(2 * np.pi / card_U_angle * angle_true) + y_dist
            x_k[1, k + 1] = bound(0, card_X_xy - 1, x_k[1, k + 1])
            # Extract the corresponding index of the state to compute the binary state dynamics
            x_next_idx = bound(0, card_X_xy - 1, round(x_k[0, k + 1]))
            y_next_idx = bound(0, card_X_xy - 1, round(x_k[1, k + 1]))
            x_k[2, k + 1] = binaryDynamics(x_k[2, k],x_next_idx,y_next_idx)

            # Safe cost of the trajectory
            cost_of_trajectory += speed

        # Average safety and cost of the trajectory
        safety += successIndicator(x_k[2, N])/trial_number
        cost += cost_of_trajectory/trial_number
        trajectory_collection[trial,:,:] = x_k

    # Draw the trajectories
    if draw==1:
        plt.figure(2024)
        if OBJECTIVE_TYPE=="I":
            cmap = colors.ListedColormap(['black', 'white'])
        if OBJECTIVE_TYPE=="R":
            cmap = colors.ListedColormap(['white', 'blue'])
        if OBJECTIVE_TYPE=="RA":
            cmap = colors.ListedColormap(['black', 'white', 'blue'])

        plt.imshow(1-map_avoid + map_reach, cmap=cmap, interpolation='nearest',zorder=1, origin='upper')

        for trial in range(0,trial_number,500):
            if successIndicator(trajectory_collection[trial,2,N])==0:
                plt.plot(trajectory_collection[trial,1,:], trajectory_collection[trial,0,:], color='red', linewidth=2,zorder=trial+1)
            else:
                plt.plot(trajectory_collection[trial,1,:], trajectory_collection[trial,0,:], color='green', linewidth=2,zorder=trial+1)
        # Draw the initial state
        plt.scatter(x_0[1],x_0[0], s=15**2,zorder=trial_number+1, c = "black")
        plt.scatter(x_0[1],x_0[0], marker="x", s=12**2,zorder=trial_number+2, c = "white")
        plt.show() # Plots track
        plt.xlabel('x-coordinate (m)', fontsize=30)
        plt.ylabel('y-coordinate (m)', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
    return [cost, safety]

# Simulates the final mixed policy; used to generate the plots in the paper
def simulateAndPlotMixedPolicy(trial_number, safe_policy, cheap_policy, p_safe):
    trajectory_collection = np.zeros((trial_number, 3, N + 1))
    for trial in range(trial_number):
        x_k = np.zeros((3, N + 1))  # the third state is b_k
        x_k[0, 0] = x_0[0]
        x_k[1, 0] = x_0[1]
        x_k[2, 0] = b_0

        # Roll the dice to pick the respective policy of the mixed policy
        randomNumber = np.random.uniform(0,1)
        if randomNumber<= p_safe:
            pi = safe_policy
        else:
            pi = cheap_policy

        for k in range(N):
            y_idx = round(x_k[1, k])
            y_idx = bound(0, card_X_xy - 1, y_idx)
            x_idx = round(x_k[0, k])
            x_idx = bound(0, card_X_xy - 1, x_idx)
            b_idx = int(x_k[2, k])

            u_star = np.argwhere(pi[k, x_idx, y_idx, b_idx, :, :] == 1)
            u_star = u_star[0]  # Get the first coordinates of the arguments of prior command
            speed = u_star[0]
            angle_aimed = u_star[1]

            angle_true = angle_aimed + np.random.normal(0, .5)
            x_dist = np.random.normal(0, 1)
            x_k[0, k + 1] = x_k[0, k] + (speed) * np.cos(2 * np.pi / card_U_angle * angle_true) + x_dist
            x_k[0, k + 1] = bound(0, card_X_xy - 1, x_k[0, k + 1])
            y_dist = np.random.normal(0, 1)
            x_k[1, k + 1] = x_k[1, k] + (speed) * np.sin(2 * np.pi / card_U_angle * angle_true) + y_dist
            x_k[1, k + 1] = bound(0, card_X_xy - 1, x_k[1, k + 1])
            x_next_idx = bound(0, card_X_xy - 1, round(x_k[0, k + 1]))
            y_next_idx = bound(0, card_X_xy - 1, round(x_k[1, k + 1]))
            x_k[2, k + 1] = binaryDynamics(x_k[2, k], x_next_idx, y_next_idx)
            b_idx_next = int(x_k[2, k + 1])
        trajectory_collection[trial, :, :] = x_k

    plt.figure(OBJECTIVE_TYPE,figsize=(5,6))
    if OBJECTIVE_TYPE == "I":
        cmap = colors.ListedColormap(['black', 'white'])
    if OBJECTIVE_TYPE == "R":
        cmap = colors.ListedColormap(['white', 'blue'])
    if OBJECTIVE_TYPE == "RA":
        cmap = colors.ListedColormap(['black', 'white', 'blue'])

    plt.imshow(1 - map_avoid + map_reach, cmap=cmap, interpolation='nearest', zorder=1, origin='upper')

    for trial in range(0, trial_number):
        if successIndicator(trajectory_collection[trial, 2, N]) == 0:
            plt.plot(trajectory_collection[trial, 1, :], trajectory_collection[trial, 0, :], color='red',
                     linewidth=2, zorder=trial + 1)
        else:
            plt.plot(trajectory_collection[trial, 1, :], trajectory_collection[trial, 0, :], color='green',
                     linewidth=2, zorder=trial + 1)
    plt.scatter(x_0[1], x_0[0], s=25 ** 2, zorder=trial_number + 1, c="black")
    plt.scatter(x_0[1], x_0[0], marker="x", s=22 ** 2, zorder=trial_number + 2, c="white")
    plt.show()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout(pad=0)
    if OBJECTIVE_TYPE=="I":
        plt.xlabel('Invariance', fontsize=30)
    if OBJECTIVE_TYPE=="R":
        plt.xlabel('Reachability', fontsize=30)
    if OBJECTIVE_TYPE=="RA":
        plt.xlabel('Reach-Avoidance', fontsize=30)

    plt.figure("blanc"+OBJECTIVE_TYPE, figsize=(5, 6))
    if OBJECTIVE_TYPE == "I":
        cmap = colors.ListedColormap(['black', 'white'])
    if OBJECTIVE_TYPE == "R":
        cmap = colors.ListedColormap(['white', 'blue'])
    if OBJECTIVE_TYPE == "RA":
        cmap = colors.ListedColormap(['black', 'white', 'blue'])

    plt.imshow(1 - map_avoid + map_reach, cmap=cmap, interpolation='nearest', zorder=1, origin='upper')
    plt.scatter(x_0[1], x_0[0], s=25 ** 2, zorder=trial_number + 1, c="black")
    plt.scatter(x_0[1], x_0[0], marker="x", s=22 ** 2, zorder=trial_number + 2, c="white")
    plt.show()  # Plots track
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout(pad=0)
    if OBJECTIVE_TYPE == "I":
        plt.xlabel('Invariance', fontsize=30)
    if OBJECTIVE_TYPE == "R":
        plt.xlabel('Reachability', fontsize=30)
    if OBJECTIVE_TYPE == "RA":
        plt.xlabel('Reach-Avoidance', fontsize=30)


# Main function
if __name__ == '__main__':
    np.random.seed(10)
    plt.rcParams['pdf.fonttype'] = 42       # Needed to prevent figures from containing Type 3 fonts
    plt.rcParams['ps.fonttype'] = 42        # Needed to prevent figures from containing Type 3 fonts

    GENERATENEW = False     # True: Regenerate dynamics and resolve LPs; False: Load from safe-files
    OBJECTIVE_TYPE = 'RA'    # We consider three different settings: Invariance "I", Reachability "R", Reach-Avoid "RA"

    # N: Time-Horizon; alpha: Safety Probability
    if OBJECTIVE_TYPE=='I': # Load the Invariance Example
        track_name = 'invariance_low.png'
        filename = 'invariance.pkl'
        filename2 = 'invariance2.pkl'
        N=15
        alpha=0.9
    elif OBJECTIVE_TYPE=='R': # Load the Reachability Example
        track_name = 'reachability_low.png'
        filename = 'reachability.pkl'
        filename2 = 'reachability2.pkl'
        N=15
        alpha=0.6
    elif OBJECTIVE_TYPE=='RA': # Load the Reach-Avoid Example
        track_name = 'reachavoid_low.png'
        filename = 'reachavoid.pkl'
        filename2 = 'reachavoid2.pkl'
        N=15
        alpha=0.35

    # Load environment with safe, target and usafe sets from .png file
    [map_safe, map_reach, map_avoid, x_0, b_0] = generateWorld()

    # Define parameters of discretization for the states and inputs
    card_X_xy = np.shape(map_safe)[0]   # x,y follows size of the .png
    monte_carlo_trials = 10000          # Trials used to evaluate the result on the continuous dynamics using Monte-Carlo
    card_U_speed = 3    # number of speed-values considered
    card_U_angle = 4    # number of angles considered
    card_D = 400        # number of disturbance samples at every state and action to estimate the stochastic dynamics
    # Cardinality of the augmented state
    if OBJECTIVE_TYPE=='I':
        card_b = 2
    elif OBJECTIVE_TYPE=='R':
        card_b = 2
    elif OBJECTIVE_TYPE=='RA':
        card_b = 3

    # Generate new dynamics if GENERATENEW==True, otherwise load from file.
    if GENERATENEW:
        # Generate transition matrix and cost vectors
        [C, T, T_vec, indicator_delta_of_b, indicator_of_safe_augmented_states, indicator_of_target_augmented_states] = generateDynamicsAndCosts()
        storeData(filename, C, T, T_vec, indicator_delta_of_b,indicator_of_safe_augmented_states)
    else:
        with open(filename, 'rb') as file:
            [C, T, T_vec, indicator_delta_of_b,indicator_of_safe_augmented_states] = pickle.load(file)

    # Index of the initial state
    x_0_idx = b_0 + card_b * (x_0[1] + card_X_xy * x_0[0])

    # Compute the safest policy and its cost, as well as the cheapest policy in terms of control cost and its safety
    [pi_safest, safety_pi_safest] = maxSafetyEvaluation()                       # Safest policy
    [pi_safest_star, cost_pi_safest] = costEvaluation(pi_safest)                # and its cost
    [pi_cheapest, cost_pi_cheapest] = minCostEvaluation()                       # Lowest control cost policy
    [pi_cheapest_star, safety_pi_cheapest] = safetyEvaluation(pi_cheapest)      # and its safety

    print("\n")
    print("\033[4mComputing the safety and cost of the safest, cheapest and JCC policy under the discretized dynamics using dynamic programming:\033[0m")
    print("The highest achievable safety is obtained by a policy with (safety, cost) (%.4f" %safety_pi_safest, ", %.4f)" %cost_pi_safest)
    print("The lowest achievable cost is obtained by a policy with (safety, cost) (%.4f" %safety_pi_cheapest, ", %.4f)" %cost_pi_cheapest)

    # Solve for optimal joint chance constrained policy if GENERATENEW=="True", other load from file.
    if GENERATENEW:
        time_keeper_start = time.time()
        # Generate transition matrix and cost vectors
        [J_0N, J_0N_vec, J_0N_vec_k, lambda_star, A, b, pi_star] = solveLP("Lambda",0)
        print("\033[4m\n\033[0m")
        print("The optimal lambda is ", lambda_star)
        [J_0N, J_0N_vec, J_0N_vec_k, lambda_star, A, b, pi_star] = solveLP("J",lambda_star) #+1e-2
        time_keeper_end = time.time()
        print("The optimization took (s) ", time_keeper_end - time_keeper_start)
        storeData(filename2, J_0N, J_0N_vec_k, lambda_star, pi_star, 0)
    else:
        # and to load the session again:
        with open(filename2, 'rb') as file:
            # Deserialize and retrieve the variable from the file
            [J_0N, J_0N_vec_k, lambda_star, pi_star, dummy] = pickle.load(file)

    # Evaluate the safety and cost of the safest lambda optimal policy
    [pi_star_safest, safety_pi_star_safest] = safetyEvaluation(pi_star)
    [pi_star_safest, cost_pi_star_safest] = costEvaluation(pi_star_safest)
    # Evaluate the safety and cost of the cheapest lambda optimal policy
    [pi_star_cheapest, cost_pi_star_cheapest] = costEvaluation(pi_star)
    [pi_star_cheapest, safety_pi_star_cheapest] = safetyEvaluation(pi_star_cheapest)
    # Compute optimal mixing probabilities
    p_lambda_overline = (alpha - safety_pi_star_cheapest) / (safety_pi_star_safest - safety_pi_star_cheapest)

    # Compute safety and cost of the optimal mixed policy when applied to the discretized dynamics.
    safety_mixed_policy = p_lambda_overline*safety_pi_star_safest + (1-p_lambda_overline)*safety_pi_star_cheapest
    cost_mixed_policy = p_lambda_overline*cost_pi_star_safest + (1-p_lambda_overline)*cost_pi_star_cheapest

    print("Out of the lambda optimal policies, the highest achievable safety is obtained by a policy with (safety, cost) (%.4f" %safety_pi_star_safest, ", %.4f)" %cost_pi_star_safest)
    print("Out of the lambda optimal policies, the lowest achievable cost is obtained by a policy with (safety, cost) (%.4f" %safety_pi_star_cheapest, ",%.4f)"%cost_pi_star_cheapest)
    print("The optimal mixed policy would consequently result in a (safety, cost) (%.4f" %safety_mixed_policy,", %.4f)" %cost_mixed_policy)

    # Evaluate all prior policies on the continuous dynamics using monte-carlo simulation
    print("\n")
    print("\033[4mComputing the safety and cost of the safest, cheapest and JCC policy under the continuous dynamics using monte carlo simulation (%.0f"%monte_carlo_trials, "trials)\033[0m")
    [cost_safest_star_mc,safety_safest_star_mc]=monteCarlo(monte_carlo_trials,pi_safest_star,0)
    print("Monte-Carlo Simulation for the safest policy resulted in a (safety, cost) (%.4f" % safety_safest_star_mc, ", %.4f)" % cost_safest_star_mc)
    [cost_cheapest_star_mc,safety_cheapest_star_mc]=monteCarlo(monte_carlo_trials,pi_cheapest_star,0)
    print("Monte-Carlo Simulation for the cheapest policy resulted in a (safety, cost) (%.4f" % safety_cheapest_star_mc, ", %.4f)" % cost_cheapest_star_mc)
    [cost_star_safest_mc,safety_star_safest_mc]=monteCarlo(monte_carlo_trials,pi_star_safest,0)
    print("Monte-Carlo Simulation for the safest lambda optimal policy resulted in a (safety, cost) (%.4f" % safety_star_safest_mc, ", %.4f)" % cost_star_safest_mc)
    [cost_star_cheapest_mc,safety_star_cheapest_mc]=monteCarlo(monte_carlo_trials,pi_star_cheapest,0)
    print("Monte-Carlo Simulation for the cheapest lambda optimal policy resulted in a (safety, cost) (%.4f" % safety_star_cheapest_mc, ", %.4f)" % cost_star_cheapest_mc)
    safety_mixed_policy_mc = p_lambda_overline*safety_star_safest_mc + (1-p_lambda_overline)*safety_star_cheapest_mc
    cost_mixed_policy_mc = p_lambda_overline*cost_star_safest_mc + (1-p_lambda_overline)*cost_star_cheapest_mc
    print("Monte-Carlo Simulation for the optimal mixed policy would consequently result in a (safety, cost) (%.4f" % safety_mixed_policy_mc, ", %.4f)" % cost_mixed_policy_mc)

    # Generate the plots in the numerical results section of the paper by simulating the optimal mixed policy
    simulateAndPlotMixedPolicy(10, pi_star_safest, pi_star_cheapest, p_lambda_overline)