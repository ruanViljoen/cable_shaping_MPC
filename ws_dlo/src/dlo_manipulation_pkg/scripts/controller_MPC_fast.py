#!/usr/bin/env python

# our proposed controller

import numpy as np
from matplotlib import pyplot as plt
import time
import sys, os

import rospy
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as sciR

import copy

from RBF import JacobianPredictor
from utils.state_index import I

import casadi as ca

params_end_vel_max = 0.1
params_normalized_error_thres = 0.2/8
params_control_gain = 1.0
params_lambda_weight = 0.1
params_over_stretch_cos_angle_thres = 0.998



def get_matrices_C1_C2(state):
    
    left_end_pos = state[31:34]
    right_end_pos = state[38:41]
    fps_pos = state[1:31].reshape(-1,3)
    C1 = np.zeros((1, 12))
    C2 = np.zeros((6, 12))

    # decide whether the current state is near over-stretched
    b_over_stretch = False
    segments = fps_pos.copy()
    segments[1:, :] = (fps_pos[1:, :] - fps_pos[0:-1, :])
    cos_angles = np.ones((6, ))
    for i in range(2, 6):
        cos_angles[i-1] = np.dot(segments[i, :], segments[i+1, :]) / (
            np.linalg.norm(segments[i, :]) * np.linalg.norm(segments[i+1, :]))

    ends_distance = (right_end_pos - left_end_pos).reshape(-1, 1)
    params_over_stretch_cos_angle_thres = 0.998
    if np.all(cos_angles > params_over_stretch_cos_angle_thres):
        b_over_stretch = True

    # calculate the C1 and C2 matrix
    if b_over_stretch:
        pd = ends_distance
        C1 = np.concatenate(
            [-pd.T, np.zeros((1, 3)), pd.T, np.zeros((1, 3))], axis=1)
        C2_1 = np.concatenate([np.zeros((3, 3)), np.eye(
            3), np.zeros((3, 3)), np.zeros((3, 3))], axis=1)
        C2_2 = np.concatenate([np.zeros((3, 3)), np.zeros(
            (3, 3)), np.zeros((3, 3)), np.eye(3)], axis=1)
        C2 = np.concatenate([C2_1, C2_2], axis=0)
    
    return C1, C2

class Controller(object):


    # --------------------------------------------------------------------
    def __init__(self):
        self.numFPs = rospy.get_param("DLO/num_FPs")
        self.env_dim = rospy.get_param("env/dimension")
        self.env = rospy.get_param("env/sim_or_real")
        self.bEnableEndRotation = rospy.get_param("controller/enable_end_rotation")
        self.b_left_arm = rospy.get_param("controller/enable_left_arm")
        self.b_right_arm = rospy.get_param("controller/enable_right_arm")
        self.targetFPsIdx = rospy.get_param("controller/object_fps_idx")
        self.project_dir = rospy.get_param("project_dir")
        self.offline_model_name = rospy.get_param("controller/offline_model")
        self.control_law = rospy.get_param("controller/control_law")
        self.controlRate = rospy.get_param("ros_rate/env_rate")

        # the non-zero dimension of the control input
        self.validJacoDim = self.getValidControlInputDim(self.env_dim, self.bEnableEndRotation, self.b_left_arm, self.b_right_arm)

        self.solver = self.generate_MPC_solver()
        self.jacobianPredictor = JacobianPredictor()
        self.jacobianPredictor.LoadModelWeights()

        self.k = 0
        self.case_idx = 0
        self.state_save = []

    # --------------------------------------------------------------------
    def generate_MPC_solver(self):
        print("Creating MPC solver...")
        # current position of fps
        x0 = ca.SX.sym('x0', (24,1))
        # desired position of fps
        xd = ca.SX.sym('xd', (24,1))
        Jacobian = ca.SX.sym('Jacobian', (24, 12))
        
        window_length = 10
        nu = 12  # 12 control inputs
        dt = 0.1  # freq = 10Hz (although doesn't have to be the same...)

        # Decison vars
        # x_k goes from k=0 to k=N
        X = ca.SX.sym('X', (24, window_length+1))
        # dot_x_k goes from k=0 to k=N-1
        # dot_X = ca.SX.sym('dot_X', (30, window_length))
        U = ca.SX.sym('U', (nu, window_length))  # u_k goes from k=0 to k=N-1

        # Define objective function
        err = 0
        g = []
        
        # Add initial state constraint
        g.append(X[:, 0]-x0)

        for i in range(window_length):
            u = U[:, i]
            dot_x_i = Jacobian @ u
            
            # Objective function
            state_err = X[:, i+1] - xd
            # err += U[:, i].T @ weight @ U[:, i] + 10*state_err.T @ state_err #+ dot_x_i.T @ dot_x_i # real system cost function
            err += u.T @ u + state_err.T @ state_err #+ dot_x_i.T @ dot_x_i # simulation cost function
            #Constraints
            g.append(X[:, i] - X[:, i+1] + dt*dot_x_i)
            
        # add terminal cost
        # err += 10*(X[:, -1] - xd).T @ (X[:, -1] - xd)

        g = ca.vertcat(*g)
        # Add initial state constraint
        # g = ca.vertcat(g, X[:, 0]-x0)
        
        # create lbg and ubg (should be 0 for both, except for C1 constraints, which should be <= 0)
        g_shape = g.shape[0]
        self.lbg = np.array([0.0]*g_shape)
        self.ubg = np.array([0.0]*g_shape)

        decision_vars = ca.vertcat(
        ca.vec(U),
        ca.vec(X)
        )

        x_shape = decision_vars.shape[0]
        self.lbx = np.array([-np.inf]*x_shape)
        self.ubx = np.array([np.inf]*x_shape)

        # Constraints on robot velocities
        u_lim = 10*np.array([1]*12)
        print("U lim changed")
        # Create matrix for all control input limits
        U_lim = np.tile(u_lim, window_length)
        # self.lbx[:nu*window_length] = -U_lim
        # self.ubx[:nu*window_length] = U_lim

        p = ca.vertcat(x0, xd, ca.vec(Jacobian))

        nlp = {'x': decision_vars, 'f': err, 'g': g, 'p': p}
        jit_options = {"flags": ["-Ofast"], "verbose": False}

        opts = {'print_time': False, "jit": True, "compiler": "shell", "jit_options": jit_options, 'osqp.verbose': False} 
        solver = ca.qpsol('solver', 'osqp', nlp, opts)
        # solver = ca.qpsol('solver', 'qpoases', nlp)


        print("Solver successfully created! :D")

        self.x0 = np.random.randn(decision_vars.shape[0])
        return solver


    # --------------------------------------------------------------------
    def normalizeTaskError(self, task_error):
        norm =  np.linalg.norm(task_error)
        thres = params_normalized_error_thres * len(self.targetFPsIdx)
        if norm <= thres:
            return task_error
        else:
            return task_error / norm * thres

    # --------------------------------------------------------------------
    def generateControlInput(self, state):
        state_save = copy.deepcopy(state)

        # start control timer
        start_time = time.time()

        fpsPositions = state[I.fps_pos_idx]
        desiredPositions = state[I.desired_pos_idx]

        full_task_error = np.zeros((self.numFPs, 3), dtype='float32')
        full_task_error[self.targetFPsIdx, :] = np.array(fpsPositions - desiredPositions).reshape(self.numFPs, 3)[self.targetFPsIdx, :]

        target_task_error = np.zeros((3 * len(self.targetFPsIdx), 1))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            target_task_error[3*i : 3*i +3, :] = full_task_error[targetIdx, :].reshape(3, 1)

        normalized_target_task_error = self.normalizeTaskError(target_task_error)

        # calcualte the current Jacobian, and do the online learning
        Jacobian = self.jacobianPredictor.OnlineLearningAndPredictJ(state, self.normalizeTaskError(full_task_error.reshape(-1, )))

        # get the target Jacobian
        target_J = np.zeros((3 * len(self.targetFPsIdx), len(self.validJacoDim)))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            target_J[3*i : 3*i+3, :] = Jacobian[3*targetIdx : 3*targetIdx+3, self.validJacoDim]

        # calculate the ideal target point velocity
        alpha = params_control_gain
        fps_vel_ref = - alpha * normalized_target_task_error

        lambd = params_lambda_weight * np.linalg.norm(normalized_target_task_error)
        v_max = params_end_vel_max

        # get the matrix of the constraints for avoiding over-stretching
        # C1, C2 = self.validStateConstraintMatrix(state)

        fps_positions = state[1+3:31-3]
        fps_desired_positions = state[-30+3:-3]
        
        fps_positions = fps_positions.reshape(-1, 1)
        fps_desired_positions = fps_desired_positions.reshape(-1, 1)
        
        p = ca.vertcat(
            fps_positions,
            fps_desired_positions,
            ca.vec(Jacobian[3:-3,:]),
            # ca.vec(C1),
            # ca.vec(C2)
        )

        # Solve the problem
        sol = self.solver(x0=self.x0, ubg=self.ubg, lbg=self.lbg,
                        ubx=self.ubx, lbx=self.lbx, p=p)

        # Get the solution
        x_opt = sol['x']

        # Save solution as initial guess for next iteration
        self.x0 = x_opt

        # Extract control commands
        u = x_opt[:12].full().flatten()

        # u_12DoF = np.zeros((12, ))
        # u_12DoF[self.validJacoDim] = u.reshape(-1, )

        # # ensure safety
        # if np.linalg.norm(u_12DoF) > v_max:
        #     u_12DoF = u_12DoF / np.linalg.norm(u_12DoF) * v_max

        self.k += 1

        # calculate the control time
        control_time = time.time() - start_time
        if control_time > 0.1:
            print(f"control time: {1000*control_time:.3f} ms")

        self.state_save.append(np.array( list(state_save)+[control_time] ))

        return u


    # --------------------------------------------------------------------
    def validStateConstraintMatrix(self, state):
        state = np.array(state)
        left_end_pos = state[I.left_end_pos_idx]
        right_end_pos = state[I.right_end_pos_idx]

        if self.env_dim == '3D':
            fps_pos = state[I.fps_pos_idx].reshape(self.numFPs, 3)
            left_end_pos = state[I.left_end_pos_idx]
            right_end_pos = state[I.right_end_pos_idx]
            C1 = np.zeros((1, 12))
            C2 = np.zeros((6, 12))
        elif self.env_dim == '2D':
            fps_pos = state[I.fps_pos_idx].reshape(self.numFPs, 3)[:, 0:2]
            left_end_pos = state[I.left_end_pos_idx][0:2]
            right_end_pos = state[I.right_end_pos_idx][0:2]
            C1 = np.zeros((1, 6))
            C2 = np.zeros((2, 6))

        # decide whether the current state is near over-stretched
        b_over_stretch = False
        segments = fps_pos.copy()
        segments[1:, :] = (fps_pos[1:, :] - fps_pos[0:-1, :]) 
        cos_angles = np.ones((self.numFPs - 2, ))
        for i in range(2, self.numFPs - 2):
            cos_angles[i-1] = np.dot(segments[i, :], segments[i+1, :]) / (np.linalg.norm(segments[i, :]) * np.linalg.norm(segments[i+1, :]))

        ends_distance =  (right_end_pos - left_end_pos).reshape(-1, 1)
        if np.all(cos_angles > params_over_stretch_cos_angle_thres):
            b_over_stretch = True

        # calculate the C1 and C2 matrix
        if b_over_stretch:  
            pd =  ends_distance
            if self.env_dim == '3D':
                C1 = np.concatenate([-pd.T, np.zeros((1, 3)), pd.T, np.zeros((1, 3))], axis=1)
                C2_1 = np.concatenate([np.zeros((3, 3)), np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))], axis=1)
                C2_2 = np.concatenate([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)], axis=1)
                C2 = np.concatenate([C2_1, C2_2], axis=0)
            elif self.env_dim == '2D':
                C1 = np.concatenate([-pd.T, np.zeros((1, 1)), pd.T, np.zeros((1, 1))], axis=1)
                C2_1 = np.concatenate([np.zeros((1, 2)), np.eye(1), np.zeros((1, 2)), np.zeros((1, 1))], axis=1)
                C2_2 = np.concatenate([np.zeros((1, 2)), np.zeros((1, 1)), np.zeros((1, 2)), np.eye(1)], axis=1)
                C2 = np.concatenate([C2_1, C2_2], axis=0)
            return C1, C2
        else:
            return C1, C2


    # --------------------------------------------------------------------
    def reset(self, state):
        
        result_dir = self.project_dir + "results/" + self.env + "/control/" + self.control_law + "/" + self.env_dim + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        np.save(result_dir + "state_" + str(self.case_idx) + ".npy", self.state_save)
        
        self.case_idx += 1
        self.state_save = []
        self.k = 0

        if (self.case_idx == 100):
            rospy.signal_shutdown("finish.")

        self.jacobianPredictor.LoadModelWeights()


    # --------------------------------------------------------------------
    def getValidControlInputDim(self, env_dim, bEnableEndRotation, b_left_arm, b_right_arm):
        if env_dim == '2D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 5, 6, 7, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 6, 7]
                elif b_left_arm:
                    validJacoDim = [0, 1]
                elif b_right_arm:
                    validJacoDim = [6, 7]
                else:
                    validJacoDim = np.empty()
        elif env_dim == '3D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8, 9, 10, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 6, 7, 8]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8]
                else:
                    validJacoDim = np.empty()
        else:
            print("Error: the environment dimension must be '2D' or '3D'.")

        return validJacoDim