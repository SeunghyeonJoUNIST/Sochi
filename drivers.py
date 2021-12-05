import numpy as np
import pandas as pd
import time
import math
from math import cos, sin, atan, pi
from random import randint
import matplotlib.pyplot as plt
from matplotlib import animation

# from genpy.message import check_type


# Parameters
k = 0.3  # look forward gain
Lfc = 0.5  # [m] look-ahead distance
Kp = 0.35  # speed proportional gain
WB = 1  # [m] wheel base of vehicle
speed = 3  # [m/s]

# centerline data
center_data = pd.read_csv("./maps/Sochi_centerline.csv")
cx = np.array(center_data["x_m"])
cy = np.array(center_data["y_m"])
cx = np.concatenate([cx, cx])
cy = np.concatenate([cy, cy])


show_animation = True


# drives straight ahead at a speed of 5
class SimpleDriver:
    def process_lidar(self, ranges, poses):
        print(poses)
        # print("x:", poses[0][0])
        # print("y:", poses[1][0])
        # print("delta:", poses[2][0])
        speed = 1.0
        steering_angle = 0.0
        return speed, steering_angle


# --------------------------- Pure pursuit ----------------------------- #
"""
Path tracking simulation with pure pursuit steering and PID speed control.
"""


class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, time_step, poses):
        self.x = poses[0][0]
        self.y = poses[1][0]
        self.yaw = poses[2][0]
        self.v += a * time_step
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        # if self.old_nearest_point_index is None:
        # search nearest point index
        dx = [state.rear_x - icx for icx in self.cx]
        dy = [state.rear_y - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        ind = np.argmin(d)
        # print("nearest: ", ind)
        self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1
        # print("target idx: ", ind)
        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    # if pind >= ind:
    #     ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
    # print("pose x, pose y(rear):", state.rear_x, state.rear_y)
    # print("target x, y:", tx, ty)
    # print("vector:", tx - state.rear_x, ty - state.rear_y)
    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

    return delta, ind


# drives toward the furthest point it sees

# ------------------------- Pure pursuit end--------------------------- #

class AnotherDriver:
    def __init__(self):
        self.speed = 0
        self.state = State(x=0, y=0, yaw=0, v=0)
        self.states = States()

        self.time_now = time.time()
        self.time_prev = self.time_now
        self.target_course = TargetCourse(cx, cy)
        self.target_idx, _ = self.target_course.search_target_index(self.state)
        self.states.append(self.time_now, self.state)
        self.boost_button = 0

    def process_lidar(self, ranges, poses):
        # Calc control input
        # print(poses)
        target_speed = 10

        print("poses[0][0], poses[1][0]:", poses[0][0], poses[1][0])
        gap = 3
        if poses[0][0] < 0 + gap and poses[0][0] > 0 - gap:
            if poses[1][0] < 0 + gap and poses[1][0] > 0 - gap:
                self.boost_button = 3
        if poses[0][0] < -56 + gap and poses[0][0] > -56 - gap:
            if poses[1][0] < -42 + gap and poses[1][0] > -42 - gap:
                self.boost_button = 2
        if poses[0][0] < -78 + gap and poses[0][0] > -78 - gap:
            if poses[1][0] < -42 + gap and poses[1][0] > -42 - gap:
                self.boost_button = 0

        if poses[0][0] < -112 + gap and poses[0][0] > -112 - gap:
            if poses[1][0] < -13 + gap and poses[1][0] > -13 - gap:
                self.boost_button = 1
        if poses[0][0] < -60 + gap and poses[0][0] > -60 - gap:
            if poses[1][0] < -23 + gap and poses[1][0] > -23 - gap:
                self.boost_button = 2
        if poses[0][0] < -29 + gap and poses[0][0] > -29 - gap:
            if poses[1][0] < -30 + gap and poses[1][0] > -30 - gap:
                self.boost_button = 0
        if poses[0][0] < -23 + gap and poses[0][0] > -23 - gap:
            if poses[1][0] < -11 + gap and poses[1][0] > -11 - gap:
                self.boost_button = 4
        if poses[0][0] < -12 + gap and poses[0][0] > -12 - gap:
            if poses[1][0] < 5 + gap and poses[1][0] > 5 - gap:
                self.boost_button = 2
        if poses[0][0] < -6 + gap and poses[0][0] > -6 - gap:
            if poses[1][0] < 12 + gap and poses[1][0] > 12 - gap:
                self.boost_button = 0

        if self.boost_button == 1:
            target_speed = 15
        elif self.boost_button == 2:
            target_speed = 7
        elif self.boost_button == 3:
            target_speed = 14.6
        elif self.boost_button == 4:
            target_speed = 13.6
        else:
            target_speed = 10.8

        print("boost button:", self.boost_button)
        print("target_speed:", target_speed)
        steering_angle, self.target_idx = pure_pursuit_steer_control(
            self.state, self.target_course, self.target_idx)
        if (steering_angle > 0.05 or steering_angle < -0.05):
            target_speed = 4
        acc = proportional_control(target_speed, self.state.v)

        # print("ind", self.target_idx)
        # print(">>> steering angle:", steering_angle)
        # print("poses x y:", poses)

        self.time_now = time.time()
        time_step = self.time_now - self.time_prev
        self.time_prev = self.time_now

        self.state.update(acc, time_step, poses)  # Control vehicle

        self.states.append(time, self.state)

        # if show_animation:  # pragma: no cover
        #     plt.cla()
        #     # for stopping simulation with the esc key.
        #     plt.gcf().canvas.mpl_connect(
        #         'key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])
        #     plot_arrow(state.x, state.y, state.yaw)
        #
        #     # trajectory
        #     plt.plot(cx, cy, "-r", label="course")
        #     plt.plot(states.x, states.y, "-b", label="trajectory")
        #     plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        #     plt.pause(0.00001)

        self.speed = self.state.v

        print(">>> speed:", self.speed)
        return self.speed, steering_angle
