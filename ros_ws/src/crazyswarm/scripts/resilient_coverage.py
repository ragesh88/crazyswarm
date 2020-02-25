#!/usr/bin/env python

from __future__ import print_function
from get_rad_tune import get_rad_tune

import argparse
import os
import os.path
import rospy
import math

import _multiprocessing

import numpy as np
import csv

import matlab.engine

from pycrazyswarm import *
import uav_trajectory

import pycrazyswarm.cfsim.cffirmware as firm


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("path",
    #     help="directory containing numbered subdirectories for each robot," +
    #         "each of which contains numbered <n>.csv files for each formation change")
    # args, unknown = parser.parse_known_args()

    # data paths for the scripts
    # path to the matlab script generating the paths btw strt and goal
    trj_gen_pth = './crazyswarm-planning/'
    # path to the matlab script that computes the initial positions of the robots
    init_pos_pth = './resilient_coverage_team_selection/experiments/'
    # path to the matlab script that computes the reconfigured positions
    reconf_pos_pth = './resilient_coverage_team_selection/experiments/'
    # path to the folder containing the generated path
    data_folder = '/media/ragesh/Disk1/data/resilient_coverage/exp/'

    # timing parameters
    timescale = 1.0
    pause_between = 1.5
    takeoff_time = 3.0
    land_time = 4.0

    # parameters for the team
    A_n = 7  # number of robots
    # define the possible quadrotor states
    HOVER = 0
    MOVE = 1
    FAILED = 2
    LANDING = 3

    #
    # CRAZYSWARM INITIALIZATION
    #
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    crazyflies = allcfs.crazyflies

    # prepare the robot for take off
    # check that crazyflies.yaml initial positions match sequences.
    # only compare xy positions.
    # assume that cf id numerical order matches sequence order,
    # but id's don't need to be 1..N.
    crazyflies = sorted(crazyflies, key=lambda cf: cf.id)
    # the map between matlab index and robot id of the robots
    Rob_lab2id = {i: cf.id for cf, i in zip(crazyflies, range(1, A_n+1))}
    # the map between robot id and matlab index of the robots
    Rob_id2lab = {cf.id: i for cf, i in zip(crazyflies, range(1, A_n + 1))}
    # the dictionary containing the states {HOVER, MOVE, FAILED, LANDING} of different robots
    Rob_state = {cf.id: MOVE for cf in crazyflies}

    # make the robots to take off all together
    # initial start pos to generate trajectories for the robots
    # to find initial max coverage position
    Rob_start_pos = np.array([[0.0, -1.5, 1.5],
                              [0.0, -1.0, 1.5],
                              [0.0, -0.5, 1.5],
                              [0.0, 0.0, 1.5],
                              [0.0, 0.5, 1.5],
                              [0.0, 1.0, 1.5],
                              [0.0, 1.5, 1.5]])

    # generate the initial coordinates and trajectory for the robots
    eng = matlab.engine.start_matlab()
    eng.cd(init_pos_pth)
    mat_out = eng.exp_init_traj(A_n, Rob_start_pos, data_folder, nargout=3)
    Rob_active_pos = mat_out[0]
    b_box = mat_out[1]
    data_folder = mat_out[2]

    # ...trajectory sequences...
    robot_dirs = sorted(os.listdir(data_folder), key=int)
    seqs = [load_all_csvs(os.path.join(data_folder, d)) for d in robot_dirs]
    steps = len(seqs[0])

    init_positions = np.stack([cf.initialPosition for cf in crazyflies])
    evals = [seq[0].eval(0.0).pos for seq in seqs]
    traj_starts = np.stack(evals)
    errs = init_positions - traj_starts
    errnorms = np.linalg.norm(errs[:, :2], axis=1)
    # check if x,y coordinate of the robot start positions
    # match with that of the initial trajectories
    assert not np.any(np.abs(errnorms) > 0.1)

    print("initial trajectory loading complete")
    # validate sequences w.r.t. each other
    assert all(len(seq) == steps for seq in seqs)
    for i in range(steps):
        agent_lens = [seq[i].duration for seq in seqs]
        assert all(agent_lens == agent_lens[0])
    step_lens = [t.duration for t in seqs[0]]

    print("validation complete")

    # planners for takeoff and landing
    planners = [firm.planner() for cf in crazyflies]
    for p in planners:
        firm.plan_init(p)

    # takeoff
    print("takeoff")
    z_init = traj_starts[0, 2]

    for cf, p in zip(crazyflies, planners):
        p.lastKnownPosition = cf.position()
        vposition = firm.mkvec(*p.lastKnownPosition)
        firm.plan_takeoff(p, vposition, 0.0, z_init, takeoff_time, 0.0)

    poll_planners(crazyflies, timeHelper, planners, takeoff_time)
    end_pos = np.stack([cf.position() for cf in crazyflies])


    # pause - all is well...
    hover(crazyflies, timeHelper, end_pos, pause_between)

    # if everything goes well then the robot would have took off by now
    # would be hover at the desired positions

    # transposed copy - by timestep instead of robot
    seqs_t = zip(*seqs)

    Rob_active_mat = np.ones((A_n, 1))  # vector indicating active robots
    # list containing the matlab indices of the active robots
    Rob_active_lab_mat = range(1, A_n+1)



    # main loop!
    for i_1 in range(A_n-1):
        # get the position of all active robots
        j_1 = 0
        for lab in Rob_active_lab_mat:
            for cf in crazyflies:
                if cf.id == Rob_lab2id[lab]:
                    Rob_active_pos[j_1, :] = cf.position()[:, 0:2]
                    j_1 += 1
                    break
        Rob_active_pos = Rob_active_pos[:, 0:j_1]
        # randomly select a robot to fail
        # we assume its a matlab index
        indx = math.floor(np.random.uniform()*(Rob_active_mat.sum()))
        if indx == 0:
            indx = 1
        fail_rob_lab_mat = Rob_active_lab_mat[indx - 1]
        Rob_active_mat[fail_rob_lab_mat - 1] = 0
        # find the position of the failed robot
        for cf in crazyflies:
            if cf.id == Rob_lab2id[fail_rob_lab_mat]:
                fail_rob_pos = cf.position()[:, 0:2]
        # get radius tune from the user
        rad_tune = get_rad_tune(b_box, Rob_active_lab_mat, fail_rob_lab_mat, Rob_active_pos, fail_rob_pos)
        # set state of failed robot and make it land
        Rob_state[fail_rob_lab_mat] = FAILED
        # land() # TODO add a function to make the robot land
        # call the matlab function to get trajectories, robot labels in failed neighborhood
        mat_out = matlab_get_trajectories(eng) # TODO complete this
        fail_trjs = mat_out[0]  # trajectories of the robots in the failed robot nbh
        fail_rob_nbh = mat_out[1]  # trajectories matches the elements and are sorted
        com_rob_nbh = mat_out[2]
        Rob_active_lab_mat = com_rob_nbh + fail_rob_nbh
        # set the states of the robots
        for lab in Rob_active_lab_mat:
            if lab in fail_rob_nbh:
                Rob_state[lab] = MOVE
            if lab in com_rob_nbh:
                Rob_state[lab] = HOVER







    for step in range(steps):

        # move - new configuration after capability loss
        print("executing trajectory", step, "/", steps)
        poll_trajs(crazyflies, timeHelper, seqs_t[step], timescale)
        end_pos = np.stack([cf.position() for cf in crazyflies])

        # done with this step's trajs - hover for a few sec
        hover(crazyflies, timeHelper, end_pos, pause_between)

        # change the LEDs - another capability loss
        # if step < steps - 1:
        #     set_colors(step + 2)

        # hover some more
        hover(crazyflies, timeHelper, end_pos, pause_between)

    # land
    print("landing")

    end_pos = np.stack([cf.position() for cf in crazyflies])
    for cf, p, pos in zip(crazyflies, planners, end_pos):
        vposition = firm.mkvec(*pos)
        firm.plan_land(p, vposition, 0.0, 0.06, land_time, 0.0)

    poll_planners(crazyflies, timeHelper, planners, land_time)

    # cut power
    print("sequence complete.")
    allcfs.emergency()


def poll_trajs(crazyflies, timeHelper, trajs, timescale):
    duration = trajs[0].duration
    start_time = rospy.Time.now()
    rate = rospy.Rate(100) # hz
    while not rospy.is_shutdown():
        t = (rospy.Time.now() - start_time).to_sec() / timescale
        if t > duration:
            break
        for cf, traj in zip(crazyflies, trajs):
            ev = traj.eval(t)
            cf.cmdFullState(
                ev.pos,
                ev.vel,
                ev.acc,
                ev.yaw,
                ev.omega)
        rate.sleep()
        #timeHelper.sleep(1e-6)


def my_poll_trajs(crazyflies, timeHelper, trajs, timescale, Robot_state, Rob_lab2id, Rob_id2lab,
                  com_rob_nbh, fail_rob_nbh):
    duration = trajs[0].duration
    rob_lab_mat = sorted(Robot_state.keys())
    start_time = rospy.Time.now()
    rate = rospy.Rate(100)  # hz
    zero = np.zeros(3)
    while not rospy.is_shutdown():
        t = (rospy.Time.now() - start_time).to_sec() / timescale
        if t > duration:
            break
        for cf, lab in zip(crazyflies, rob_lab_mat):
            if lab in fail_rob_nbh:
                # they need to move
                indx = fail_rob_nbh.index(lab)
                ev = trajs[indx].eval(t)
                cf.cmdFullState(
                    ev.pos,
                    ev.vel,
                    ev.acc,
                    ev.yaw,
                    ev.omega)
            if lab in com_rob_nbh:
                # they need to hover
                indx = com_rob_nbh.index(lab)
                cf.cmdFullState(
                    cf.position(),
                    zero,
                    zero,
                    0.0,
                    zero)
        rate.sleep()


def poll_planners(crazyflies, timeHelper, planners, duration):
    start_time = rospy.Time.now()
    rate = rospy.Rate(100) # hz
    while not rospy.is_shutdown():
        t = (rospy.Time.now() - start_time).to_sec()
        if t > duration:
            break
        for cf, planner in zip(crazyflies, planners):
            ev = firm.plan_current_goal(planner, t)
            cf.cmdFullState(
                firm2arr(ev.pos),
                firm2arr(ev.vel),
                firm2arr(ev.acc),
                ev.yaw,
                firm2arr(ev.omega))
        rate.sleep()
        #timeHelper.sleep(1e-6)


def hover(crazyflies, timeHelper, positions, duration):
    start_time = rospy.Time.now()
    rate = rospy.Rate(100) # hz
    zero = np.zeros(3)
    while not rospy.is_shutdown():
        t = (rospy.Time.now() - start_time).to_sec()
        if t > duration:
            break
        for cf, pos in zip(crazyflies, positions):
            cf.cmdFullState(
                pos,
                zero,
                zero,
                0.0,
                zero)
        rate.sleep()
        #timeHelper.sleep(1e-6)


def firm2arr(vec):
    return np.array([vec.x, vec.y, vec.z])


def load_all_csvs(path):
    csvs = os.listdir(path)
    csvs = sorted(csvs, key=lambda s: int(os.path.splitext(s)[0])) # numerical order
    names, exts = zip(*[os.path.splitext(os.path.basename(f)) for f in csvs])
    assert all(e == ".csv" for e in exts)
    steps = len(names)
    assert set(names) == set([str(i) for i in range(1, steps + 1)])
    trajs = [uav_trajectory.Trajectory() for _ in range(steps)]
    for t, csv in zip(trajs, csvs):
        t.loadcsv(os.path.join(path, csv))
    return trajs


def matlab_get_trajectories(eng):
    # implement the function
    pass


if __name__ == "__main__":
    main()
