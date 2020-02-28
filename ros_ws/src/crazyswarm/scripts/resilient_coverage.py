#!/usr/bin/env python

from __future__ import print_function
from get_rad_tune import get_rad_tune

import argparse
import os
import os.path
import math

import pdb
import copy

import _multiprocessing

import numpy as np
import csv

import matlab.engine

from pycrazyswarm import *
import uav_trajectory

import pycrazyswarm.cfsim.cffirmware as firm


# define the possible quadrotor states
HOVER = 0
MOVE = 1
FAILED = 2
LANDING = 3
POLL_RATE = 100  # Hz

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
    Rob_start_pos = [[0.0, -1.5, 1.5],
                              [0.0, -1.0, 1.5],
                              [0.0, -0.5, 1.5],
                              [0.0, 0.0, 1.5],
                              [0.0, 0.5, 1.5],
                              [0.0, 1.0, 1.5],
                              [0.0, 1.5, 1.5]]

    # generate the initial coordinates and trajectory for the robots
    eng = matlab.engine.start_matlab()
    eng.cd(init_pos_pth)
    eng.strt_exp_scrpt(nargout=0)
    eng.cd(init_pos_pth)
    mat_Rob_start_pos = matlab.double(Rob_start_pos)
    mat_out = eng.exp_init_traj(A_n, mat_Rob_start_pos, data_folder, matlab.double([]), nargout=3)
    Rob_active_pos = np.array(mat_out[0])
    b_box = np.array(mat_out[1])
    data_folder = mat_out[2]

    # set active robots to color red
    for cf in crazyflies:
        cf.setLEDColor(1, 0, 0)

    # ...trajectory sequences...
    robot_dirs = sorted(os.listdir(data_folder + '/trajectories/'), key=int)
    seqs = [load_all_csvs(os.path.join(data_folder + '/trajectories', d)) for d in robot_dirs]
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

    # make the robot move to the desired positions
    poll_trajs(crazyflies, timeHelper, seqs_t[0], timescale)

    Rob_active_mat = np.ones((A_n, 1))  # vector indicating active robots
    # list containing the matlab indices of the active robots
    Rob_active_lab_mat = range(1, A_n+1)

    # main loop!
    for i_1 in range(4):
        # get the position of all active robots
        j_1 = 0
        for lab in Rob_active_lab_mat:
            for cf in crazyflies:
                if cf.id == Rob_lab2id[lab]:
                    Rob_active_pos[j_1, :] = cf.position()[0:2]
                    j_1 += 1
                    break
        Rob_active_pos = Rob_active_pos[0:j_1, :]
        # randomly select a robot to fail
        # we assume its a matlab index
        indx = int(math.floor(np.random.uniform()*(Rob_active_mat.sum())))
        if indx == 0:
            indx = 1
        fail_rob_lab_mat = Rob_active_lab_mat[indx - 1]
        Rob_active_mat[fail_rob_lab_mat - 1] = 0
        # find the position of the failed robot
        fail_rob_pos = 0.0
        for cf in crazyflies:
            if cf.id == Rob_lab2id[fail_rob_lab_mat]:
                fail_rob_pos = cf.position()[0:2]
        # get radius tune from the user
        # TODO uncomment below after debugging
        rad_tune = get_rad_tune(b_box, Rob_active_lab_mat, fail_rob_lab_mat, Rob_active_pos, fail_rob_pos)
        # set the led color of the robot to black
        for cf in crazyflies:
            if cf.id == Rob_lab2id[fail_rob_lab_mat]:
                cf.setLEDColor(0, 0, 0)
                break
        # set state of failed robot and make it land
        Rob_state[fail_rob_lab_mat] = LANDING
        land(crazyflies, timeHelper, Rob_state, land_time)
        # set the state of the failed robot to failed
        Rob_state[fail_rob_lab_mat] = FAILED
        # call the matlab function to get trajectories, robot labels in failed neighborhood
        # mat_out = matlab_get_trajectories(eng)
        mat_out = eng.exp_inter_traj(A_n, rad_tune, indx,
                                     matlab.double(Rob_active_lab_mat),
                                     matlab.double(Rob_active_pos.tolist()), data_folder, nargout=2)
        # load the trajectories from the csv files
        robot_dirs = sorted(os.listdir(data_folder + '/trajectories/'), key=int)
        # trajectories of the robots in the failed robot nbh
        fail_trjs = [load_all_csvs(os.path.join(data_folder + '/trajectories', d)) for d in robot_dirs]
        fail_rob_nbh = np.array(mat_out[0]).tolist()  # trajectories matches the elements and are sorted
        fail_rob_nbh = [int(f[0]) for f in fail_rob_nbh]
        fail_rob_nbh = sorted(fail_rob_nbh)
        com_rob_nbh = np.array(mat_out[1]).tolist()
        if type(com_rob_nbh) != float and len(com_rob_nbh) > 0:
            com_rob_nbh = [int(f[0]) for f in com_rob_nbh]
            Rob_active_lab_mat = copy.deepcopy(com_rob_nbh + fail_rob_nbh)
        else:
            if type(com_rob_nbh) == float:
                Rob_active_lab_mat = copy.deepcopy(fail_rob_nbh)
                Rob_active_lab_mat.append(int(com_rob_nbh))
                com_rob_nbh = [int(com_rob_nbh)]
            else:
                Rob_active_lab_mat = copy.deepcopy(fail_rob_nbh)
                com_rob_nbh = []
        print("Fail label", fail_rob_nbh)
        print("Com label", com_rob_nbh)
        print("Active label", Rob_active_lab_mat)
        # set the states of the robots
        for lab in Rob_active_lab_mat:
            if lab in fail_rob_nbh:
                Rob_state[lab] = MOVE
            if lab in com_rob_nbh:
                Rob_state[lab] = HOVER
        # execute the trajectories
        fail_trjs_t = zip(*fail_trjs)
        my_poll_trajs(crazyflies, timeHelper, fail_trjs_t[0], timescale, Rob_state, com_rob_nbh, fail_rob_nbh)
        # reset the robot states to hover
        for lab in Rob_active_lab_mat:
            Rob_state[lab] = HOVER


def poll_trajs(crazyflies, timeHelper, trajs, timescale):
    duration = trajs[0].duration
    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = (timeHelper.time() - start_time) / timescale
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
        timeHelper.sleepForRate(POLL_RATE)


def my_poll_trajs(crazyflies, timeHelper, trajs, timescale, Robot_state, com_rob_nbh, fail_rob_nbh):
    duration = trajs[0].duration
    rob_lab_mat = sorted(Robot_state.keys())
    start_time = timeHelper.time()
    zero = np.zeros(3)
    while not timeHelper.isShutdown():
        t = (timeHelper.time() - start_time) / timescale
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
                cf.cmdFullState(
                    cf.position(),
                    zero,
                    zero,
                    0.0,
                    zero)
        timeHelper.sleepForRate(POLL_RATE)


def land(crazyflies, timeHelper, Robot_state, duration):
    start_time = timeHelper.time()
    rob_lab_mat = sorted(Robot_state.keys())
    zero = np.zeros(3)
    while not timeHelper.isShutdown():
        t = (timeHelper.time() - start_time)
        if t > duration:
            break
        for cf, lab in zip(crazyflies, rob_lab_mat):
            if Robot_state[lab] == LANDING:
                # command the drone to land
                pos = cf.position()
                pos[2] = 0.0
                cf.cmdFullState(
                    pos,
                    zero,
                    zero,
                    0.0,
                    zero)
            else:
                if Robot_state[lab] != FAILED:
                    # make the non failed robots hover
                    cf.cmdFullState(
                    cf.position(),
                    zero,
                    zero,
                    0.0,
                    zero)
        timeHelper.sleepForRate(POLL_RATE)


def poll_planners(crazyflies, timeHelper, planners, duration):
    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
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
        timeHelper.sleepForRate(POLL_RATE)


def hover(crazyflies, timeHelper, positions, duration):
    start_time = timeHelper.time()
    zero = np.zeros(3)
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > duration:
            break
        for cf, pos in zip(crazyflies, positions):
            cf.cmdFullState(
                pos,
                zero,
                zero,
                0.0,
                zero)
        timeHelper.sleepForRate(POLL_RATE)



def firm2arr(vec):
    return np.array([vec.x, vec.y, vec.z])


def load_all_csvs(path):
    csvs = os.listdir(path)
    csvs = sorted(csvs, key=lambda s: int(os.path.splitext(s)[0]))  # numerical order
    names, exts = zip(*[os.path.splitext(os.path.basename(f)) for f in csvs])
    assert all(e == ".csv" for e in exts)
    steps = len(names)
    assert set(names) == set([str(i) for i in range(1, steps + 1)])
    trajs = [uav_trajectory.Trajectory() for _ in range(steps)]
    for t, csv in zip(trajs, csvs):
        t.loadcsv(os.path.join(path, csv))
    return trajs



if __name__ == "__main__":
    main()
