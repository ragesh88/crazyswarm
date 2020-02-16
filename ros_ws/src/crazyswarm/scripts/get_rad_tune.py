import tkinter as tk
import numpy as np


def get_rad_tune(b_box, Rob_active_lab, fail_rob_lab, Rob_active_pos,
                 fail_rob_pos, rad_tune=0.5):
    """
    This function creates a tkinter gui enabling the user to chose a radius
    of appropriate size to pick the surrounding neighbours
    :param b_box: the numpy array containing the limits of the boundary
    :param Rob_active_lab: the numpy array containing labels of the active robots
    :param fail_rob_lab: the label of the failed robot
    :param Rob_active_pos: the positions of the active robots
    :param fail_rob_pos: the position of the failed robot
    :param rad_tune: the radius around failed robot
    :return: the user defined radius
    """
    # compute the canvas length and width
    scale = 100
    cns_orgin = [20, 100]
    cns_w_off = cns_orgin[0]
    cns_h_off = cns_orgin[1]
    canvas_w = int((b_box[0, 1]-b_box[0, 0])*scale) + 2*cns_w_off
    canvas_h = int((b_box[1, 1]-b_box[1, 0])*scale) + 2*cns_h_off

    # compute the robot positions in the canvas coordinates
    rob_act_cns = []
    for i in range(Rob_active_pos.shape[0]):
        rob_act_cns.append([int((Rob_active_pos[i, 0] - b_box[0, 0])*scale) + cns_orgin[0],
                            canvas_h - int((Rob_active_pos[i, 1] - b_box[1, 0])*scale + cns_orgin[1])])
    # compute the fail robot canvas coordinates
    fail_rob_pos_cns = [int((fail_rob_pos[0, 0] - b_box[0, 0])*scale) + cns_orgin[0],
                        canvas_h - int((fail_rob_pos[0, 1] - b_box[1, 0])*scale + cns_orgin[1])]
    # radius in canvas
    rad_tune_cns = int(rad_tune*scale)

    # construct the canvas and mark the robot positions
    marker_size = 10

    master = tk.Tk()

    w = tk.Canvas(master, width=canvas_w, height=canvas_h)

    w.pack()

    for i in range(Rob_active_pos.shape[0]):
        if Rob_active_lab[i] == fail_rob_lab:
            w.create_rectangle(rob_act_cns[i][0] - marker_size,
                               rob_act_cns[i][1] - marker_size,
                               rob_act_cns[i][0] + marker_size, rob_act_cns[i][1] + marker_size,
                               fill="#F1091E")
            w.create_text(rob_act_cns[i][0], rob_act_cns[i][1], text=str(Rob_active_lab[i]))
        else:
            w.create_rectangle(rob_act_cns[i][0] - marker_size, rob_act_cns[i][1] - marker_size,
                               rob_act_cns[i][0] + marker_size, rob_act_cns[i][1] + marker_size,
                               fill="#F1EA09")
            w.create_text(rob_act_cns[i][0], rob_act_cns[i][1], text=str(Rob_active_lab[i]))

    # draw a line to mark the separation between the controls and the visualization
    w.create_line(0, canvas_h - cns_h_off, canvas_w, canvas_h - cns_h_off, width=3)

    # create a slider to tune the neighborhood
    max_dim = max([b_box[0][1]-b_box[0][0], b_box[1][1]-b_box[1][0]])
    w1 = tk.Scale(master, from_=0.0, to=max_dim, resolution=0.5, orient=tk.HORIZONTAL, length=canvas_w,
                  showvalue=1)
    w1.set(rad_tune)
    w1.pack()

    # rectangle around the failed robot
    nbh_rec = w.create_rectangle(fail_rob_pos_cns[0] - rad_tune_cns,
                                 fail_rob_pos_cns[1] - rad_tune_cns,
                                 fail_rob_pos_cns[0] + rad_tune_cns,
                                 fail_rob_pos_cns[1] + rad_tune_cns)
    def rect(event):
        rad_tune_1 = w1.get()
        rad_tune_1 = int(rad_tune_1*scale)
        w.coords(nbh_rec, fail_rob_pos_cns[0] - rad_tune_1,
                           fail_rob_pos_cns[1] - rad_tune_1,
                           fail_rob_pos_cns[0] + rad_tune_1,
                           fail_rob_pos_cns[1] + rad_tune_1)
    w1.bind("<ButtonRelease-1>", rect)

    # set the done button

    done_butt = tk.Button(master, text="DONE", command=master.quit).pack()

    tk.mainloop()
    # return selected radius
    rad_tune = w1.get()#/float(scale)
    print "inside function", rad_tune
    return rad_tune


if __name__ == "__main__":
    b_box = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    Rob_active_lab = range(1, 11)
    fail_rob_lab = 2
    Rob_active_pos = np.array([[-0.75000,	-0.7500],
                               [1.0000,    1.0000],
                               [-1.0000,    1.2500],
                               [1.2500,   -1.0000],
                               [-1.0000,   -1.2500],
                               [-1.2500,    0.7500],
                               [1.2500,    1.0000],
                               [1.0000,   -1.2500],
                               [-1.2500,   -1.0000],
                               [-0.7500,    2.0000]])
    fail_rob_pos = np.array([[1.0000,    1.0000]])
    print get_rad_tune(b_box, Rob_active_lab, fail_rob_lab, Rob_active_pos, fail_rob_pos)
