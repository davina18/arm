#!/usr/bin/env python3
import numpy as np

def compute_forward_kinematics(q):
    """
    Inputs:
    q: [q1, q2, q3, q4] joint angles of the arm
    Outputs:
    EE position: (x, y, z) end effector position in the arm base frame
    """

    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    
    # Robot measurements in meters
    L2 = 142 / 1000.0
    L3 = 158.8 / 1000.0
    d_grip = 56.5 / 1000.0
    d_base_vertical = 108 / 1000.0
    d_base_horizontal = 13.2 / 1000.0
    h_wrist = 72.2 / 1000.0

    # Planar reach
    r = d_base_horizontal - (L2 * np.sin(q2)) + (L3 * np.cos(q3)) + d_grip
    z_E = -d_base_vertical - (L2 * np.cos(q2)) - (L3 * np.sin(q3)) + h_wrist

    # End effector position
    x_E = r * np.cos(q1)
    y_E = r * np.sin(q1)
    z_E = z_E

    return np.array([x_E, y_E, z_E]).reshape(3, 1)