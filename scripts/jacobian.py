import numpy as np

def compute_jacobian(q):
    """
    Inputs:
    q: [q1, q2, q3, q4] joint angles of the arm (radians)
    Outputs:
    Jacobian matrix: 3x4 (rows: x, y, z; columns: joint velocities)
    """
    q1, q2, q3, q4 = q[:4]

    # Robot measurements in meters
    L2 = 142 / 1000.0          
    L3 = 158.8 / 1000.0       
    d_grip = 56.5 / 1000.0 
    d_base_horizontal = 13.2 / 1000.0

    # Trigonometric values
    sin_q1 = np.sin(q1)
    cos_q1 = np.cos(q1)
    sin_q2 = np.sin(q2)
    cos_q2 = np.cos(q2)
    sin_q3 = np.sin(q3)
    cos_q3 = np.cos(q3)

    # Planar reach
    r = d_base_horizontal - (L2 * sin_q2) + (L3 * cos_q3) + d_grip

    # Partial derivatives
    dr_dq2 = -L2 * cos_q2
    dr_dq3 = -L3 * sin_q3
    dz_dq2 = L2 * sin_q2
    dz_dq3 = -L3 * cos_q3

    # Jacobian terms
    dx_dq1 = -r * sin_q1
    dx_dq2 = dr_dq2 * cos_q1
    dx_dq3 = dr_dq3 * cos_q1

    dy_dq1 = r * cos_q1
    dy_dq2 = dr_dq2 * sin_q1
    dy_dq3 = dr_dq3 * sin_q1

    dz_dq1 = 0.0
    dz_dq2 = dz_dq2
    dz_dq3 = dz_dq3

    # Build Jacobian
    J = np.array([
        [dx_dq1, dx_dq2, dx_dq3, 0.0],
        [dy_dq1, dy_dq2, dy_dq3, 0.0],
        [dz_dq1, dz_dq2, dz_dq3, 0.0]
    ])

    return J
