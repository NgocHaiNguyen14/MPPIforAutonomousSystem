import numpy as np

def quadrotor(in1, in2):
# in1: states
# in2: control inputs

    pitch_t = in1[4]
    pitch_dot_t = in1[10]
    roll_t = in1[3]
    roll_dot_t = in1[9]
    x_dot_t = in1[6]
    y_dot_t = in1[7]
    yaw_t = in1[5]
    yaw_dot_t = in1[11]
    z_dot_t = in1[8]

    u1 = in2[0]
    u2 = in2[1]
    u3 = in2[2]
    u4 = in2[3]

    t2 = np.cos(pitch_t)
    t3 = np.cos(roll_t)
    t4 = np.cos(yaw_t)
    t5 = np.sin(pitch_t)
    t6 = np.sin(roll_t)
    t7 = np.sin(yaw_t)
    t8 = pitch_t * 2.0
    t9 = roll_t * 2.0
    t10 = yaw_dot_t ** 2
    t16 = u1 + u2 + u3 + u4
    t11 = t2 ** 2
    t12 = t3 ** 2
    t13 = np.sin(t8)
    t14 = np.sin(t9)
    t15 = 1.0 / t11

    mt1 = np.array([
        x_dot_t,
        y_dot_t,
        z_dot_t,
        roll_dot_t,
        pitch_dot_t,
        yaw_dot_t,
        (t16 * (t6 * t7 + t3 * t4 * t5)) / 2.0,
        -t16 * (t4 * t6 - t3 * t5 * t7) / 2.0,
        (t2 * t3 * t16) / 2.0 - 9.81e2 / 1.0e2
    ]).reshape(-1, 1)

    mt2 = np.array([
        (t15 * (
            -1.15e2 * u2 + 1.15e2 * u4
            - 9.2e1 * t5 * u1 + 9.2e1 * t5 * u2 - 9.2e1 * t5 * u3 + 9.2e1 * t5 * u4
            + 5.5e1 * t12 * u2 - 5.5e1 * t12 * u4
            + 2.3e1 * pitch_dot_t * roll_dot_t * t13
            + 4.4e1 * t5 * t12 * u1 - 4.4e1 * t5 * t12 * u2 + 4.4e1 * t5 * t12 * u3 - 4.4e1 * t5 * t12 * u4
            - 5.5e1 * t11 * t12 * u2 + 5.5e1 * t11 * t12 * u4
            + 1.058e3 * pitch_dot_t * t2 * yaw_dot_t
            - 5.06e2 * t3 * t6 * t10 * t11
            - 5.06e2 * pitch_dot_t * t2 * t12 * yaw_dot_t
            + 5.06e2 * pitch_dot_t ** 2 * t3 * t6 * t11
            - 5.06e2 * pitch_dot_t * t2 ** 3 * t12 * yaw_dot_t
            + 5.06e2 * pitch_dot_t * roll_dot_t * t2 * t5 * t12
            - 5.5e1 * t2 * t3 * t5 * t6 * u1 + 5.5e1 * t2 * t3 * t5 * t6 * u3
            + 5.06e2 * roll_dot_t * t3 * t5 * t6 * t11 * yaw_dot_t
        )) / 5.52e2
    ])

    mt3 = np.array([
        (-1.0 / 5.52e2 * (
            6.0e1 * t2 * u1 - 6.0e1 * t2 * u3
            + 2.2e1 * t14 * u1 - 2.2e1 * t14 * u2 + 2.2e1 * t14 * u3 - 2.2e1 * t14 * u4
            + 5.5e1 * t2 * t12 * u1 - 5.5e1 * t2 * t12 * u3
            + 5.52e2 * roll_dot_t * t11 * yaw_dot_t
            + 5.06e2 * t5 * t10 * t11 * t12
            + 5.5e1 * t3 * t5 * t6 * u2 - 5.5e1 * t3 * t5 * t6 * u4
            - 5.06e2 * roll_dot_t * t11 * t12 * yaw_dot_t
            + 5.06e2 * pitch_dot_t * roll_dot_t * t2 * t3 * t6
            - 5.06e2 * pitch_dot_t * t2 * t3 * t5 * t6 * yaw_dot_t
        )) / t2
    ])

    mt4 = np.array([
        (t15 * (
            -9.2e1 * u1 + 9.2e1 * u2 - 9.2e1 * u3 + 9.2e1 * u4
            - 1.15e2 * t5 * u2 + 1.15e2 * t5 * u4
            + 4.4e1 * t12 * u1 - 4.4e1 * t12 * u2 + 4.4e1 * t12 * u3 - 4.4e1 * t12 * u4
            + 4.6e1 * pitch_dot_t * roll_dot_t * t2
            + 5.5e1 * t5 * t12 * u2 - 5.5e1 * t5 * t12 * u4
            + 5.29e2 * pitch_dot_t * t13 * yaw_dot_t
            + 5.06e2 * pitch_dot_t * roll_dot_t * t2 * t12
            - 5.5e1 * t2 * t3 * t6 * u1 + 5.5e1 * t2 * t3 * t6 * u3
            - 5.06e2 * t3 * t5 * t6 * t10 * t11
            - 5.06e2 * pitch_dot_t * t2 * t5 * t12 * yaw_dot_t
            + 5.06e2 * roll_dot_t * t3 * t6 * t11 * yaw_dot_t
        )) / 5.52e2
    ])

    f = np.vstack((mt1, mt2, mt3, mt4))
    flatten_list = f.flatten().tolist()
    return flatten_list

