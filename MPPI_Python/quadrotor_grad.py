import numpy as np

def quadrotor_grad(in1, in2):
# in1: states
# in2: control inputs
    pitch_t = in1[4][0]
    pitch_dot_t = in1[10][0]
    roll_t = in1[3][0]
    roll_dot_t = in1[9][0]
    yaw_t = in1[5][0]
    yaw_dot_t = in1[11][0]
    
    u1 = in2[0][0]
    u2 = in2[1][0]
    u3 = in2[2][0]
    u4 = in2[3][0]

    t2 = np.cos(pitch_t)
    t3 = np.cos(roll_t)
    t4 = np.cos(yaw_t)
    t5 = np.sin(pitch_t)
    t6 = np.sin(roll_t)
    t7 = np.sin(yaw_t)
    t8 = pitch_t * 2.0
    t9 = pitch_dot_t ** 2
    t10 = roll_t * 2.0
    t11 = yaw_dot_t ** 2
    t25 = u1 + u2 + u3 + u4
    t12 = np.cos(t8)
    t13 = t2 ** 2
    t14 = t2 ** 3
    t15 = np.cos(t10)
    t16 = t3 ** 2
    t17 = np.sin(t8)
    t18 = t5 ** 2
    t19 = np.sin(t10)
    t20 = t6 ** 2
    t21 = t2 * 60.0
    t22 = 1.0 / t2
    t26 = t5 * 92.0
    t27 = t5 * 115.0
    t32 = (t2 * t3) / 2.0
    t33 = (t4 * t6) / 2.0
    t34 = (t6 * t7) / 2.0
    t41 = t2 * t3 * t6 * 55.0
    t42 = t3 * t5 * t6 * 55.0
    t43 = (t3 * t5 * t7) / 2.0
    t50 = (t3 * t4 * t5) / 2.0
    t23 = 1.0 / t13
    t24 = 1.0 / t14
    t28 = -t26
    t29 = t16 * 44.0
    t30 = t16 * 55.0
    t31 = t19 * 22.0
    t37 = -t33
    t44 = -t5 * t16 * 44.0
    t45 = -t5 * t16 * 55.0
    t52 = t5 * t41
    t53 = pitch_dot_t * t2 * t16 * 506.0
    t54 = -t2 * t16 * u3 * 55.0
    t59 = roll_dot_t * t13 * t16 * yaw_dot_t * 506.0
    t60 = -pitch_dot_t * t2 * t5 * t16 * yaw_dot_t * 506.0
    t62 = t5 * t11 * t13 * t16 * 506.0
    t63 = t34 + t50
    t35 = -t29
    t36 = -t30
    t38 = t2 * t30
    t39 = t5 * t29
    t40 = t5 * t30
    t51 = t13 * t30
    t55 = t45 * u4
    t56 = roll_dot_t * t53
    t57 = t5 * t53
    t61 = -t59
    t64 = t37 + t43
    t46 = t38 * u1
    t47 = t38 * u3
    t48 = t40 * u2

    et1 = (t23 * (t2 * u1 * -9.2e+1 + t2 * u2 * 9.2e+1 - t2 * u3 * 9.2e+1 + t2 * u4 * 9.2e+1 + pitch_dot_t * roll_dot_t * t12 * 4.6e+1 - t2 * t16 * u2 * 4.4e+1 - t2 * t16 * u4 * 4.4e+1 + t2 * t29 * u1 + t2 * t29 * u3 - pitch_dot_t * t5 * yaw_dot_t * 1.058e+3 + pitch_dot_t * roll_dot_t * t13 * t16 * 5.06e+2 - pitch_dot_t * roll_dot_t * t16 * t18 * 5.06e+2 - t3 * t6 * t13 * u1 * 5.5e+1 + t2 * t5 * t16 * u2 * 1.1e+2 + t3 * t6 * t13 * u3 * 5.5e+1 - t2 * t5 * t16 * u4 * 1.1e+2 + t3 * t6 * t18 * u1 * 5.5e+1 - t3 * t6 * t18 * u3 * 5.5e+1 + pitch_dot_t * t5 * t16 * yaw_dot_t * 5.06e+2 - t2 * t3 * t5 * t6 * t9 * 1.012e+3 + t2 * t3 * t5 * t6 * t11 * 1.012e+3 + pitch_dot_t * t5 * t13 * t16 * yaw_dot_t * 1.518e+3 + roll_dot_t * t3 * t6 * t14 * yaw_dot_t * 5.06e+2 - roll_dot_t * t2 * t3 * t6 * t18 * yaw_dot_t * 1.012e+3)) / 5.52e+2

    et2 = (t5 * t24 * (u2 * -1.15e+2 + u4 * 1.15e+2 + t5 * t56 - t5 * u1 * 9.2e+1 - t5 * u3 * 9.2e+1 - t16 * u4 * 5.5e+1 + t26 * u2 + t26 * u4 + t30 * u2 + t39 * u1 + t39 * u3 + t44 * u2 + t44 * u4 + t51 * u4 + t52 * u3 + pitch_dot_t * roll_dot_t * t17 * 2.3e+1 - t13 * t16 * u2 * 5.5e+1 + pitch_dot_t * t2 * yaw_dot_t * 1.058e+3 + t3 * t6 * t9 * t13 * 5.06e+2 - t3 * t6 * t11 * t13 * 5.06e+2 - pitch_dot_t * t2 * t16 * yaw_dot_t * 5.06e+2 - pitch_dot_t * t14 * t16 * yaw_dot_t * 5.06e+2 - t2 * t3 * t5 * t6 * u1 * 5.5e+1 + roll_dot_t * t3 * t5 * t6 * t13 * yaw_dot_t * 5.06e+2)) / 2.76e+2

    et3 = t23 * (t2 * u2 * 1.15e+2 - t2 * u4 * 1.15e+2 + t38 * u4 + t42 * u3 + pitch_dot_t * roll_dot_t * t5 * 4.6e+1 - t2 * t16 * u2 * 5.5e+1 - pitch_dot_t * t12 * yaw_dot_t * 1.058e+3 + pitch_dot_t * roll_dot_t * t5 * t16 * 5.06e+2 + t3 * t6 * t11 * t14 * 5.06e+2 - t3 * t5 * t6 * u1 * 5.5e+1 + pitch_dot_t * t13 * t16 * yaw_dot_t * 5.06e+2 - pitch_dot_t * t16 * t18 * yaw_dot_t * 5.06e+2 - t2 * t3 * t6 * t11 * t18 * 1.012e+3 + roll_dot_t * t2 * t3 * t5 * t6 * yaw_dot_t * 1.012e+3) * (-1.0 / 5.52e+2)

    et4 = (t5 * t24 * (t48 + t55 + t56 + t60 - u1 * 9.2e+1 + u2 * 9.2e+1 - u3 * 9.2e+1 + u4 * 9.2e+1 - t5 * u2 * 1.15e+2 - t16 * u2 * 4.4e+1 - t16 * u4 * 4.4e+1 + t29 * u1 + t27 * u4 + t29 * u3 + t41 * u3 + pitch_dot_t * roll_dot_t * t2 * 4.6e+1 + pitch_dot_t * t17 * yaw_dot_t * 5.29e+2 - t2 * t3 * t6 * u1 * 5.5e+1 - t3 * t5 * t6 * t11 * t13 * 5.06e+2 + roll_dot_t * t3 * t6 * t13 * yaw_dot_t * 5.06e+2)) / 2.76e+2

    mt1 = [0.0] * 42 + [(t25 * (t3 * t7 - t4 * t5 * t6)) / 2.0, t25 * (t3 * t4 + t5 * t6 * t7) * (-1.0 / 2.0), t2 * t6 * t25 * (-1.0 / 2.0)]

    mt2 = [(t23 * (t5 * t47 + t5 * t59 + t9 * t13 * t16 * 5.06e+2 - t11 * t13 * t16 * 5.06e+2 - t9 * t13 * t20 * 5.06e+2 + t11 * t13 * t20 * 5.06e+2 - 
                   t3 * t6 * u2 * 1.1e+2 + t3 * t6 * u4 * 1.1e+2 + t2 * t45 * u1 - t3 * t5 * t6 * u1 * 8.8e+1 + t3 * t5 * t6 * u2 * 8.8e+1 - 
                   t3 * t5 * t6 * u3 * 8.8e+1 + t3 * t5 * t6 * u4 * 8.8e+1 + t3 * t6 * t13 * u2 * 1.1e+2 - t3 * t6 * t13 * u4 * 1.1e+2 + 
                   t2 * t5 * t20 * u1 * 5.5e+1 - t2 * t5 * t20 * u3 * 5.5e+1 + pitch_dot_t * t2 * t3 * t6 * yaw_dot_t * 1.012e+3 + pitch_dot_t * t3 * t6 * t14 * yaw_dot_t * 1.012e+3 - 
                   roll_dot_t * t5 * t13 * t20 * yaw_dot_t * 5.06e+2 - pitch_dot_t * roll_dot_t * t2 * t3 * t5 * t6 * 1.012e+3)) / 5.52e+2]

    mt3 = [t22 * (t48 + t55 + t56 + t60 + t15 * u1 * 4.4e+1 - t15 * u2 * 4.4e+1 + t15 * u3 * 4.4e+1 - t15 * u4 * 4.4e+1 - t5 * t20 * u2 * 5.5e+1 + 
                  t5 * t20 * u4 * 5.5e+1 - pitch_dot_t * roll_dot_t * t2 * t20 * 5.06e+2 - t2 * t3 * t6 * u1 * 1.1e+2 + 
                  t2 * t3 * t6 * u3 * 1.1e+2 - t3 * t5 * t6 * t11 * t13 * 1.012e+3 + pitch_dot_t * t2 * t5 * t20 * yaw_dot_t * 5.06e+2 + 
                  roll_dot_t * t3 * t6 * t13 * yaw_dot_t * 1.012e+3) * (-1.0 / 5.52e+2)]

    mt4 = [
        t23 * (t46 + t54 + t61 + t62 + t3 * t6 * u1 * 8.8e+1 - t3 * t6 * u2 * 8.8e+1 + 
                t3 * t6 * u3 * 8.8e+1 - t3 * t6 * u4 * 8.8e+1 - t2 * t20 * u1 * 5.5e+1 + 
                t2 * t20 * u3 * 5.5e+1 - t5 * t11 * t13 * t20 * 5.06e+2 + 
                t3 * t5 * t6 * u2 * 1.1e+2 - t3 * t5 * t6 * u4 * 1.1e+2 + 
                roll_dot_t * t13 * t20 * yaw_dot_t * 5.06e+2 + 
                pitch_dot_t * roll_dot_t * t2 * t3 * t6 * 1.012e+3 - 
                pitch_dot_t * t2 * t3 * t5 * t6 * yaw_dot_t * 1.012e+3
            ) * (-1.0 / 5.52e+2), 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        t4 * t25 * t32, 
        t7 * t25 * t32, 
        t3 * t5 * t25 * (-1.0 / 2.0), 
        et1 + et2
    ] #58

    mt5 = [
        (t22 * (t5 * u1 * 6.0e+1 - t5 * u3 * 6.0e+1 + t40 * u1 + t41 * u4 + t45 * u3 - t11 * t14 * t16 * 5.06e+2 + 
                t2 * t11 * t16 * t18 * 1.012e+3 - t2 * t3 * t6 * u2 * 5.5e+1 + roll_dot_t * t2 * t5 * yaw_dot_t * 1.104e+3 + 
                pitch_dot_t * roll_dot_t * t3 * t5 * t6 * 5.06e+2 + pitch_dot_t * t3 * t6 * t13 * yaw_dot_t * 5.06e+2 - 
                pitch_dot_t * t3 * t6 * t18 * yaw_dot_t * 5.06e+2 - roll_dot_t * t2 * t5 * t16 * yaw_dot_t * 1.012e+3)) / 5.52e+2 - 
        (t5 * t23 * (t46 + t54 + t61 + t62 - t2 * u3 * 6.0e+1 - t19 * u2 * 2.2e+1 + t21 * u1 - t19 * u4 * 2.2e+1 + 
                    t31 * u1 + t31 * u3 + t42 * u2 + roll_dot_t * t13 * yaw_dot_t * 5.52e+2 - t3 * t5 * t6 * u4 * 5.5e+1 + 
                    pitch_dot_t * roll_dot_t * t2 * t3 * t6 * 5.06e+2 - pitch_dot_t * t2 * t3 * t5 * t6 * yaw_dot_t * 5.06e+2)) / 5.52e+2,
        et3 + et4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ] # 66

    mt6 = [
        (t25 * (t4 * t6 - t3 * t5 * t7)) / 2.0,
        (t25 * (t6 * t7 + t3 * t4 * t5)) / 2.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        (t23 * (t57 + pitch_dot_t * t17 * 2.3e+1 + t3 * t5 * t6 * t13 * yaw_dot_t * 5.06e+2)) / 5.52e+2,
        t22 * (t13 * yaw_dot_t * 5.52e+2 - t13 * t16 * yaw_dot_t * 5.06e+2 + pitch_dot_t * t2 * t3 * t6 * 5.06e+2) * (-1.0 / 5.52e+2),
        (t23 * (t53 + pitch_dot_t * t2 * 4.6e+1 + t3 * t6 * t13 * yaw_dot_t * 5.06e+2)) / 5.52e+2,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ] #129 or 130

    mt7 = [
        (t23 * (roll_dot_t * t17 * 2.3e+1 + t2 * yaw_dot_t * 1.058e+3 - t2 * t16 * yaw_dot_t * 5.06e+2 - 
                t14 * t16 * yaw_dot_t * 5.06e+2 + pitch_dot_t * t3 * t6 * t13 * 1.012e+3 + roll_dot_t * t2 * t5 * t16 * 5.06e+2)) / 5.52e+2,
        t22 * (roll_dot_t * t2 * t3 * t6 * 5.06e+2 - t2 * t3 * t5 * t6 * yaw_dot_t * 5.06e+2) * (-1.0 / 5.52e+2),
        (t23 * (roll_dot_t * t2 * 4.6e+1 + t17 * yaw_dot_t * 5.29e+2 + roll_dot_t * t2 * t16 * 5.06e+2 - 
                t2 * t5 * t16 * yaw_dot_t * 5.06e+2)) / 5.52e+2,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        t23 * (t53 - pitch_dot_t * t2 * 1.058e+3 + pitch_dot_t * t14 * t16 * 5.06e+2 + t3 * t6 * t13 * yaw_dot_t * 1.012e+3 - 
                roll_dot_t * t3 * t5 * t6 * t13 * 5.06e+2) * (-1.0 / 5.52e+2)
    ] #142

    mt8 = [
        t22 * (roll_dot_t * t13 * 5.52e+2 - roll_dot_t * t13 * t16 * 5.06e+2 + t5 * t13 * t16 * yaw_dot_t * 1.012e+3 - 
                pitch_dot_t * t2 * t3 * t5 * t6 * 5.06e+2) * (-1.0 / 5.52e+2),
        (t23 * (pitch_dot_t * t17 * 5.29e+2 - pitch_dot_t * t2 * t5 * t16 * 5.06e+2 + roll_dot_t * t3 * t6 * t13 * 5.06e+2 - 
                t3 * t5 * t6 * t13 * yaw_dot_t * 1.012e+3)) / 5.52e+2
    ]

    A = np.reshape(np.array([mt1 + mt2 + mt3 + mt4 + mt5 + mt6 + mt7 + mt8]), (12, 12))

    mt9 = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        t63, t64, t32, 
        t23 * (t26 + t44 + t52) * (-1.0 / 5.52e+2), 
        t22 * (t21 + t31 + t38) * (-1.0 / 5.52e+2), 
        t23 * (t35 + t41 + 9.2e+1) * (-1.0 / 5.52e+2), 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        t63, t64, t32, 
        t23 * (t28 + t36 + t39 + t51 + 1.15e+2) * (-1.0 / 5.52e+2), 
        (t22 * (t31 - t42)) / 5.52e+2, 
        t23 * (t27 + t29 + t45 - 9.2e+1) * (-1.0 / 5.52e+2), 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        t63, t64, t32, 
        (t23 * (t28 + t39 + t52)) / 5.52e+2, 
        (t22 * (t21 - t31 + t38)) / 5.52e+2, 
        (t23 * (t29 + t41 - 9.2e+1)) / 5.52e+2, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        t63, t64, t32, 
        (t23 * (t26 + t36 + t44 + t51 + 1.15e+2)) / 5.52e+2, 
        (t22 * (t31 + t42)) / 5.52e+2
    ]
    mt10 = [(t23 * (t27 + t35 + t45 + 9.2e+1)) / 5.52e+2]
    B = np.reshape(np.concatenate((mt9, mt10)), (12, 4))
    
    return A, B
