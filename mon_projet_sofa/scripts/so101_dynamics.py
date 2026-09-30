import numpy as np
from so101_kinematics import JOINTS, JOINT_ORDER, make_T

N = len(JOINT_ORDER)
CHAIN = [j["child"] for j in JOINTS]

# --- Inertie réelle extraite du URDF (masse, centre de masse, tenseur d'inertie 3x3) ---
INERTIAL = {
    "shoulder_link": dict(mass=0.100006, com=[-0.0307604, -1.66727e-05, -0.0252713],
        I=[[8.3759e-05, 7.55525e-08, -1.16342e-06], [7.55525e-08, 8.10403e-05, 1.54663e-07], [-1.16342e-06, 1.54663e-07, 2.39783e-05]]),
    "upper_arm_link": dict(mass=0.103, com=[-0.0898471, -0.00838224, 0.0184089],
        I=[[4.08002e-05, -1.97819e-05, -4.03016e-08], [-1.97819e-05, 0.000147318, 8.97326e-09], [-4.03016e-08, 8.97326e-09, 0.000142487]]),
    "lower_arm_link": dict(mass=0.104, com=[-0.0980701, 0.00324376, 0.0182831],
        I=[[2.87438e-05, 7.41152e-06, 1.26409e-06], [7.41152e-06, 0.000159844, -4.90188e-08], [1.26409e-06, -4.90188e-08, 0.00014529]]),
    "wrist_link": dict(mass=0.079, com=[-0.000103312, -0.0386143, 0.0281156],
        I=[[3.68263e-05, 1.7893e-08, -5.28128e-08], [1.7893e-08, 2.5391e-05, 3.6412e-06], [-5.28128e-08, 3.6412e-06, 2.1e-05]]),
    "gripper_link": dict(mass=0.087, com=[0.000213627, 0.000245138, -0.025187],
        I=[[2.75087e-05, -3.35241e-07, -5.7352e-06], [-3.35241e-07, 4.33657e-05, -5.17847e-08], [-5.7352e-06, -5.17847e-08, 3.45059e-05]]),
    "moving_jaw_so101_v1_link": dict(mass=0.012, com=[-0.00157495, -0.0300244, 0.0192755],
        I=[[6.61427e-06, -3.19807e-07, -5.90717e-09], [-3.19807e-07, 1.89032e-06, -1.09945e-07], [-5.90717e-09, -1.09945e-07, 5.28738e-06]]),
}
for _link in INERTIAL:
    INERTIAL[_link]["com_arr"] = np.array(INERTIAL[_link]["com"])
    INERTIAL[_link]["I_arr"] = np.array(INERTIAL[_link]["I"])

# --- Butées mécaniques réelles (rad) et couple max, extraits du URDF ---
JOINT_LIMITS = {
    "shoulder_pan":  dict(lower=-1.91986, upper=1.91986, effort=10.0),
    "shoulder_lift": dict(lower=-1.74533, upper=1.74533, effort=10.0),
    "elbow_flex":    dict(lower=-1.69,    upper=1.69,    effort=10.0),
    "wrist_flex":    dict(lower=-1.65806, upper=1.65806, effort=10.0),
    "wrist_roll":    dict(lower=-2.74385, upper=2.84121, effort=10.0),
    "gripper":       dict(lower=-0.174533, upper=1.74533, effort=10.0),
}
LIMIT_K = 40.0   # raideur de la butée mécanique (ressort) en N.m/rad
LIMIT_C = 0.5    # amortissement de la butée


def forward_all(angles):
    """Pivots et axes (monde) de chaque articulation + pose (4x4) de chaque lien de la chaîne."""
    T = np.eye(4)
    pivots = np.zeros((N, 3))
    axes = np.zeros((N, 3))
    links = {}
    for i, j in enumerate(JOINTS):
        T_origin = make_T(j["xyz"], j["rpy"])
        T_pivot = T @ T_origin
        pivots[i] = T_pivot[:3, 3]
        axes[i] = T_pivot[:3, :3] @ np.array([0.0, 0.0, 1.0])
        T_theta = make_T([0, 0, 0], [0, 0, angles[i]])
        T = T_pivot @ T_theta
        links[j["child"]] = T
    return pivots, axes, links


def _dynamics_substep(angles, angvels, torques, dt, g, damping, ground_k, ground_c, ground_z):
    pivots, axes, links = forward_all(angles)
    M = np.zeros((N, N))
    tau_grav = np.zeros(N)
    tau_contact = np.zeros(N)
    tau_limits = np.zeros(N)
    angvels_arr = np.asarray(angvels)

    for k, link in enumerate(CHAIN):
        data = INERTIAL[link]
        m, c, Ibody = data["mass"], data["com_arr"], data["I_arr"]
        T0 = links[link]
        R0 = T0[:3, :3]
        com_pos = T0[:3, 3] + R0 @ c

        kk = k + 1  # joints 0..k affectent le lien k
        Jv = np.zeros((3, N))
        Jw = np.zeros((3, N))
        Jw[:, :kk] = axes[:kk].T
        Jv[:, :kk] = np.cross(axes[:kk], com_pos - pivots[:kk]).T

        M += m * (Jv.T @ Jv) + Jw.T @ (R0 @ Ibody @ R0.T) @ Jw
        tau_grav += Jv.T @ np.array([0, 0, -m * g])

        if com_pos[2] < ground_z:
            pen = ground_z - com_pos[2]
            v_point = Jv @ angvels_arr
            f_z = max(ground_k * pen - ground_c * v_point[2], 0.0)
            tau_contact += Jv.T @ np.array([0, 0, f_z])

    # Butées mécaniques (ressort-amortisseur quand on dépasse la limite)
    for i, name in enumerate(JOINT_ORDER):
        lim = JOINT_LIMITS[name]
        if angles[i] > lim["upper"]:
            tau_limits[i] -= LIMIT_K * (angles[i] - lim["upper"]) + LIMIT_C * angvels_arr[i]
        elif angles[i] < lim["lower"]:
            tau_limits[i] -= LIMIT_K * (angles[i] - lim["lower"]) + LIMIT_C * angvels_arr[i]

    M += 1e-9 * np.eye(N)
    torques = np.asarray(torques)
    tau_total = torques + tau_grav + tau_contact + tau_limits - damping * angvels_arr
    angaccel = np.linalg.solve(M, tau_total)
    angvels_new = angvels_arr + angaccel * dt
    angles_new = np.asarray(angles) + angvels_new * dt
    return angles_new, angvels_new


def dynamics_step(angles, angvels, torques, dt_frame, n_substeps=50,
                   g=9.81, damping=0.05, ground_k=800.0, ground_c=40.0, ground_z=0.0):
    """Avance la simulation de dt_frame (ex: 0.02s = un pas SOFA), en sous-échantillonnant
    pour rester stable (l'intégrateur explicite exige un pas très fin vu les faibles inerties)."""
    dt_sub = dt_frame / n_substeps
    torques = [max(-JOINT_LIMITS[n]["effort"], min(JOINT_LIMITS[n]["effort"], t))
               for n, t in zip(JOINT_ORDER, torques)]
    angles = np.asarray(angles, dtype=float)
    angvels = np.asarray(angvels, dtype=float)
    for _ in range(n_substeps):
        angles, angvels = _dynamics_substep(angles, angvels, torques, dt_sub,
                                             g, damping, ground_k, ground_c, ground_z)
    return angles.tolist(), angvels.tolist()