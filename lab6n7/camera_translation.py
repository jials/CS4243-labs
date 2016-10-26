import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def q_multi(p, q):
    # quaternion multiplication
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    w = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    x = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
    y = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
    z = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0
    return [w, x, y, z]


def q_conjugate(q):
    w, x, y, z = q
    return [w, -x, -y, -z]


# http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
def normalise(v, tolerance=0.00001):
    mag_square = sum(n * n for n in v)
    if abs(mag_square - 1.0) > tolerance:
        mag = math.sqrt(mag_square)
        v = [n / mag for n in v]
    return v


# q * p * q_conjugate
def qv_multi(q, v):
    p = [0.0] + v
    return q_multi(q_multi(q, p), q_conjugate(q))[1:]  # only take x, y, z


def axisangle_to_q(v, theta):
    v = normalise(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x *= np.sin(theta)
    y *= np.sin(theta)
    z *= np.sin(theta)
    return [w, x, y, z]


def q_to_rot(q):
    w, x, y, z = q
    return np.matrix([
        [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), w * w + y * y - x * x - z * z, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), w * w + z * z - x * x - y * y]
    ])


def perspective_proj(s, t, i, j, k, u=0, v=0, b_u=1, b_v=1, f=1):
    u_out = f * np.dot(s - t, i) / (np.dot(s - t, k) * b_u) + u
    v_out = f * np.dot(s - t, j) / (np.dot(s - t, k) * b_v) + v
    return u_out, v_out


def orthographic_proj(s, t, i, j, u=0, v=0, b_u=1, b_v=1):
    u_out = np.dot(s - t, i) * b_u + u
    v_out = np.dot(s - t, j) * b_v + v
    return u_out, v_out


# defining the rotation matrix for the -30 degree rotation
y_axis_unit = [0, 1, 0]
rot_quat_pos = axisangle_to_q(y_axis_unit, -np.pi / 6)

# define camera translation
cam_pos1 = [0, 0, -5]
cam_pos2 = qv_multi(rot_quat_pos, cam_pos1)
print "pos2: ", cam_pos2
cam_pos3 = qv_multi(rot_quat_pos, cam_pos2)
print "pos3: ", cam_pos3
cam_pos4 = qv_multi(rot_quat_pos, cam_pos3)
print "pos4: ", cam_pos4

positions = np.array([cam_pos1, cam_pos2, cam_pos3, cam_pos4])

# define camera orientation
rot_quat_orient = axisangle_to_q(y_axis_unit, np.pi / 6)
rot_mat_orient = q_to_rot(rot_quat_orient)
quatmat_1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
quatmat_2 = quatmat_1 * rot_mat_orient
print "orientation2: \n", quatmat_2
quatmat_3 = quatmat_2 * rot_mat_orient
print "orientation3: \n", quatmat_3
quatmat_4 = quatmat_3 * rot_mat_orient
print "orientation4: \n", quatmat_4

cam_orientations = np.array([quatmat_1, quatmat_2, quatmat_3, quatmat_4])

pts = np.zeros([11, 3])
pts[0, :] = [-1, -1, -1]
pts[1, :] = [1, -1, -1]
pts[2, :] = [1, 1, -1]
pts[3, :] = [-1, 1, -1]
pts[4, :] = [-1, -1, 1]
pts[5, :] = [1, -1, 1]
pts[6, :] = [1, 1, 1]
pts[7, :] = [-1, 1, 1]
pts[8, :] = [-0.5, -0.5, -1]
pts[9, :] = [0.5, -0.5, -1]
pts[10, :] = [0, 0.5, -1]

# projecting 3D shape points onto camera image planes
# perspective projection
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
axs = [ax1, ax2, ax3, ax4]
perspective_points = []
for cam_pos, cam_orientation, ax in zip(positions, cam_orientations, axs):
    points = [perspective_proj(pt, cam_pos, cam_orientation[0, :], cam_orientation[1, :], cam_orientation[2, :]) for pt in pts]
    perspective_points.append(points)
    for point_x, point_y in points:
        ax.plot(point_x, point_y, 'bo')

# plt.show()
plt.savefig("perspective_projection.png")

# orthogonal projection
_, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
axs = [ax1, ax2, ax3, ax4]
for cam_pos, cam_orientation, ax in zip(positions, cam_orientations, axs):
    points = [orthographic_proj(pt, cam_pos, cam_orientation[0, :], cam_orientation[1, :]) for pt in pts]
    for point_x, point_y in points:
        ax.plot(point_x, point_y, 'bo')

# plt.show()
plt.savefig("orthogonal_projection.png")

M = np.zeros([10, 9])
# only choose pts 0, 1, 2, 3, 8
for i, j in zip(range(0, 10, 2), [0, 1, 2, 3, 8]):
    point = np.append(pts[j, :2], [1])
    # using frame 3
    M[i, :] = np.append(np.append(point, [0, 0, 0]), perspective_points[2][j][0] * -point)
    M[i+1, :] = np.append(np.append([0, 0, 0], point), perspective_points[2][j][1] * -point)

U, S, VT = la.svd(M)
min_index = np.argmin(S)
# if the smallest value is bigger than threshold
if S[min_index] > 0.000001:
    min_index = -1

H = np.array(VT[min_index, :])
H = np.resize(H, (3, 3))
H = H / H[2, 2]  # normalise with h_33
print H

file = open("homograph.txt", "w")
file.write(str(H))
file.close()