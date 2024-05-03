import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

ijkl_pressed = [False, False, False, False];

def on_key_press(event):
    print(f'Key pressed: {event.key}')
    if (event.key in ('y', 'g', 'h', 'j')):
        ijkl_pressed[('y', 'g', 'h', 'j').index(event.key)] = True;

def on_key_released(event):
    print(f'Key released: {event.key}')
    if (event.key in ('y', 'g', 'h', 'j')):
        ijkl_pressed[('y', 'g', 'h', 'j').index(event.key)] = False;

proj_point = np.array([0, .5, 1]);

# vertices of the shape
# MAKE SURE THEY'RE ALL ON THE SAME PLANE!!
vertices = np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0.5, 1, 0],
                     [.4, 1.1, 0]], dtype=np.float64)

faces = np.array([vertices])

# project point into the vector space defined by the orthogonal basis
def project_to_orthogonal(orthogonal_basis, point):
    if (len(orthogonal_basis) == 0):
        return np.zeros(point.shape);

    result = np.zeros(point.shape);
    for i in range(len(orthogonal_basis)):
        b = orthogonal_basis[i]
        result += np.dot(point, b) / (np.linalg.norm(b) ** 2) * b

    return result;

# proejct a point into the vector space defined by the basis
def project(basis, point):
    # orhogonalize he basis
    for i in range(len(basis)):
        newbas = basis[i] - project_to_orthogonal(basis[:i], basis[i]);
        basis[i] = newbas;

    return project_to_orthogonal(basis, point);

# project a point into the affine space defined by the affine set of points.
def project_affine(affine_space, point, origin_index=0):
    return project(np.array([affine_space[i] - affine_space[origin_index] for i in range(len(affine_space)) if i != origin_index]), point - affine_space[origin_index]);

r = project_affine(vertices.copy(), proj_point);
# print(r);
# print(vertices);

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_released)

wall_faces = [];
for i in range(len(faces[0])):
    a, b = faces[0][i], faces[0][(i + 1) % len(faces[0])];
    wall_faces.append([a, b, b, a]);

print(wall_faces);

wall_faces = np.array(wall_faces, dtype=np.float64);

print(wall_faces.shape)

triangle = ax.add_collection3d(Poly3DCollection(faces, linewidths=1, edgecolors='k', alpha=.25))
extension_part = ax.add_collection3d(Poly3DCollection(faces, linewidths=1, edgecolors='k', alpha=.25))
extension_wall = ax.add_collection3d(Poly3DCollection(wall_faces, linewidths=1, edgecolors='k', alpha=.25));

proj_disp = ax.scatter(*proj_point, c='b', marker='o', label='Points');
# res_proj_disp = ax.scatter(*r, c='b', marker='o', label='Points');

a_k = 1;

def animate(frame):
    global a_k;
    for i in range(len(faces[0])):
        faces[0][i] = proj_point + a_k * (faces[0][i] - proj_point);

    for i in range(len(wall_faces)):
        wall_faces[i][2] = proj_point + a_k * (wall_faces[i][2] - proj_point);
        wall_faces[i][3] = proj_point + a_k * (wall_faces[i][3] - proj_point);

    extension_part.set_verts(faces);
    a_k = max(0.01, a_k - .001);

    print(wall_faces);
    extension_wall.set_verts(wall_faces);

ani = FuncAnimation(fig, animate, frames=np.arange(0, 360, 10), interval=50)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

#plt.axis('off')
plt.show()
