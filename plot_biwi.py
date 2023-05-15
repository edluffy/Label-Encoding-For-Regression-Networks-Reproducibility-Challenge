import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_head_pose(path, yaw, pitch, roll):
  pitch -= 65.0
  roll -= 55.0
  yaw -= 74.0
  x = -np.radians(pitch)
  y = np.radians(roll)
  z = np.radians(yaw)

  sx, cx = np.sin(x), np.cos(x)
  sy, cy = np.sin(y), np.cos(y)
  sz, cz = np.sin(z), np.cos(z)

  Rx = np.array([[1, 0, 0],
                 [0, cx, -sx],
                 [0, sx, cx]])
  Ry = np.array([[cy, 0, sy],
                 [0, 1, 0],
                 [-sy, 0, cy]])
  Rz = np.array([[cz, -sz, 0],
                 [sz, cz, 0],
                 [0, 0, 1]])
  R = np.dot(Rz, np.dot(Ry, Rx))
  
  x_axis = np.array([1, 0, 0])
  y_axis = np.array([0, 1, 0])
  z_axis = np.array([0, 0, 1])
  
  x_axis = np.dot(R, x_axis)
  y_axis = np.dot(R, y_axis)
  z_axis = np.dot(R, z_axis)
  
  fig = plt.figure()
  ax = plt.axes([0, 0, 1, 1], frameon=False)
  ax.set_axis_off()

  im = plt.imread(path)
  implot = plt.imshow(im)

  ax = plt.axes([0, 0, 1, 1], projection='3d')
  ax.set_facecolor('None')
  ax.grid(False)
  plt.axis('off')
  
  ax.plot([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]], color='r', linewidth=8)
  ax.plot([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]], color='b', linewidth=8)
  ax.plot([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]], color='lime', linewidth=8)
  
  ax.set_xlim3d(-0.75, 0.75)
  ax.set_ylim3d(-0.75, 0.75)
  ax.set_zlim3d(-0.75, 0.75)
  ax.invert_zaxis()
  ax.azim = -90
  ax.elev = 0
  
  plt.show()
  print('Yaw:', yaw, 'Pitch:', pitch, 'Roll:', roll)

plot_head_pose('frame_00137_rgb.png', 57, 101, 51)
plot_head_pose('frame_00626_rgb.png', 104, 93, 64)
plot_head_pose('frame_00058_rgb.png', 71, 85, 56)
plot_head_pose('frame_00253_rgb.png', 73, 97, 59)
plot_head_pose('frame_00732_rgb.png', 81, 59, 55)
