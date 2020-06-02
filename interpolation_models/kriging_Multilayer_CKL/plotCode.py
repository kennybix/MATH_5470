import numpy as np
import Benchmark_Problems as BP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

XlowerLimit = -6
XUpperLimit = 6
n = 100

x1 = np.linspace(XlowerLimit,XUpperLimit,n)
x2 = np.linspace(XlowerLimit,XUpperLimit,n)
X,Y = np.meshgrid(x1,x2)
Z = BP.himmelblau(X,Y)
YlowerLimit = min(Z.reshape((n*n),1))
YupperLimit = max(Z.reshape((n*n),1))

# fig = plt.figure(1,figsize=(9,6))
# ax = fig.gca(projection='3d')
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z,cmap=RdGy,linewidth=0, antialiased=False)
# '''
# X and Y values are normalized
# Z must not be normalized to give the needed trend
# '''
# #labelling
# plt.xlabel('X1')
# plt.ylabel('X2')
# # Customize the z axis.
# ax.set_zlim(YlowerLimit, YupperLimit)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()


plt.figure(1,figsize=(9,6))
plt.contourf(X,Y,Z,20,cmap="RdGy")
plt.title('Himmelblau Function',fontsize=20)
plt.xlabel('X1',fontsize=20)
plt.ylabel('X2',fontsize=20)
plt.colorbar()
plt.show()