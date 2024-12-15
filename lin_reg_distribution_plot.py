import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm

if __name__ == "__main__":
    # Set up the X and Y dimensions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Use the recommended method
    X = np.arange(0, 20, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)

    # Create the univariate normal coefficients and conditional density
    beta0 = -5.0
    beta1 = 0.5
    Z = norm.pdf(Y, beta0 + beta1 * X, 1.0)

    # Plot the surface with the "coolwarm" colormap
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Set the limits of the z axis and major line locators
    ax.set_zlim(0, np.max(Z))  # Dynamically adjust Z-limit
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Label all the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P(Y | X)')

    # Adjust the viewing angle and axes direction
    ax.view_init(elev=30., azim=50.0)
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Plot the probability density
    plt.show()
