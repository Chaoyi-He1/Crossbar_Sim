import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_array(array, vmin = 0, vmax = 255):
    plt.figure(figsize = (4,20))
    ax = plt.gca()
    im = ax.imshow(array, vmin = vmin, vmax = vmax, cmap = 'viridis')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.show()