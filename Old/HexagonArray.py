import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line, disk
import math

def hexagon_outline_ndarray(image_shape, center, s, thickness=1):
    """
    Draws a hexagon outline (not filled) into a 2D ndarray.
    
    Parameters:
        image_shape (list): [height, width] of the output ndarray.
        center (list): [y, x] center of the hexagon.
        s (float): side length of the hexagon.
        thickness (int): thickness in pixel of the hegaxon. If not entered set at 1.

    Returns:
        np.ndarray: ndarray with hexagon outline written as 1s.
    """
    #height, width = image_shape
    height = image_shape[0]
    width = image_shape[1]
    img = np.zeros((height, width), dtype=np.uint8)

    #cx, cy = center
    cx = center[1]/2
    cy = center[0]/2
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
    x_vertices = cx + s * np.cos(angles)
    y_vertices = cy + s * np.sin(angles)

    # Draw lines between consecutive vertices
    for i in range(6):
        x0, y0 = int(round(x_vertices[i])), int(round(y_vertices[i]))
        x1, y1 = int(round(x_vertices[(i + 1) % 6])), int(round(y_vertices[(i + 1) % 6]))
        rr, cc = line(y0, x0, y1, x1)  # Notice (row, col) = (y, x)
        # For each pixel in the line, draw a small disk of radius thickness//2
        for r, c in zip(rr, cc):
            dr, dc = disk((r, c), radius=thickness // 2, shape=img.shape)
            img[dr, dc] = 1

    return img