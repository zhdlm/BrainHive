import numpy as np
from skimage.draw import disk, line

class HexagonMath:

    def __init__(self, height):

        self.height = height
        self.side = round(self.height/(2*np.sin(np.pi/3)))
        self.diagonal = 2*self.side

    @classmethod
    def get_side_from_height(cls, height):
        """Compute hexagon side from height."""
        return round(height/(2*np.sin(np.pi/3)))
    
    @classmethod
    def get_diagonal_from_height(cls, height):
        """Compute hexagon diagonal from height."""
        return round(2*height/(2*np.sin(np.pi/3)))

    @classmethod
    def hexagon_outline_ndarray(cls, image_shape, center, side, thickness=1, return_type='hexagon'):
        """Draw a hexagon outline (not filled) into a 2D ndarray.
        
        Parameters
        ----------
        image_shape: list | tuple
            [height, width] of the output ndarray.
        center: list | tuple
            [y, x] center of the hexagon.
        side: float
            side length of the hexagon.
        return type: str. {'hexagon', 'side'}
            Type of outline requested:
            - 'hexagon': a np.ndarray with the hexagon outline set at 1.
            - 'side': a list of 6 elements composed np.ndarray with the 
            hexagon side outline set at 1. Each element correspond to one
            side of the hexagon.

        Returns
        -------
        out: np.ndarray | list of np.ndarray
            -np.ndarray: Image of hexagon outline written as 1s and rest set at 0
            (if 'hexagon' requested)
            - list of np.ndarray: Each list element is the outline of one side of
            the hexagon if 'side' requested.
        """

        height = image_shape[0]
        width = image_shape[1]
        if return_type == 'hexagon':
            img = np.zeros((height, width), dtype=np.int8)
        elif return_type == 'side':
            img = [np.zeros((height, width), dtype=np.int8) for _ in range(6)]
        elif return_type == 'coord-side':
            img = []

        cx = center[1]
        cy = center[0]
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
        x_vertices = cx + side * np.cos(angles)
        y_vertices = cy + side * np.sin(angles)

        # Draw lines between consecutive vertices
        for i in range(6):
            x0, y0 = int(round(x_vertices[i])), int(round(y_vertices[i]))
            x1, y1 = int(round(x_vertices[(i + 1) % 6])), int(round(y_vertices[(i + 1) % 6]))
            rr, cc = line(y0, x0, y1, x1)  # Notice (row, col) = (y, x)
            # For each pixel in the line, draw a small disk of radius thickness//2
            if return_type == 'coord-side':
                img.append([x0,y0])
                img.append([x1, y1])
            else:
                for r, c in zip(rr, cc):
                    if return_type == 'hexagon':
                        dr, dc = disk((r, c), radius=thickness // 2, shape=img.shape)
                        img[dr, dc] = 1
                    elif return_type == 'side':
                        dr, dc = disk((r, c), radius=thickness // 2, shape=img[0].shape)
                        img[i][dr, dc] = 1

        return img
    