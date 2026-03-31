import os
from pprint import pprint
import czifile
from natsort import natsorted
import numpy as np
from math import floor
import SimpleITK  as sitk
import networkx as nx
import pandas as pd
from skimage.draw import line_nd
from scipy.ndimage import gaussian_filter
from osteoid import Skeleton
#from utils.usefull_functions import flatten_list_recursive
from skimage.morphology import label
import napari
import kimimaro

#For tests
class FakeSkeleton:
    def __init__(self, vertices, edges, radii):
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        self.radii = np.array(radii)

def create_microvessel_stack():
    """
    ```
    top view                (diago)
                          8 ●
                           /     
                        7 ●
                       4 /   
        ● — ● — ● — ● — ● — ● — ● (main)
        0   1   2   3   |   5   6
                  \     ● 9
                    ●   |
                  12  \ ● 10
                        |
                        ● 11
                        (bottom)
    ```
    """

    vertices = [
        (30,50, 5), (30,50, 30), (40,50,40), (60,50,50), (60,50,60), (40,50,80), (40,50,90), #main
        (60,60,60), (60,70,70), # diago
        (60,40,50), (50,30,50), (40,20,50), #bottom
        (40,40,20) #loop
    ]

    edges = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), #main
        (4,7), (7,8),
        (4,9), (9,10), (10,11),
        (2,12), (11,12)
    ]
    
    radii = [2.9, 3, 3.1, 4, 5, 4, 3.5, 
             3, 2.5,
             3.5, 3.1, 3, 
             2
    ]
    
    vertices = np.array(vertices, dtype='uint8')
    edges = np.array(edges, dtype='uint8')
    radii = np.array(radii, dtype='uint8')

    # --- Create 3D grid ---
    margin = int(np.max(radii)) + 2

    grid = np.zeros((100,100,100), dtype=np.uint8)

    # --- Helper: draw sphere ---
    def draw_circle(grid, center, radius):
        x0, y0, z0 = center.astype(int)
        r = int(np.ceil(radius))
        z = set()
        for x in range(x0 - r, x0 + r + 1):
            for y in range(y0 - r, y0 + r + 1):
                for z in range(z0 - r, z0 + r + 1):
                    if (
                        0 <= x < grid.shape[0] and
                        0 <= y < grid.shape[1] and
                        0 <= z < grid.shape[2]
                    ):
                        if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                            val = 100 - z/100
                            # print(val)
                            grid[x, y, z] = 100 - z/10

    # --- Draw tubes ---
    for i, j in edges:
        p0 = vertices[i]
        p1 = vertices[j]
        r0 = radii[i]
        r1 = radii[j]

        length = np.linalg.norm(p1 - p0)
        steps = int(length * 2) + 1  # sampling density

        for t in np.linspace(0, 1, steps):
            p = (1 - t) * p0 + t * p1
            r = (1 - t) * r0 + t * r1
            draw_circle(grid, p, r)

        #Apply blur
        grid = gaussian_filter(grid, 0.5)
    
    print(type(grid), grid.shape, grid.min(), grid.max())

    return grid

    def world_to_voxel(p):
        return np.array([
            p[0] / vz,
            p[1] / vy,
            p[2] / vx
        ])

    def draw_sphere(center_vox, radius_phys, intensity):

        rz = radius_phys / vz
        ry = radius_phys / vy
        rx = radius_phys / vx

        cz, cy, cx = center_vox

        zmin = max(0, int(cz - rz - 1))
        zmax = min(Z, int(cz + rz + 2))
        ymin = max(0, int(cy - ry - 1))
        ymax = min(Y, int(cy + ry + 2))
        xmin = max(0, int(cx - rx - 1))
        xmax = min(X, int(cx + rx + 2))

        for z in range(zmin, zmax):
            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    dz = (z - cz) * vz
                    dy = (y - cy) * vy
                    dx = (x - cx) * vx

                    if dz*dz + dy*dy + dx*dx <= radius_phys**2:
                        vol[z, y, x] = max(vol[z, y, x], intensity)

    for e_idx, (i, j) in enumerate(edges):
        p0 = np.array(vertices[i])
        p1 = np.array(vertices[j])

        r = radii[e_idx]

        # intensity linked to radius
        if radius_intensity:
            intensity = int(60 + (r / max(radii)) * 180)
        else:
            intensity = base_intensity

        # convert to voxel space
        v0 = world_to_voxel(p0)
        v1 = world_to_voxel(p1)

        length = np.linalg.norm((v1 - v0) * np.array([vz, vy, vx]))

        # sampling density (important!)
        step = max(r / 2, 0.5)
        n_samples = max(2, int(length / step))

        for t in np.linspace(0, 1, n_samples):
            p = v0 * (1 - t) + v1 * t
            draw_sphere(p, r, intensity)

    return vol

def create_test_skeleton_no_loop():
    """
    ```
            (branch) 8
            |
            ● 7
            |
    ● — ● — ● — ● — ● — ● — ●
    0   1   |2  3   4   5   6
          9 ●
            |
         10 ●
            |
         11 (branch)

    ```
    """
    vertices = [
        (-1, 0, 0),   # 0
        (0, 0, 1),   # 1
        (0, 0, 2),   # 2 ← branch point
        (0, 0, 3),   # 3
        (0, 0, 4),   # 4
        (0, 0, 5),   # 5
        (0, 0, 6),   # 6

        (0, 1, 2),   # 7 (branch up)
        (0, 2, 2),   # 8

        (0, -1, 2),  # 9 (branch down)
        (0, -2, 2),  # 10
        (0, -3, 2)   # 11
    ]

    edges = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6),
        (2,7), (7,8),
        (2,9), (9,10), (10,11)
    ]

    # Radii per node
    radii = [
        1.0,  # 0
        1.1,  # 1
        1.2,  # 2 (slightly larger at junction)
        1.1,  # 3
        1.0,  # 4
        1.1,  # 5
        1.0,  # 6

        0.8,  # 7
        0.7,  # 8

        0.8,  # 9
        0.7,  # 10
        0.7  # 11
    ]

    return FakeSkeleton(vertices, edges, radii)

def create_test_skeleton_loop():
    """
    ```
         8  x
            |   13
         7  x — ● — ● 12 loop
            |       |
    x — ● — x — ● — x — ● — x
    0   1   |2  3   4   5   6
          9 ●
            |
         10 ●
            |
         11 x

    ```
    """
    vertices = [
        (-1, 0, 0),   # 0
        (0, 0, 1),   # 1
        (0, 0, 2),   # 2 ← branch point
        (0, 0, 3),   # 3
        (0, 0, 4),   # 4
        (0, 0, 5),   # 5
        (0, 0, 6),   # 6

        (0, 1, 2),   # 7 (branch up)
        (0, 2, 2),   # 8

        (0, -1, 2),  # 9 (branch down)
        (0, -2, 2),  # 10
        (0, -3, 2),  # 11

        (0, 1, 4),   # 12 (loop)
        (0, 1, 3)    # 13   



    ]

    edges = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6),
        (2,7), (7,8),
        (2,9), (9,10), (10,11),
        (4,12), (12,13), (7,13)
    ]

    # Radii per node
    radii = [
        1.0,  # 0
        1.1,  # 1
        1.2,  # 2 (slightly larger at junction)
        1.1,  # 3
        1.0,  # 4
        1.1,  # 5
        1.0,  # 6

        0.8,  # 7
        0.7,  # 8

        0.8,  # 9
        0.7,  # 10
        0.7,  # 11

        0.7,  # 12
        0.7,  # 13
    ]

    return FakeSkeleton(vertices, edges, radii)

#For real data
def load_stack_metadata(day_path: str, filename: str, czi_mode: str):

    """Returns the full stack as ndarray (C,Z,Y,X), the associated metadata adn the scaling (Z,Y,X)"""

    full_stack = None
    scaling = None
    metadata = None

    if czi_mode == 'single':
        #Store filenames
        files = os.listdir(day_path)
        filenames = [f for f in files if filename in f]
        filenames = natsorted(filenames)
        if "(" not in filenames[-1]:
            filenames.insert(0, filenames[-1])
            del filenames[-1]
        
        #Load metadata and save dimension
        czi = czifile.CziFile(os.path.join(day_path, filenames[0]))
        metadata = czi.metadata(asdict=True)
        dim_loc = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]
        if "SizeC" in dim_loc.keys():
            dim_C = dim_loc["SizeC"]
        else:
            dim_C = 1
        dims = (dim_C, dim_loc["SizeZ"], dim_loc["SizeY"], dim_loc["SizeX"])
        full_stack = np.ndarray(dims, dtype='uint16')
        del dims
        czi.close()

        #Open files and store them in stack
        c=0
        for i,f in enumerate(filenames):
            if (i % dim_C == 0):
                c=0
            else:
                c+=1
            if i != 0 and ((i+1) % dim_C == 0):
                z_i = floor((i)/dim_C)
            else:
                z_i = floor((i+1)/dim_C)
            czi = czifile.CziFile(os.path.join(day_path, f))
            full_stack[c,z_i,:,:] = czi.asarray()
            czi.close()

    elif czi_mode == 'stack':
        czi = czifile.CziFile(os.path.join(day_path, filename)) # CZI file + metadata
        metadata = czi.metadata(asdict=True)
        dim_loc = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]
        stack = czi.asarray() #Full stack of shape (Channel, Z, Y, X)
        if "SizeC" in dim_loc.keys():
            dim_C = dim_loc["SizeC"]
            full_stack = stack
        else:
            dim_C = 1
            dims = (dim_C, dim_loc["SizeZ"], dim_loc["SizeY"], dim_loc["SizeX"])
            full_stack = np.ndarray(dims, dtype='uint16')
            full_stack[0,:,:,:] = stack


    # Store voxel value
    pxls2um = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    scaling = (pxls2um[2]["Value"]*1000000, pxls2um[0]["Value"]*1000000, pxls2um[0]["Value"]*1000000)
    z_step_um = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Z"]["Positions"]["Interval"]["Increment"]

    return full_stack, scaling, metadata

def get_isotropic_stack(stack, scaling: tuple):
    """Return resliced stack as np.ndarray of dimension (Z,Y,X) with isotropic voxel."""
    img = sitk.GetImageFromArray(stack)
    img.SetSpacing((scaling[1], scaling[1], scaling[0]))

    original_size = np.array(img.GetSize())
    original_spacing = np.array(img.GetSpacing())

    new_spacing = np.array([scaling[1], scaling[1], scaling[1]])
    new_size = (original_size * (original_spacing / new_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_img = resampler.Execute(img)

    stack_iso = sitk.GetArrayFromImage(resampler.Execute(img))
    del img, resampler, new_spacing

    return stack_iso

def skeletons_to_volume(skeletons: dict[Skeleton], shape: tuple, anisotropy: tuple):
    """Transform Kkimimaro.skeleton dictionnary into numpy stack (np.ndarray(z,y,x))"""

    vol = np.zeros(shape, dtype=np.uint8)

    for skel in skeletons.values():
        verts = skel.vertices / anisotropy
        verts = np.round(verts).astype(int)

        for v1, v2 in skel.edges:
            p1 = verts[v1]
            p2 = verts[v2]

            rr = line_nd(p1, p2)
            vol[rr] = 1

    return vol

def norm_3d(v1, v2):
    """Return length between two nodes"""
    s=0
    for i in range(len(v1)):
        s += np.pow((v2[i] - v1[i]), 2)
    return np.sqrt(s)

def skeleton_to_graph(skel: Skeleton):
    """Creates Networkx Graph object from skeleton object (osteid library) and compute metrics"""
    
    G = nx.Graph()

    # Add nodes with 3D coordinates
    for i, v in enumerate(skel.vertices):
        G.add_node(i, coord=tuple(v), radius=skel.radii[i])

    # Add edges
    for e in skel.edges:
        n1, n2 = int(e[0]), int(e[1])

        #Compute branch metrics
        p1 = skel.vertices[n1]
        p2 = skel.vertices[n2]
        length = norm_3d(p1, p2)
        direction = (p2 - p1) / norm_3d(p1, p2)
        # radius = skel.radii[n1]

        G.add_edge(n1, n2, length=length, direction=direction)# , radius=radius)

    return G

def get_brenches_data(G: nx.Graph):
    """Return metrics about brenches composing the strcuture encoded in G."""
    
    special_node = [node for node in G.nodes if G.degree[node] != 2]
    brenches = []
    brench_label = 0
    visited_brench = set()
    for sp_node in special_node:
        
        for neighbor in G.neighbors(sp_node):

            edge = tuple(sorted((sp_node, neighbor)))
            if edge in visited_brench:
                continue
            
            length_brench = G.edges[sp_node, neighbor]['length']
            diameters = [G.nodes[sp_node]['radius']]
            visited_brench.add(edge)

            current = neighbor
            past = sp_node
            brench_nodes = [past]
            brench_edges = [edge]

            while G.degree[current] == 2:
                
                next_node = [n for n in G.neighbors(current) if n != past][0]
                edge = tuple(sorted((current, next_node)))
                visited_brench.add(edge)

                length_brench += G.edges[current, next_node]['length']
                diameters.append((G.nodes[current]["radius"]))

                brench_nodes.append(current)
                brench_edges.append(edge)

                past = current
                current = next_node
            
            diameters.append((G.nodes[current]["radius"]))
            brench_nodes.append(current)
                
            if length_brench <= 10:
                brench_type = "sprout"
            else:
                brench_type = "brench"

            brenches.append({
            "Label": brench_label,
            "Endpoints": {sp_node: G.nodes[sp_node]['coord'], current: G.nodes[current]['coord']}, 
            "Length": length_brench, 
            "Mean diameter": np.mean(diameters), 
            "Std diameter": np.std(diameters),
            "Tortuosity": length_brench/norm_3d(G.nodes[current]['coord'],G.nodes[sp_node]['coord']),
            "Type": brench_type,
            "Nodes": [brench_nodes],
            "Edges": [brench_edges]})

            brench_label += 1
    
    return pd.DataFrame(brenches)

def brenches_post_process(G: nx.Graph, brenches: pd.DataFrame):

    pass

def get_structure_data(G: nx.Graph, brenches: pd.DataFrame, scaling: tuple):
    """Return metrics about the structure encoded in G."""

    #Remove sprouts from brenches and G
    rm_nodes = brenches[brenches["Type"] == "sprout"]["Nodes"].to_list()
    G_updated = G.copy()
    rm_edges = [tup for tup in G.edges if any(n in tup for n in rm_nodes)]
    # rm_nodes = [elt[0][1:-1] for elt in rm_nodes]
    # rm_nodes = [elt for skel in rm_nodes for elt in skel if G.degree[elt] <= 2]
    # G_updated.remove_nodes_from(rm_nodes)
    G_updated.remove_edges_from(rm_edges)
    brenches_no_sprouts = brenches[brenches["Type"] != "sprout"]

    vertices = [G.nodes[n]['coord'] for n in G.nodes]
    bbox_coord = bbox_coordinate(vertices)
    bbox_vol = bbox_volume(bbox_coord, scaling)
    n_brench = brenches.shape[0]
    n_junction = len([d for _,d in G.degree if d> 2])
    n_endpoints = len([d for _,d in G.degree if d == 1])
    n_cycle = len(nx.cycle_basis(G))

    row_global = {
        "Bbox coordinate": [bbox_coord],
        "Total length": brenches_no_sprouts["Length"].sum(),
        "Mean Length": brenches_no_sprouts["Length"].mean(),
        "Std Length": brenches_no_sprouts["Length"].std(ddof=0),
        "Longest Path": None,
        "Longest Path length": None,
        "Lowest node": lowest_node(vertices),
        "Mean diameter": brenches_no_sprouts["Mean diameter"].mean(),
        "Std diameter": brenches_no_sprouts["Mean diameter"].std(ddof=0),
        "Mean Tortuosity": brenches_no_sprouts["Tortuosity"].mean(),
        "Std Tortuosity": brenches_no_sprouts["Tortuosity"].std(ddof=0),
        "Bbox volume": bbox_vol,
        "Brench number": n_brench,        
        "Junction number": n_junction,        
        "Loop number": n_cycle,      
        "Endpoint number": n_endpoints        
    }
    if bbox_vol is None:
        density = {
            "Brench density": None,
            "Junction density": None,
            "Loop density": None,
            "Endpoint density": None
        }
    else:
        density = {
            "Brench density": n_brench/bbox_vol,
            "Junction density": n_junction/bbox_vol,
            "Loop density": n_cycle/bbox_vol,
            "Endpoint density": n_endpoints/bbox_vol
        }

    row_global = row_global |density

    return pd.DataFrame(row_global, index=[0]), G_updated

def longest_path_tree(G: nx.Graph):
    # Step 1: pick arbitrary node
    start = list(G.nodes)[0]

    # Step 2: find farthest node from it
    lengths = nx.single_source_dijkstra_path_length(G, start, weight='length')
    u = max(lengths, key=lengths.get)

    # Step 3: from u, find farthest node
    lengths, paths = nx.single_source_dijkstra(G, u, weight='length')
    v = max(lengths, key=lengths.get)

    return paths[v], lengths[v]

def bbox_coordinate(vertices):
    """Return bbox coordinate of structure as (z_min, y_min, x_min), (z_max, y_max, x_max)"""
    nodes_coords = pd.DataFrame(vertices, columns=["z", "y", "x"])
    return (tuple(nodes_coords.min()), tuple(nodes_coords.max()))

def lowest_node(vertices):
    """Return the nodes located the lowest on z-axis."""
    nodes_coords = pd.DataFrame(vertices, columns=["z", "y", "x"])
    low_z = nodes_coords.z.min()
    idx_low_z = nodes_coords.index[nodes_coords.z == low_z]
    lowest_node = [tuple(float(e) for e in elt) for elt in nodes_coords.iloc[idx_low_z].values]

    return [tuple(lowest_node)]

def bbox_volume(bbox_coordinate, scaling):
    """Return the volume of the bbox"""

    volume = []
    for i in range(len(bbox_coordinate[0])):
        volume.append(abs(bbox_coordinate[0][i] - bbox_coordinate[1][i]))
    if 0 in volume:
        volume.remove(0)
        if 0 in volume: 
            volume.remove(0)
            return scaling*scaling*volume[0]
        return scaling*volume[0]*volume[1]
    return volume[0]*volume[1]*volume[2]

def get_network_data(endo_vessel_bin):
    """Return the metrics of the network."""

    network_data = None

    return network_data

name = "C:\\Users\\ChimieENS\\Documents\\Layla\\Data\\251117_Exp004-E\\D14-confocal-analysis\\Exp004-E_D-+CHIR-_bin.npy"
scaling = 1.3837763682730653
bin = np.load(name)
teasar_param = kimimaro.intake.DEFAULT_TEASAR_PARAMS.copy()
for s in [3.5]:
    for c in [0,0.5, 1]:
        if s == 3 and c != 1:
            continue
        for e in [8]:
            print(s, c, e)
            teasar_param.update({
                "scale": s, #defines skeleton detail (high value for less detail). Default is 1.5
                "const": c, # control path pruning during TEASAR algo. Default is 300
                "pdrf_exponent": e, # how much branch are penalized if close to edge of object (low values = low penalty). Default is 4.
                })

            kimi = kimimaro.skeletonize(label(bin), dust_threshold=10, anisotropy=(scaling, scaling, scaling), teasar_params=teasar_param)
            skeleton = skeletons_to_volume(kimi, bin.shape, (scaling, scaling, scaling))
            tmp = name.replace("bin", f"skeleton_s{s}_c{c}_e{e}")
            np.save(tmp, skeleton)
