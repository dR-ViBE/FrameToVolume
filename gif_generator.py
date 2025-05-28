
# This code is for generating GIF using the point cloud generated in the depth_estimator.py
import numpy as np
import open3d as o3d
import imageio
import os

def gif(op):
    pcd = o3d.io.read_point_cloud(op)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    unwanted = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(unwanted)

    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    center = mesh.get_center()
    mesh.translate(-center)

    v = o3d.visualization.Visualizer()
    v.create_window(visible=False)
    v.add_geometry(mesh)

    opt = v.get_render_option()
    opt.mesh_show_back_face = True

    # vc.set_lookat((0, 0, 0))
    # vc.set_front([0, 0, -1])
    # vc.set_up([0, 1, 1])
    # vc.set_zoom(0.7)

    vc = v.get_view_control()
    vc.set_lookat((0, 0, 0))
    vc.set_front([0, 0, -1])
    vc.set_up([0, 1, 0])
    vc.set_zoom(0.7)

    n = 60
    f = []
    for i in range(n):
        r = mesh.get_rotation_matrix_from_xyz((0, np.radians(360/n), 0))
        mesh.rotate(r, center=(0, 0, 0))
        v.update_geometry(mesh)
        v.poll_events()
        v.update_renderer()
        image = v.capture_screen_float_buffer(False)
        image = (np.asarray(image)*255).astype(np.uint8)
        f.append(image)

    v.destroy_window()
    return f

ip = input("Please type the output folder name for which you want 3D view: ").strip() 
# Type just the output folder name in which your desired point cloud file is present
op = "output/" + ip

p = os.path.join(op, "point_cloud_midas.ply")
f = gif(p)
o = os.path.join(op, "gif_midas.gif")
imageio.mimsave(o, f, duration=0.05)

p = os.path.join(op, "point_cloud_depth_anything.ply")
f = gif(p)
o = os.path.join(op, "gif_depth_anything.gif")
imageio.mimsave(o, f, duration=0.05)

p = os.path.join(op, "point_cloud_zoe_depth.ply")
f = gif(p)
o = os.path.join(op, "gif_zoe.gif")
imageio.mimsave(o, f, duration=0.05)
#Find all the GIFs in the output folder

