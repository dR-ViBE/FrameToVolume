
# This code is for analysing the 3 models and generating point clouds, mesh and prallax gif
import torch
import matplotlib.pyplot as plt

from PIL import Image
from transformers import pipeline
from tqdm import tqdm

import imageio
import open3d as o3d 
import os
import cv2
import numpy as np


def loading(p):
    user_image = cv2.imread(p)
    rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
    return rgb

def displaing(user_image, title):

    if user_image.ndim==2:
        value = 'gray'  
    else:
        value = None

    plt.figure(figsize=(3,3))
    plt.imshow(user_image, cmap=value)
    plt.axis("off")
    plt.title(title)
    plt.show()

def lpVarience(size):
    if size.ndim == 3:
        g = cv2.cvtColor((size * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)  
    else:
        g = (size * 255).astype(np.uint8)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def edgeSharp(size):
    # g = (size * 255).astype(np.uint8) 
    if isinstance(size, float):
        g = np.array([[size * 255]], dtype=np.uint8)
    else:
        g = (size * 255).astype(np.uint8)    
    x = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)
    y = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5)
    m = np.sqrt(x**2 + y**2)
    return np.mean(m)

def totalScore(lp, depth, sharp):
    return lp + depth + sharp

def parallaxingEffect(user_image, depth, op):
    n=150
    shift=50
    f=30 
    h, w = depth.shape
    x,y = np.meshgrid(np.arange(w), np.arange(h))
    o = []

    for t in tqdm(range(n)):

        tt = 2*np.pi*t/n
        a = np.sin(np.pi * t/n)**2
          
        x1 = shift * np.cos(tt) * a
        x2 = (x + depth * x1).astype(np.float32)

        y1 = shift * np.sin(tt) * 0.5 * a
        y2 = (y + depth * y1).astype(np.float32)
        
        w = cv2.remap(user_image, x2, y2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        o.append(w)

    imageio.mimsave(op, o, fps=f, loop=15, palettesize=256)

#point cloud generation for each image(all 6 different images)
def pc(user_image, image_dimension):
    h,w = image_dimension.shape
    d=2

    y = np.arange(0, h, d)
    x = np.arange(0, w, d)
    x, y = np.meshgrid(x, y)
    z = image_dimension[::d, ::d] * 100

    p = np.stack([x.flatten()-w/2, (h/2)-y.flatten(), z.flatten()], axis=-1)
    columns = user_image[::d, ::d].reshape(-1,3)/255.0
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(p)
    # pc.colors = o3d.utility.Vector3dVector(columns)
    # pc, pcd = pc.remove_statistical_outlier(nb_neighbors=7, std_ratio=10.0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(p)
    pc.colors = o3d.utility.Vector3dVector(columns)
    pc, pcd = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pc

#mesh generation for each image(all 6 different images)
def mesh(pc, obj):
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc.orient_normals_towards_camera_location(camera_location=np.array([0,0,-1]))

    m, d = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9, scale=1.1, linear_fit=False)
    mask = np.asarray(d) < np.quantile(d, 0.05)
    m.remove_vertices_by_mask(mask)
    m.compute_vertex_normals()

    b = m.get_axis_aligned_bounding_box()
    m.translate(-b.get_center())
    m = m.filter_smooth_laplacian(number_of_iterations=3)

    o3d.io.write_triangle_mesh(obj, m)
    return m

ip = input("Please type the image file name from the /input: ").strip()
ip = "input/" + ip
op  = input("Please type the output folder name inside /output: ").strip() or "results"
op = "output/" + op
os.makedirs(op, exist_ok=True)

rgb = loading(ip)
displaing(rgb, "Input")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True) # Using the DPT_Hybrid model as mentioned in the propsal
trans = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
model.to(device).eval()

def midas(image):
    input = trans(image).to(device)
    with torch.no_grad():
        p = model(input)
        p = torch.nn.functional.interpolate(p.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    m = p - p.min()
    n = p.max()-p.min()+1e-6
    d = m/n
    return d

mi = midas(rgb)

m_lp = lpVarience(mi)
m_depth = float(mi.max() - mi.min())
m_sharp = edgeSharp(mi)
m_total = totalScore(m_lp, m_depth, m_sharp)

c = cv2.applyColorMap((mi * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
cv2.imwrite(os.path.join(op, "map_midas.png"), cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

p = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")  # Using the Depth-Anything-V2-Small-hf model as mentioned in the propsal

def depth_anything(image):
    d = np.array(p(Image.fromarray(image))["depth"])
    m = d - d.min()
    n = d.max()-d.min()+1e-6
    return m/n

da = depth_anything(rgb)

da_lp = lpVarience(da)
da_depth = float(da.max() - da.min())
da_sharp = edgeSharp(da)
da_total = totalScore(da_lp, da_depth, da_sharp)

c = cv2.applyColorMap((da * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
cv2.imwrite(os.path.join(op, "map_depth_anything.png"), cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

z = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=False)   # Using the ZoeD_N model as mentioned in the propsal
z.to(device).eval()
checkpoint = torch.hub.load_state_dict_from_url("https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt?raw=true",map_location=device)

# z.load_state_dict(checkpoint["model"], strict=False)
# for mod in z.modules():
#     if hasattr(mod, "drop_path1") and not hasattr(mod, "drop_path"):
#         setattr(mod, "drop_path", getattr(mod, "drop_path1"))
        
z.load_state_dict(checkpoint["model"], strict=False)
for mod in z.modules():
    if hasattr(mod, "drop_path1") and not hasattr(mod, "drop_path"):
        setattr(mod, "drop_path", getattr(mod, "drop_path1"))

def zoe_depth(image):
    with torch.no_grad():
        d = z.infer_pil(Image.fromarray(image))
    m = d - d.min()
    n = d.max()-d.min()+1e-6
    return m/n

z = zoe_depth(rgb)

z_lp = lpVarience(z)
z_depth = float(z.max() - z.min())
z_sharp = edgeSharp(z)
z_total = totalScore(z_lp, z_depth, z_sharp)

print(f"\nEVALUATION METRICS of MIDAS:")
print(f"Laplacian Variance: {m_lp:.4f}")
print(f"Depth Range: {m_depth:.4f}")
print(f"Edge Sharpness: {m_sharp:.4f}")
print(f"Combined Score: {m_total:.4f}")

print(f"\nEVALUATION METRICS of DEPTHANYTHING:")
print(f"Laplacian Variance: {da_lp:.4f}")
print(f"Depth Range: {da_depth:.4f}")
print(f"Edge Sharpness: {da_sharp:.4f}")
print(f"Combined Score: {da_total:.4f}")

print(f"\nEVALUATION METRICS of ZEODEPTH:")
print(f"Laplacian Variance: {z_lp:.4f}")
print(f"Depth Range: {z_depth:.4f}")
print(f"Edge Sharpness: {z_sharp:.4f}")
print(f"Combined Score: {z_total:.4f}")

c = cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
cv2.imwrite(os.path.join(op, "map_zoe_depth.png"), cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

x = np.arange(3)
w = 0.25

plt.figure(figsize=(7, 4))
plt.legend()
plt.tight_layout()

plt.bar(x,[m_depth, da_depth, z_depth], w, label="Depth Range")
plt.bar(x - w, [m_lp, da_lp, z_lp],w, label="Laplacian Variance")
plt.bar(x + w, [m_sharp, da_sharp, z_sharp], w, label="Edge Sharpness")

plt.title("comparison chart")
plt.xticks(x, ["midas", "depth_anything", "zoe"])
plt.ylabel("Score Value")


plot_path = os.path.join(op, "analysis_plot.png") # detailed analysis explanation in report
plt.savefig(plot_path)
plt.show()

parallaxingEffect(rgb, mi, os.path.join(op, "parallaxing_midas.gif"))
parallaxingEffect(rgb, da, os.path.join(op, "parallaxing_depth_anything.gif"))
parallaxingEffect(rgb, z, os.path.join(op, "parallaxing_zoe_depth.gif"))

cloud_mi = pc(rgb, mi)
o3d.io.write_point_cloud(os.path.join(op, "point_cloud_midas.ply"), cloud_mi)
cloud_da = pc(rgb, da)
o3d.io.write_point_cloud(os.path.join(op, "point_cloud_depth_anything.ply"), cloud_da)
cloud_z = pc(rgb, z)
o3d.io.write_point_cloud(os.path.join(op, "point_cloud_zoe_depth.ply"), cloud_z)

mesh(cloud_mi, os.path.join(op, "mesh_midas.obj"))
mesh(cloud_da, os.path.join(op, "mesh_depth_anything.obj"))
mesh(cloud_z, os.path.join(op, "mesh_zoe_depth.obj"))

print(f'\n Open "{op}/" for the results')
# Use Blender for visualising the point cloud .ply and mesh .obj files