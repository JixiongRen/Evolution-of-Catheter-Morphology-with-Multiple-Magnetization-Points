# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import webbrowser

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None  # optional

from scipy.spatial import ConvexHull


def load_points(npz_path: Path, use_success: bool = True):
    d = np.load(npz_path, allow_pickle=True)
    P = np.asarray(d["p"], dtype=np.float64)
    success = np.asarray(d["success"]).astype(bool) if "success" in d else np.ones((P.shape[0],), dtype=bool)
    if use_success:
        P_ok = P[success]
        P_fail = P[~success]
        return P_ok, P_fail
    else:
        # return all points as ok, empty failed
        return P, np.empty((0, 3), dtype=np.float64)


def surface_via_open3d_alpha(P: np.ndarray, alpha: float):
    if o3d is None:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(alpha))
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        return None


def surface_via_open3d_poisson(P: np.ndarray, depth: int = 8):
    if o3d is None:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        return None


def surface_via_convex_hull(P: np.ndarray):
    # Fallback: convex hull as an outer surface
    hull = ConvexHull(P)
    faces = hull.simplices  # indices of triangles
    V = P
    return V, faces


def plot_mesh_matplotlib(V: np.ndarray, F: np.ndarray, out_png: Path, elev: float = 20, azim: float = -60):
    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    tris = [V[tri] for tri in F]
    coll = Poly3DCollection(tris, alpha=0.8)
    coll.set_facecolor((0.4, 0.6, 0.9, 0.6))
    coll.set_edgecolor((0.1, 0.1, 0.1, 0.2))
    ax.add_collection3d(coll)
    ax.scatter(V[:,0], V[:,1], V[:,2], s=4, c='k', alpha=0.3)

    xyz_min = V.min(axis=0)
    xyz_max = V.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    extent = (xyz_max - xyz_min).max()
    lims = np.vstack([center - extent/2, center + extent/2])
    ax.set_xlim(lims[0,0], lims[1,0])
    ax.set_ylim(lims[0,1], lims[1,1])
    ax.set_zlim(lims[0,2], lims[1,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def show_mesh_matplotlib(V: np.ndarray, F: np.ndarray, P: np.ndarray | None = None, elev: float = 20, azim: float = -60, Pf: np.ndarray | None = None):
    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    tris = [V[tri] for tri in F]
    coll = Poly3DCollection(tris, alpha=0.8)
    coll.set_facecolor((0.4, 0.6, 0.9, 0.6))
    coll.set_edgecolor((0.1, 0.1, 0.1, 0.2))
    ax.add_collection3d(coll)
    if P is not None:
        ax.scatter(P[:,0], P[:,1], P[:,2], s=6, c='k', alpha=0.35)
    if Pf is not None and Pf.size > 0:
        ax.scatter(Pf[:,0], Pf[:,1], Pf[:,2], s=10, c='r', alpha=0.8)

    xyz_min = V.min(axis=0)
    xyz_max = V.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    extent = (xyz_max - xyz_min).max()
    lims = np.vstack([center - extent/2, center + extent/2])
    ax.set_xlim(lims[0,0], lims[1,0])
    ax.set_ylim(lims[0,1], lims[1,1])
    ax.set_zlim(lims[0,2], lims[1,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    plt.show()


def show_mesh_plotly(V: np.ndarray, F: np.ndarray, P: np.ndarray | None, out_html: Path | None) -> Path | None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return None

    i = F[:, 0].astype(int)
    j = F[:, 1].astype(int)
    k = F[:, 2].astype(int)
    mesh = go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=i, j=j, k=k, opacity=0.6, color='lightblue')
    data = [mesh]
    if P is not None:
        scatter = go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], mode='markers', marker=dict(size=2, color='black', opacity=0.5))
        data.append(scatter)
    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), margin=dict(l=0, r=0, t=30, b=0))
    if out_html is None:
        out_html = Path.cwd() / 'workspace_surface_view.html'
    fig.write_html(str(out_html), include_plotlyjs='cdn', auto_open=False)
    try:
        webbrowser.open(out_html.as_uri())
    except Exception:
        pass
    return out_html


def _save_single_view(V: np.ndarray, F: np.ndarray, P: np.ndarray | None, elev: float, azim: float, out_png: Path, Pf: np.ndarray | None = None, depth_axis: str = 'z'):
    fig = plt.figure(figsize=(5.2, 4.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    tris = [V[tri] for tri in F]
    coll = Poly3DCollection(tris, alpha=0.9)
    coll.set_facecolor((0.5, 0.7, 0.95, 0.9))
    coll.set_edgecolor((0.1, 0.1, 0.1, 0.15))
    ax.add_collection3d(coll)
    if P is not None:
        ax.scatter(P[:,0], P[:,1], P[:,2], s=4, c='k', alpha=0.25)
    if Pf is not None and Pf.size > 0:
        ax.scatter(Pf[:,0], Pf[:,1], Pf[:,2], s=8, c='r', alpha=0.9)

    xyz_min = V.min(axis=0)
    xyz_max = V.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    extent = (xyz_max - xyz_min).max()
    lims = np.vstack([center - extent/2, center + extent/2])
    ax.set_xlim(lims[0,0], lims[1,0])
    ax.set_ylim(lims[0,1], lims[1,1])
    ax.set_zlim(lims[0,2], lims[1,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    # Orthographic projection to remove perspective / depth distortion
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass
    # cleaner look
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for tick in axis.get_ticklines():
            tick.set_visible(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor((1,1,1,0))
        pane.set_facecolor((1,1,1,0))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_three_views(V: np.ndarray, F: np.ndarray, P: np.ndarray | None, out_prefix: Path, Pf: np.ndarray | None = None):
    # Standard views: top (XY), front (XZ), side (YZ)
    top = out_prefix.parent / (out_prefix.stem + '_view_top.png')
    front = out_prefix.parent / (out_prefix.stem + '_view_front.png')
    side = out_prefix.parent / (out_prefix.stem + '_view_side.png')
    # Matplotlib view_init angles:
    # - Top (XY): elev=90, azim=-90
    # - Front (XZ, looking from +Y): elev=0, azim=-90
    # - Side (YZ, looking from +X): elev=0, azim=0
    _save_single_view(V, F, P, elev=90, azim=-90, out_png=top, Pf=Pf, depth_axis='z')   # top view, depth=z
    _save_single_view(V, F, P, elev=0, azim=-90, out_png=front, Pf=Pf, depth_axis='y') # front view, depth=y
    _save_single_view(V, F, P, elev=0, azim=0, out_png=side, Pf=Pf, depth_axis='x')    # side view, depth=x
    return top, front, side


def laplacian_smooth(V: np.ndarray, F: np.ndarray, iters: int = 10, lam: float = 0.5) -> np.ndarray:
    if iters <= 0 or lam <= 0:
        return V


def _build_adjacency(F: np.ndarray, n_vert: int):
    nbrs = [[] for _ in range(n_vert)]
    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        nbrs[a].extend([b, c])
        nbrs[b].extend([a, c])
        nbrs[c].extend([a, b])
    for i in range(n_vert):
        if not nbrs[i]:
            nbrs[i] = [i]
    return [np.unique(n) for n in nbrs]


def taubin_smooth(V: np.ndarray, F: np.ndarray, iters: int = 10, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    # Taubin smoothing: alternate Laplacian steps with positive and negative factors to reduce shrinkage
    if iters <= 0:
        return V
    V = np.asarray(V, dtype=np.float64).copy()
    n = V.shape[0]
    nbrs = _build_adjacency(F, n)
    for _ in range(int(iters)):
        # step with lambda
        V_new = V.copy()
        for i in range(n):
            mean_n = V[nbrs[i]].mean(axis=0)
            V_new[i] = V[i] + lam * (mean_n - V[i])
        V = V_new
        # step with mu
        V_new = V.copy()
        for i in range(n):
            mean_n = V[nbrs[i]].mean(axis=0)
            V_new[i] = V[i] + mu * (mean_n - V[i])
        V = V_new
    return V


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, default='workspace_determine/workspace_out/samples.npz')
    parser.add_argument('--out_prefix', type=str, default='workspace_determine/workspace_out/workspace_surface')
    parser.add_argument('--method', type=str, default='alpha', choices=['alpha','poisson','convex'])
    parser.add_argument('--alpha', type=float, default=0.02, help='alpha for alpha-shape (scale roughly scene units)')
    parser.add_argument('--poisson_depth', type=int, default=8)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--use_success', dest='use_success', action='store_true')
    g.add_argument('--no_use_success', dest='use_success', action='store_false')
    parser.set_defaults(use_success=True)
    parser.add_argument('--include_failed', action='store_true', default=True, help='Plot failed samples as red dots')
    parser.add_argument('--viewer', type=str, default='plotly', choices=['plotly','mpl'], help='Viewer for 3D display')
    parser.add_argument('--show', action='store_true', help='Show interactive 3D view after building surface')
    parser.add_argument('--save_views', action='store_true', help='Also save three orthographic views as PNGs')
    parser.add_argument('--smooth_iters', type=int, default=0, help='Smoothing iterations (0 to disable)')
    parser.add_argument('--smooth_lambda', type=float, default=0.5, help='Smoothing step size in (0,1]')
    parser.add_argument('--smoother', type=str, default='taubin', choices=['laplacian','taubin'], help='Smoothing scheme')
    parser.add_argument('--inflate_frac', type=float, default=0.0, help='Uniform inflation after smoothing (e.g., 0.05 = +5% outward)')
    args = parser.parse_args()

    npz_path = Path(args.npz).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    P_ok, P_fail = load_points(npz_path, use_success=True)
    P = P_ok if bool(args.use_success) else np.vstack([P_ok, P_fail])
    if P.shape[0] < 4:
        raise ValueError('Not enough points to form a surface')

    out_prefix = Path(args.out_prefix).resolve()
    out_png = out_prefix.with_suffix('.png')
    out_ply = out_prefix.with_suffix('.ply')

    V, F = None, None

    if args.method in ('alpha','poisson') and o3d is None:
        print('open3d not available, falling back to convex hull')
        args.method = 'convex'

    if args.method == 'alpha':
        mesh = surface_via_open3d_alpha(P, float(args.alpha))
        if mesh is None or len(mesh.triangles) == 0:
            print('alpha-shape failed; falling back to convex hull')
            V, F = surface_via_convex_hull(P)
        else:
            o3d.io.write_triangle_mesh(str(out_ply), mesh)
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
    elif args.method == 'poisson':
        mesh = surface_via_open3d_poisson(P, int(args.poisson_depth))
        if mesh is None or len(mesh.triangles) == 0:
            print('poisson reconstruction failed; falling back to convex hull')
            V, F = surface_via_convex_hull(P)
        else:
            o3d.io.write_triangle_mesh(str(out_ply), mesh)
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
    else:
        V, F = surface_via_convex_hull(P)

    # Optional smoothing on the mesh
    smooth_iters = int(getattr(args, 'smooth_iters', 0)) if hasattr(args, 'smooth_iters') else 0
    smooth_lambda = float(getattr(args, 'smooth_lambda', 0.5)) if hasattr(args, 'smooth_lambda') else 0.5
    if smooth_iters > 0:
        if args.smoother == 'taubin':
            V = taubin_smooth(V, F, iters=smooth_iters, lam=smooth_lambda, mu=-0.53)
        else:
            V = laplacian_smooth(V, F, iters=smooth_iters, lam=smooth_lambda)
    # Optional inflation to re-enclose points after smoothing (reduce shrinkage)
    if float(getattr(args, 'inflate_frac', 0.0)) != 0.0:
        frac = float(args.inflate_frac)
        c = np.mean(V, axis=0)
        V = c + (1.0 + frac) * (V - c)

    # Plot & save PNG
    plot_mesh_matplotlib(V, F, out_png)

    # Also save raw arrays for reuse
    out_npz = out_prefix.with_suffix('.npz')
    np.savez(out_npz, V=V, F=F.astype(np.int32), points=P, npz=str(npz_path), method=str(args.method))
    print('saved:', out_png)
    print('saved:', out_npz)
    if (args.method in ('alpha','poisson')) and o3d is not None:
        print('saved:', out_ply)
    if args.save_views:
        t, f, s = save_three_views(V, F, P, out_prefix, Pf=(P_fail if args.include_failed else None))
        print('saved:', t)
        print('saved:', f)
        print('saved:', s)

    # Optional interactive view
    if args.show:
        if args.viewer == 'plotly':
            out_html = out_prefix.with_suffix('.html')
            page = show_mesh_plotly(V, F, P, out_html)
            if page is None:
                print('Plotly not available. Falling back to Matplotlib viewer...')
                show_mesh_matplotlib(V, F, P, Pf=(P_fail if args.include_failed else None))
            else:
                print('opened in browser:', page)
        else:
            show_mesh_matplotlib(V, F, P, Pf=(P_fail if args.include_failed else None))


if __name__ == '__main__':
    main()
