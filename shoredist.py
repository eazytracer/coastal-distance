"""Distance-to-shoreline rasters on BlueTopo grids.

Implements the method recommended in ``coastal_distance_methods.ipynb``: vector dead reckoning with
halo seeding and local refinement. Every pixel ends up holding (almost exactly) the Euclidean distance
from its centre to the nearest shoreline vertex, and because tile borders are seeded from a *global*
shoreline index, each tile can be computed independently and still stitch seamlessly.

The numba kernels release the GIL, so tiles can be processed concurrently with a ThreadPoolExecutor
while sharing one in-memory ShorelineIndex.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd
import pyogrio
import requests
import shapely
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree

BUCKET = "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com"
CUSP_URL = "https://nsde.ngs.noaa.gov/downloads/{name}.zip"


# --------------------------------------------------------------------------------------------- data
def fetch(url, dest, sha256=None):
    """Download url -> dest once (cached). Verifies SHA-256 when given."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.rename(dest)
    if sha256:
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == sha256, f"checksum mismatch for {dest}"
    return dest


def latest_tile_scheme(data_dir):
    """Download (once) and return the path of the current BlueTopo tile-scheme GeoPackage."""
    xml = requests.get(f"{BUCKET}/?list-type=2&prefix=BlueTopo/_BlueTopo_Tile_Scheme/", timeout=60).text
    key = sorted(re.findall(r"<Key>([^<]+\.gpkg)</Key>", xml))[-1]
    return fetch(f"{BUCKET}/{key}", Path(data_dir) / "bluetopo" / Path(key).name)


def cusp_regions(w, s, e, n):
    """Names of the 5x5 degree CUSP packages (SW-corner named, e.g. N25W085) touching a lon/lat box."""
    names = []
    for lat in range(int(math.floor(s / 5) * 5), int(math.ceil(n / 5) * 5), 5):
        for lon_w in range(int(math.ceil(-e / 5) * 5), int(math.ceil(-w / 5) * 5) + 5, 5):
            names.append(f"N{lat:02d}W{lon_w:03d}")
    return names


def load_cusp(box, crs, data_dir):
    """CUSP shoreline lines inside ``box`` (xmin, ymin, xmax, ymax in ``crs``), reprojected to ``crs``."""
    w, s, e, n = transform_bounds(crs, "EPSG:4269", *box)
    parts = []
    for name in cusp_regions(w, s, e, n):
        z = Path(data_dir) / "cusp" / f"{name}.zip"
        try:
            fetch(CUSP_URL.format(name=name), z)
        except requests.HTTPError:
            continue                                            # no package = open ocean
        parts.append(pyogrio.read_dataframe(f"/vsizip/{z}/{name}.shp", bbox=(w, s, e, n)))
    lines = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).to_crs(crs)
    lines = lines.clip(shapely.box(*box)).explode(index_parts=False)
    return lines[lines.geom_type == "LineString"].reset_index(drop=True)


# ----------------------------------------------------------------------------------- shoreline index
@dataclass
class ShorelineIndex:
    """Densified shoreline vertices, sorted into square bins, plus a KD-tree over the same order."""
    geoms: np.ndarray       # shapely LineStrings (for rasterising seeds)
    strtree: shapely.STRtree
    ptx: np.ndarray
    pty: np.ndarray
    tree: cKDTree
    cell_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    gx0: float
    gy0: float
    bin_size: float
    gW: int

    @classmethod
    def build(cls, lines, densify=2.0, bin_size=8.0):
        geoms = np.asarray(lines.geometry.values)
        pts = shapely.get_coordinates(shapely.segmentize(geoms, densify))
        gx0, gy0 = pts[:, 0].min() - bin_size, pts[:, 1].max() + bin_size
        gW = int(math.ceil((pts[:, 0].max() - gx0) / bin_size)) + 2
        cid = ((gy0 - pts[:, 1]) // bin_size).astype(np.int64) * gW + ((pts[:, 0] - gx0) // bin_size).astype(np.int64)
        order = np.argsort(cid, kind="stable")
        cell_ids, starts = np.unique(cid[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        ptx = np.ascontiguousarray(pts[order, 0])
        pty = np.ascontiguousarray(pts[order, 1])
        return cls(geoms, shapely.STRtree(geoms), ptx, pty, cKDTree(np.column_stack([ptx, pty])),
                   cell_ids, starts.astype(np.int64), ends.astype(np.int64), float(gx0), float(gy0),
                   float(bin_size), gW)

    @property
    def n_vertices(self):
        return len(self.ptx)

    def lines_in(self, box):
        return self.geoms[self.strtree.query(shapely.box(*box))]


# ------------------------------------------------------------------------------------------ kernels
@nb.njit(cache=True, nogil=True)
def _dr_sweeps(fid, d, px_x, px_y, ptx, pty, npass, offs):
    """Forward (↘) / backward (↖) raster sweeps propagating pointers to real shoreline vertices."""
    H, W = fid.shape
    for _ in range(npass):
        for sgn in (1, -1):
            for ii in range(H):
                i = ii if sgn == 1 else H - 1 - ii
                for jj in range(W):
                    j = jj if sgn == 1 else W - 1 - jj
                    best = d[i, j]
                    bk = fid[i, j]
                    for m in range(offs.shape[0]):
                        a = i + sgn * offs[m, 0]
                        c = j + sgn * offs[m, 1]
                        if a < 0 or a >= H or c < 0 or c >= W:
                            continue
                        k = fid[a, c]
                        if k < 0 or k == bk:
                            continue
                        dd = math.hypot(px_x[j] - ptx[k], px_y[i] - pty[k])
                        if dd < best:
                            best = dd
                            bk = k
                    d[i, j] = best
                    fid[i, j] = bk


@nb.njit(cache=True, nogil=True)
def _dr_init(fid, d, px_x, px_y, ptx, pty):
    H, W = fid.shape
    for i in range(H):
        for j in range(W):
            k = fid[i, j]
            d[i, j] = math.hypot(px_x[j] - ptx[k], px_y[i] - pty[k]) if k >= 0 else np.inf


@nb.njit(cache=True, nogil=True)
def _dr_refine(fid, d, px_x, px_y, ptx, pty, cell_ids, starts, ends, gx0, gy0, cs, gW):
    """Search the vertices binned in the 3x3 cells around each pixel's current vertex; keep the closest."""
    H, W = fid.shape
    for i in range(H):
        for j in range(W):
            k = fid[i, j]
            if k < 0:
                continue
            cx = int((ptx[k] - gx0) // cs)
            cy = int((gy0 - pty[k]) // cs)
            best = d[i, j]
            bk = k
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    cid = (cy + dy) * gW + (cx + dx)
                    p = np.searchsorted(cell_ids, cid)
                    if p < cell_ids.shape[0] and cell_ids[p] == cid:
                        for q in range(starts[p], ends[p]):
                            dd = math.hypot(px_x[j] - ptx[q], px_y[i] - pty[q])
                            if dd < best:
                                best = dd
                                bk = q
            d[i, j] = best
            fid[i, j] = bk


# 3x3 causal neighbours + knight moves (forward direction; backward uses the negated offsets)
OFFS_5x5 = np.array([(i, j) for i in range(-2, 1) for j in range(-2, 3)
                     if (i < 0 or j < 0) and abs(i) + abs(j) <= 3 and (i, j) != (0, 0)], np.int64)


def warmup():
    """Compile the numba kernels once (cached on disk afterwards)."""
    f = np.full((3, 3), -1, np.int64)
    f[1, 1] = 0
    d = np.empty((3, 3))
    x = np.arange(3.0)
    one = np.zeros(1)
    _dr_init(f, d, x, x, one, one)
    _dr_sweeps(f, d, x, x, one, one, 1, OFFS_5x5)
    _dr_refine(f, d, x, x, one, one, np.zeros(1, np.int64), np.zeros(1, np.int64), np.ones(1, np.int64),
               0.0, 0.0, 1.0, 1)


def pixel_centres(transform, shape):
    H, W = shape
    px_x = transform.c + (np.arange(W) + 0.5) * transform.a
    px_y = transform.f + (np.arange(H) + 0.5) * transform.e
    return px_x, px_y


def distance_on_grid(index: ShorelineIndex, transform, shape, rounds=2, offs=OFFS_5x5):
    """Distance (m) from every pixel centre of a north-up grid to the nearest shoreline vertex in ``index``.

    1. seed pixels (shoreline burned in with all_touched) point at their nearest vertex;
    2. halo: border pixels point at their *global* nearest vertex (KD-tree), so off-tile shoreline is
       accounted for without padding;
    3. two forward/backward pointer-propagation passes, then ``rounds`` x (local refinement + one pass).
    """
    H, W = shape
    px_x, px_y = pixel_centres(transform, shape)
    xmin, ymax = transform.c, transform.f
    xmax, ymin = xmin + W * transform.a, ymax + H * transform.e
    fid = np.full((H, W), -1, np.int64)

    lines = index.lines_in((xmin, ymin, xmax, ymax))
    if len(lines):
        seeds = rasterize(((g, 1) for g in lines), out_shape=(H, W), transform=transform,
                          all_touched=True, dtype="uint8").astype(bool)
    else:
        seeds = np.zeros((H, W), bool)
    si, sj = np.nonzero(seeds)
    if len(si):
        fid[si, sj] = index.tree.query(np.column_stack([px_x[sj], px_y[si]]))[1]
    halo = np.zeros((H, W), bool)
    halo[[0, -1], :] = True
    halo[:, [0, -1]] = True
    halo &= ~seeds
    bi, bj = np.nonzero(halo)
    fid[bi, bj] = index.tree.query(np.column_stack([px_x[bj], px_y[bi]]))[1]

    d = np.empty((H, W))
    _dr_init(fid, d, px_x, px_y, index.ptx, index.pty)
    _dr_sweeps(fid, d, px_x, px_y, index.ptx, index.pty, 2, offs)
    for r in range(rounds):
        _dr_refine(fid, d, px_x, px_y, index.ptx, index.pty, index.cell_ids, index.starts, index.ends,
                   index.gx0, index.gy0, index.bin_size, index.gW)
        if r < rounds - 1:
            _dr_sweeps(fid, d, px_x, px_y, index.ptx, index.pty, 1, offs)
    return d.astype(np.float32), int(seeds.sum())
