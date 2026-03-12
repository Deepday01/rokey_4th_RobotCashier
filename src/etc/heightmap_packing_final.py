#!/usr/bin/env python3
"""
Heightmap-based 3D packing baseline.

Goal:
- From a known set of real measured objects, choose n objects.
- Pack them into a basket using a heightmap/grid-based heuristic search.
- Return the best found arrangement and visualize it.

Key rules:
- 90-degree axis-aligned rotations only
- No overlap
- Must stay fully inside the basket
- Must be sufficiently supported (support ratio threshold)
- Heightmap/grid search for candidate XY placement

This is a non-learning baseline designed to be compared later with RL
and can be expanded to ROS service/message output later.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# User settings
# ============================================================

BASKET_SIZE = (170, 130, 75)  # mm
TARGET_N = 8                  # choose n objects from catalog
GRID_SIZE = 5                 # mm per cell for XY heightmap
SUPPORT_THRESHOLD = 0.70
TOP_K_CANDIDATES_PER_ITEM = 20  # keep strongest candidates per item/rotation

OBJECT_CATALOG = [
    {"name": "애크논크림", "size": (125, 35, 20), "durability": 3},
    {"name": "카라멜1", "size": (70, 45, 25), "durability": 1},
    {"name": "카라멜2", "size": (70, 45, 25), "durability": 1},
    {"name": "나비 블럭", "size": (80, 50, 40), "durability": 4},
    {"name": "아이셔", "size": (110, 75, 40), "durability": 2},
    {"name": "이클립스", "size": (80, 50, 15), "durability": 5},
    {"name": "이클립스빨강", "size": (80, 45, 25), "durability": 5},
    {"name": "이클립스노랑", "size": (80, 45, 25), "durability": 5},
]


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class Item:
    object_index: int
    name: str
    size: Tuple[int, int, int]
    durability: int


@dataclass(frozen=True)
class Placement:
    object_index: int
    name: str
    original_size: Tuple[int, int, int]
    size: Tuple[int, int, int]      # rotated size in mm
    position: Tuple[int, int, int]  # (x, y, z) in mm
    durability: int
    support_ratio: float
    rotation_rpy: Tuple[int, int, int]


@dataclass
class Solution:
    selected_items: List[Item]
    placements: List[Placement]
    score: float
    packed_count: int
    fill_ratio: float
    used_height: int
    footprint_util: float
    void_ratio: float


# ============================================================
# Geometry helpers
# ============================================================


def unique_rotations(size: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    return sorted(set(itertools.permutations(size, 3)))


# This mapping is intended for downstream message/service use.
# It provides a deterministic 90-degree RPY tag for each axis-aligned size permutation.
# For boxes with repeated dimensions, multiple RPY values could represent the same shape,
# but a single canonical mapping is sufficient for output formatting.

def size_to_rpy(original: Tuple[int, int, int], rotated: Tuple[int, int, int]) -> Tuple[int, int, int]:
    ox, oy, oz = original
    mapping = {
        (ox, oy, oz): (0, 0, 0),
        (ox, oz, oy): (90, 0, 0),
        (oy, ox, oz): (0, 0, 90),
        (oy, oz, ox): (0, 90, 0),
        (oz, ox, oy): (90, 90, 0),
        (oz, oy, ox): (0, 90, 90),
    }
    return mapping.get(rotated, (0, 0, 0))



def boxes_intersect(p1: Placement, p2: Placement) -> bool:
    x1, y1, z1 = p1.position
    w1, d1, h1 = p1.size
    x2, y2, z2 = p2.position
    w2, d2, h2 = p2.size

    overlap_x = (x1 < x2 + w2) and (x2 < x1 + w1)
    overlap_y = (y1 < y2 + d2) and (y2 < y1 + d1)
    overlap_z = (z1 < z2 + h2) and (z2 < z1 + h1)
    return overlap_x and overlap_y and overlap_z



def in_bounds(pos: Tuple[int, int, int], size: Tuple[int, int, int], basket: Tuple[int, int, int]) -> bool:
    x, y, z = pos
    w, d, h = size
    bx, by, bz = basket
    return x >= 0 and y >= 0 and z >= 0 and (x + w) <= bx and (y + d) <= by and (z + h) <= bz



def overlap_area_2d(ax: int, ay: int, aw: int, ad: int, bx: int, by: int, bw: int, bd: int) -> int:
    ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0, min(ay + ad, by + bd) - max(ay, by))
    return ox * oy



def total_item_volume(placements: Sequence[Placement]) -> int:
    return sum(p.size[0] * p.size[1] * p.size[2] for p in placements)



def bounding_box_volume(placements: Sequence[Placement]) -> int:
    if not placements:
        return 0
    max_x = max(p.position[0] + p.size[0] for p in placements)
    max_y = max(p.position[1] + p.size[1] for p in placements)
    max_z = max(p.position[2] + p.size[2] for p in placements)
    return max_x * max_y * max_z



def used_height(placements: Sequence[Placement]) -> int:
    if not placements:
        return 0
    return max(p.position[2] + p.size[2] for p in placements)


# ============================================================
# Heightmap
# ============================================================

class HeightMap:
    def __init__(self, basket_size: Tuple[int, int, int], grid_size: int):
        self.bx, self.by, self.bz = basket_size
        self.grid = grid_size
        self.gx = self.bx // self.grid
        self.gy = self.by // self.grid
        if self.gx <= 0 or self.gy <= 0:
            raise ValueError("Grid size is too large for basket dimensions.")
        self.heights = np.zeros((self.gy, self.gx), dtype=np.int32)

    def copy(self) -> "HeightMap":
        hm = HeightMap((self.bx, self.by, self.bz), self.grid)
        hm.heights = self.heights.copy()
        return hm

    def mm_to_cells_xy(self, w: int, d: int) -> Tuple[int, int]:
        wc = math.ceil(w / self.grid)
        dc = math.ceil(d / self.grid)
        return wc, dc

    def cell_to_mm_xy(self, cx: int, cy: int) -> Tuple[int, int]:
        return cx * self.grid, cy * self.grid

    def region_height(self, cx: int, cy: int, wc: int, dc: int) -> int:
        return int(self.heights[cy:cy + dc, cx:cx + wc].max())

    def region_stats(self, cx: int, cy: int, wc: int, dc: int) -> Dict[str, float]:
        region = self.heights[cy:cy + dc, cx:cx + wc]
        base_h = int(region.max())
        cover = float(np.mean(region == base_h))
        roughness = float(region.max() - region.min())
        return {"base_h": base_h, "support_ratio": cover, "roughness": roughness}

    def place(self, cx: int, cy: int, wc: int, dc: int, top_h: int) -> None:
        self.heights[cy:cy + dc, cx:cx + wc] = top_h

    def footprint_utilization(self, placements: Sequence[Placement]) -> float:
        if not placements:
            return 0.0
        occupied = np.zeros_like(self.heights, dtype=np.uint8)
        for p in placements:
            x, y, _ = p.position
            w, d, _ = p.size
            wc, dc = self.mm_to_cells_xy(w, d)
            cx = min(self.gx - wc, x // self.grid)
            cy = min(self.gy - dc, y // self.grid)
            occupied[cy:cy + dc, cx:cx + wc] = 1
        return float(occupied.mean())


# ============================================================
# Feasibility and scoring
# ============================================================


def exact_support_ratio(
    pos: Tuple[int, int, int],
    size: Tuple[int, int, int],
    placements: Sequence[Placement],
) -> float:
    x, y, z = pos
    w, d, _ = size
    if z == 0:
        return 1.0

    base_area = w * d
    support_area = 0

    for p in placements:
        px, py, pz = p.position
        pw, pd, ph = p.size
        top_z = pz + ph
        if top_z != z:
            continue
        support_area += overlap_area_2d(x, y, w, d, px, py, pw, pd)

    return support_area / base_area if base_area > 0 else 0.0



def durability_penalty(candidate: Placement, placements: Sequence[Placement]) -> float:
    x, y, z = candidate.position
    w, d, _ = candidate.size
    pen = 0.0
    for p in placements:
        px, py, pz = p.position
        pw, pd, ph = p.size
        if pz + ph != z:
            continue
        overlap = overlap_area_2d(x, y, w, d, px, py, pw, pd)
        if overlap <= 0:
            continue
        if candidate.durability > p.durability:
            pen += 1.8 * (candidate.durability - p.durability) * (overlap / max(1, w * d))
    return pen



def candidate_score(
    candidate: Placement,
    placements: Sequence[Placement],
    basket: Tuple[int, int, int],
    heightmap: HeightMap,
) -> float:
    bx, by, bz = basket
    x, y, z = candidate.position
    w, d, h = candidate.size

    new_list = list(placements) + [candidate]

    compact_fill = total_item_volume(new_list) / max(1, bounding_box_volume(new_list))
    current_top = used_height(placements)
    new_top = used_height(new_list)
    height_increase = max(0, new_top - current_top)

    wall_bonus = 0.0
    if x == 0:
        wall_bonus += 0.5
    if y == 0:
        wall_bonus += 0.5
    if z == 0:
        wall_bonus += 1.0
    if x + w == bx:
        wall_bonus += 0.2
    if y + d == by:
        wall_bonus += 0.2

    centered_penalty = 0.002 * (x + y)
    top_gap_penalty = 0.05 * max(0, (z + h) - 0.85 * bz)
    dura_pen = durability_penalty(candidate, placements)

    base_area = w * d
    flat_bonus = 0.012 * (base_area / max(1, h))

    wc, dc = heightmap.mm_to_cells_xy(w, d)
    cx = x // heightmap.grid
    cy = y // heightmap.grid
    roughness = 0.0
    if cx + wc <= heightmap.gx and cy + dc <= heightmap.gy:
        roughness = heightmap.region_stats(cx, cy, wc, dc)["roughness"]

    score = 0.0
    score += 12.0 * candidate.support_ratio
    score += 9.0 * compact_fill
    score += wall_bonus
    score += flat_bonus
    score -= 0.18 * z
    score -= 0.08 * height_increase
    score -= centered_penalty
    score -= top_gap_penalty
    score -= 0.04 * roughness
    score -= dura_pen
    return float(score)




def solution_quality(sol_placements: Sequence[Placement], basket: Tuple[int, int, int], heightmap: HeightMap) -> Tuple[float, float, int, float, float]:
    if not sol_placements:
        return 0.0, 0.0, 0, 0.0, 1.0

    total_vol = total_item_volume(sol_placements)
    used_h = used_height(sol_placements)
    effective_vol = basket[0] * basket[1] * max(1, used_h)
    fill_ratio = total_vol / effective_vol
    footprint_util = heightmap.footprint_utilization(sol_placements)
    avg_support = sum(p.support_ratio for p in sol_placements) / len(sol_placements)
    bbox_fill = total_vol / max(1, bounding_box_volume(sol_placements))
    void_ratio = max(0.0, 1.0 - fill_ratio)

    quality = 10.0 * fill_ratio + 5.0 * bbox_fill + 2.5 * avg_support + 1.5 * footprint_util - 0.06 * used_h - 2.0 * void_ratio
    return float(quality), float(fill_ratio), int(used_h), float(footprint_util), float(void_ratio)


# ============================================================
# Packing search
# ============================================================


def feasible_candidate_positions(
    item: Item,
    placements: Sequence[Placement],
    basket: Tuple[int, int, int],
    base_heightmap: HeightMap,
    support_threshold: float,
) -> List[Tuple[Placement, float, HeightMap]]:
    candidates: List[Tuple[Placement, float, HeightMap]] = []
    _, _, bz = basket

    for rot in unique_rotations(item.size):
        w, d, h = rot
        wc, dc = base_heightmap.mm_to_cells_xy(w, d)
        if wc > base_heightmap.gx or dc > base_heightmap.gy:
            continue

        rotation_rpy = size_to_rpy(item.size, rot)

        for cy in range(base_heightmap.gy - dc + 1):
            for cx in range(base_heightmap.gx - wc + 1):
                stats = base_heightmap.region_stats(cx, cy, wc, dc)
                z = stats["base_h"]
                x, y = base_heightmap.cell_to_mm_xy(cx, cy)
                pos = (x, y, z)

                if not in_bounds(pos, rot, basket):
                    continue
                if z + h > bz:
                    continue

                cand = Placement(
                    object_index=item.object_index,
                    name=item.name,
                    original_size=item.size,
                    size=rot,
                    position=pos,
                    durability=item.durability,
                    support_ratio=0.0,
                    rotation_rpy=rotation_rpy,
                )

                collision = any(boxes_intersect(cand, p) for p in placements)
                if collision:
                    continue

                support = exact_support_ratio(pos, rot, placements)
                support = min(support, 1.0) if z > 0 else 1.0
                if support < support_threshold:
                    continue

                cand = Placement(
                    object_index=item.object_index,
                    name=item.name,
                    original_size=item.size,
                    size=rot,
                    position=pos,
                    durability=item.durability,
                    support_ratio=support,
                    rotation_rpy=rotation_rpy,
                )

                score = candidate_score(cand, placements, basket, base_heightmap)
                new_hm = base_heightmap.copy()
                new_hm.place(cx, cy, wc, dc, z + h)
                candidates.append((cand, score, new_hm))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:TOP_K_CANDIDATES_PER_ITEM]



# def ordered_items_for_packing(items: Sequence[Item]) -> List[Item]:
#     def key_fn(it: Item) -> Tuple[int, int, int, int]:
#         x, y, z = it.size
#         volume = x * y * z
#         base_area = x * y
#         return (-volume, -base_area, it.durability, -z)

#     return sorted(items, key=key_fn)
def ordered_items_for_packing(items: Sequence[Item]) -> List[Item]:
    def key_fn(it: Item):
        x, y, z = it.size
        volume = x * y * z
        base_area = x * y
        return (-it.durability, -base_area, -volume, -z)
    return sorted(items, key=key_fn)


def pack_subset(
    items: Sequence[Item],
    basket: Tuple[int, int, int],
    grid_size: int,
    support_threshold: float,
) -> Solution:
    placements: List[Placement] = []
    heightmap = HeightMap(basket, grid_size)

    for item in ordered_items_for_packing(items):
        candidates = feasible_candidate_positions(
            item=item,
            placements=placements,
            basket=basket,
            base_heightmap=heightmap,
            support_threshold=support_threshold,
        )

        if not candidates:
            continue

        best_placement, _, best_hm = candidates[0]
        placements.append(best_placement)
        heightmap = best_hm

    quality, fill_ratio, used_h, footprint_util, void_ratio = solution_quality(placements, basket, heightmap)
    return Solution(
        selected_items=list(items),
        placements=placements,
        score=quality,
        packed_count=len(placements),
        fill_ratio=fill_ratio,
        used_height=used_h,
        footprint_util=footprint_util,
        void_ratio=void_ratio,
    )



def find_best_selection(
    catalog: Sequence[Item],
    target_n: int,
    basket: Tuple[int, int, int],
    grid_size: int,
    support_threshold: float,
) -> Solution:
    if target_n <= 0:
        raise ValueError("target_n must be >= 1")
    if target_n > len(catalog):
        raise ValueError("target_n cannot exceed number of catalog items")

    best: Optional[Solution] = None

    for subset in itertools.combinations(catalog, target_n):
        sol = pack_subset(subset, basket, grid_size, support_threshold)
        if best is None:
            best = sol
            continue

        key_new = (sol.packed_count, sol.score, sol.fill_ratio, sol.footprint_util)
        key_best = (best.packed_count, best.score, best.fill_ratio, best.footprint_util)
        if key_new > key_best:
            best = sol

    assert best is not None
    return best


# ============================================================
# Output conversion for future ROS expansion
# ============================================================


def build_plan_output(placements):

    result = []

    for p in placements:

        x, y, z = p.position
        roll, pitch, yaw = size_to_rpy(p.original_size, p.size)

        pos = (float(x), float(y), float(z))
        rotation = (int(roll), int(pitch), int(yaw))

        result.append((
            int(p.object_index),
            pos,
            rotation
        ))

    return result

# ============================================================
# Visualization (keep for debugging)
# ============================================================


def cuboid_faces(position: Tuple[int, int, int], size: Tuple[int, int, int]):
    x, y, z = position
    w, d, h = size
    c = np.array([
        [x, y, z],
        [x + w, y, z],
        [x + w, y + d, z],
        [x, y + d, z],
        [x, y, z + h],
        [x + w, y, z + h],
        [x + w, y + d, z + h],
        [x, y + d, z + h],
    ], dtype=float)
    return [
        [c[i] for i in [0, 1, 2, 3]],
        [c[i] for i in [4, 5, 6, 7]],
        [c[i] for i in [0, 1, 5, 4]],
        [c[i] for i in [2, 3, 7, 6]],
        [c[i] for i in [1, 2, 6, 5]],
        [c[i] for i in [0, 3, 7, 4]],
    ]



def draw_basket_wireframe(ax, basket: Tuple[int, int, int]) -> None:
    bx, by, bz = basket
    pts = np.array([
        [0, 0, 0], [bx, 0, 0], [bx, by, 0], [0, by, 0],
        [0, 0, bz], [bx, 0, bz], [bx, by, bz], [0, by, bz],
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot(*zip(pts[a], pts[b]), linewidth=1.1)



def visualize_solution(sol: Solution, basket: Tuple[int, int, int]) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["cyan", "orange", "lime", "violet", "gold", "salmon", "deepskyblue", "pink"]

    for idx, p in enumerate(sol.placements):
        faces = cuboid_faces(p.position, p.size)
        poly = Poly3DCollection(faces, facecolors=colors[idx % len(colors)], edgecolors="k", alpha=0.72)
        ax.add_collection3d(poly)
        cx = p.position[0] + p.size[0] / 2.0
        cy = p.position[1] + p.size[1] / 2.0
        cz = p.position[2] + p.size[2] / 2.0
        label = f"{p.object_index}:{p.name}"
        ax.text(cx, cy, cz, label, fontsize=8)

    draw_basket_wireframe(ax, basket)
    bx, by, bz = basket
    ax.set_xlim(0, bx)
    ax.set_ylim(0, by)
    ax.set_zlim(0, bz)
    ax.set_box_aspect((bx, by, bz))
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Heightmap Packing Result")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main / debug print
# ============================================================


def print_solution(sol: Solution) -> None:
    print("=" * 80)
    print("Selected items:")
    for item in sol.selected_items:
        print(
            f"- idx={item.object_index:2d} | {item.name:8s} | "
            f"size={item.size} | durability={item.durability}"
        )

    print("\nPlacements:")
    for order, p in enumerate(sol.placements):
        print(
            f"[{order}] idx={p.object_index:2d} | {p.name:8s} | pos={p.position} | "
            f"rotated_size={p.size} | rpy={p.rotation_rpy} | support={p.support_ratio:.3f}"
        )

    print("\nPlan output:")
    print(build_plan_output(sol.placements))

    print("\nSummary:")
    print(f"- packed_count    : {sol.packed_count}/{len(sol.selected_items)}")
    print(f"- score           : {sol.score:.4f}")
    print(f"- fill_ratio      : {sol.fill_ratio:.4f}")
    print(f"- used_height(mm) : {sol.used_height}")
    print(f"- footprint_util  : {sol.footprint_util:.4f}")
    print(f"- void_ratio      : {sol.void_ratio:.4f}")
    print("=" * 80)



def main() -> None:
    catalog = [Item(object_index=i, **obj) for i, obj in enumerate(OBJECT_CATALOG)]
    best = find_best_selection(
        catalog=catalog,
        target_n=TARGET_N,
        basket=BASKET_SIZE,
        grid_size=GRID_SIZE,
        support_threshold=SUPPORT_THRESHOLD,
    )
    print_solution(best)
    visualize_solution(best, BASKET_SIZE)


if __name__ == "__main__":
    main()
