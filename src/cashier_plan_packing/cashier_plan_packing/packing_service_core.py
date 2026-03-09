import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from cashier_interfaces.msg import Placement as PlacementMsg

# =========================
# User settings
# =========================
CHECKPOINT_PATH = "/home/leeseungmin/Desktop/Doosan/rokey_ws/src/cashier_plan_packing/cashier_plan_packing/checkpoint_heightmap_attention_policy_candidate_split_v2.pth"
BASKET_SIZE = (170, 130, 75)
BASKET_WORLD_ORIGIN = (300.0, 400.0, 500.0)
GRID_SIZE = 5
MAX_OBJECTS = 8
SUPPORT_THRESHOLD = 0.72
HIDDEN_DIM = 256
CANDIDATE_TOP_K = 32
CANDIDATE_FEAT_DIM = 12
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 2
DROPOUT = 0.10
GREEDY = False
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Geometry / Environment
# =========================
def unique_rotations(size):
    return sorted(set(__import__("itertools").permutations(size, 3)))


def size_to_rpy(original, rotated):
    ox, oy, oz = original
    mapping = {
        (ox, oy, oz): (0, 0, 0),
        (ox, oz, oy): (90, 0, 0),
        (oy, ox, oz): (0, 0, 90),
        (oy, oz, ox): (0, 90, 0),
        (oz, ox, oy): (90, 90, 0),
        (oz, oy, ox): (0, 90, 90),
    }
    return mapping.get(tuple(rotated), (0, 0, 0))


def overlap_area_2d(ax, ay, aw, ad, bx, by, bw, bd):
    ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0, min(ay + ad, by + bd) - max(ay, by))
    return ox * oy


@dataclass(frozen=True)
class Placement:
    episode_index: int
    base_index: int
    name: str
    original_size: tuple
    size: tuple
    position: tuple
    durability: int
    support_ratio: float
    rotation_rpy: tuple


class HeightMap:
    def __init__(self, basket_size, grid_size):
        self.bx, self.by, self.bz = basket_size
        self.grid = int(grid_size)
        self.gx = self.bx // self.grid
        self.gy = self.by // self.grid
        self.heights = np.zeros((self.gy, self.gx), dtype=np.int32)

    def mm_to_cells_xy(self, w, d):
        return math.ceil(w / self.grid), math.ceil(d / self.grid)

    def cell_to_mm_xy(self, cx, cy):
        return cx * self.grid, cy * self.grid

    def region_stats(self, cx, cy, wc, dc):
        region = self.heights[cy:cy + dc, cx:cx + wc]
        base_h = int(region.max())
        support_cover = float(np.mean(region == base_h))
        roughness = float(region.max() - region.min())
        return {"base_h": base_h, "support_cover": support_cover, "roughness": roughness}

    def place(self, cx, cy, wc, dc, top_h):
        self.heights[cy:cy + dc, cx:cx + wc] = top_h


def boxes_intersect(p1, p2):
    x1, y1, z1 = p1.position
    w1, d1, h1 = p1.size
    x2, y2, z2 = p2.position
    w2, d2, h2 = p2.size
    overlap_x = (x1 < x2 + w2) and (x2 < x1 + w1)
    overlap_y = (y1 < y2 + d2) and (y2 < y1 + d1)
    overlap_z = (z1 < z2 + h2) and (z2 < z1 + h1)
    return overlap_x and overlap_y and overlap_z


def in_bounds(pos, size, basket):
    x, y, z = pos
    w, d, h = size
    bx, by, bz = basket
    return x >= 0 and y >= 0 and z >= 0 and (x + w) <= bx and (y + d) <= by and (z + h) <= bz


def total_item_volume(placements):
    return sum(p.size[0] * p.size[1] * p.size[2] for p in placements)


def used_height(placements):
    if not placements:
        return 0
    return max(p.position[2] + p.size[2] for p in placements)


def bounding_box_volume(placements):
    if not placements:
        return 0
    max_x = max(p.position[0] + p.size[0] for p in placements)
    max_y = max(p.position[1] + p.size[1] for p in placements)
    max_z = max(p.position[2] + p.size[2] for p in placements)
    return max_x * max_y * max_z


class PackingEnv:
    def __init__(self, basket_size, grid_size, support_threshold=0.70, max_objects=6, candidate_top_k=32):
        self.basket_size = basket_size
        self.grid_size = grid_size
        self.max_objects = max_objects
        self.support_threshold = support_threshold
        self.candidate_top_k = candidate_top_k
        self.heightmap = HeightMap(basket_size, grid_size)
        self.objects = []
        self.remaining = []
        self.placements = []
        self.cached_candidates = {}
        self.cached_ranked_candidates = {}

    def reset(self, objects, support_threshold=None):
        self.objects = [dict(obj) for obj in objects]
        self.remaining = list(range(len(objects)))
        self.placements = []
        self.heightmap = HeightMap(self.basket_size, self.grid_size)
        if support_threshold is not None:
            self.support_threshold = support_threshold
        self.cached_candidates = {}
        self.cached_ranked_candidates = {}
        self.prev_compact_fill = 0.0
        self.prev_height_ratio = 0.0
        self.prev_feasible_obj_ratio = 1.0
        return self.get_global_state()

    def exact_support_ratio(self, pos, size):
        x, y, z = pos
        w, d, _ = size
        if z == 0:
            return 1.0
        base_area = w * d
        support_area = 0
        for p in self.placements:
            px, py, pz = p.position
            pw, pd, ph = p.size
            if pz + ph != z:
                continue
            support_area += overlap_area_2d(x, y, w, d, px, py, pw, pd)
        return support_area / base_area if base_area > 0 else 0.0

    def durability_penalty(self, candidate):
        x, y, z = candidate.position
        w, d, _ = candidate.size
        pen = 0.0
        for p in self.placements:
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

    def validate_placement(self, pos, size):
        if not in_bounds(pos, size, self.basket_size):
            return False, 0.0, "out_of_bounds"

        x, y, z = pos
        w, d, h = size
        if z + h > self.basket_size[2]:
            return False, 0.0, "over_height"

        probe = Placement(
            episode_index=-1,
            base_index=-1,
            name="probe",
            original_size=size,
            size=size,
            position=pos,
            durability=1,
            support_ratio=0.0,
            rotation_rpy=(0, 0, 0),
        )

        if any(boxes_intersect(probe, p) for p in self.placements):
            return False, 0.0, "overlap"

        support = self.exact_support_ratio(pos, size)
        support = min(support, 1.0) if z > 0 else 1.0
        if support < self.support_threshold:
            return False, float(support), "insufficient_support"

        return True, float(support), "ok"

    def candidate_score(self, candidate):
        bx, by, bz = self.basket_size
        x, y, z = candidate.position
        w, d, h = candidate.size
        new_list = list(self.placements) + [candidate]

        space_efficiency = total_item_volume(new_list) / max(1, bounding_box_volume(new_list))
        current_top = used_height(self.placements)
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
        dura_pen = self.durability_penalty(candidate)

        base_area = w * d
        flat_bonus = 0.012 * (base_area / max(1, h))

        wc, dc = self.heightmap.mm_to_cells_xy(w, d)
        cx = x // self.heightmap.grid
        cy = y // self.heightmap.grid
        roughness = 0.0
        if cx + wc <= self.heightmap.gx and cy + dc <= self.heightmap.gy:
            roughness = self.heightmap.region_stats(cx, cy, wc, dc)["roughness"]

        low_position_factor = 1.0 - min(1.0, z / max(1.0, float(bz)))
        durability_strength = (candidate.durability - 1) / 4.0
        durability_layer_bonus = 2.5 * durability_strength * low_position_factor
        weak_bottom_penalty = 1.8 * (1.0 - durability_strength) * low_position_factor

        score = 0.0
        score += 12.0 * candidate.support_ratio
        score += 9.0 * space_efficiency
        score += wall_bonus
        score += flat_bonus
        score += durability_layer_bonus
        score -= 0.18 * z
        score -= 0.08 * height_increase
        score -= centered_penalty
        score -= top_gap_penalty
        score -= 0.04 * roughness
        score -= dura_pen
        score -= weak_bottom_penalty
        return float(score)

    def _build_candidates_for_object(self, obj_idx):
        if obj_idx in self.cached_candidates:
            return self.cached_candidates[obj_idx]

        obj = self.objects[obj_idx]
        candidates = []

        for rot in unique_rotations(obj["size"]):
            w, d, h = rot
            wc, dc = self.heightmap.mm_to_cells_xy(w, d)
            if wc > self.heightmap.gx or dc > self.heightmap.gy:
                continue

            rot_rpy = size_to_rpy(obj["size"], rot)

            for cy in range(self.heightmap.gy - dc + 1):
                for cx in range(self.heightmap.gx - wc + 1):
                    stats = self.heightmap.region_stats(cx, cy, wc, dc)
                    z = stats["base_h"]
                    x, y = self.heightmap.cell_to_mm_xy(cx, cy)
                    pos = (x, y, z)

                    is_valid, support, _ = self.validate_placement(pos, rot)
                    if not is_valid:
                        continue

                    candidate = Placement(
                        episode_index=obj["episode_index"],
                        base_index=obj["base_index"],
                        name=obj["name"],
                        original_size=obj["size"],
                        size=rot,
                        position=pos,
                        durability=obj["durability"],
                        support_ratio=support,
                        rotation_rpy=rot_rpy,
                    )

                    score = self.candidate_score(candidate)
                    new_list = list(self.placements) + [candidate]
                    space_efficiency = total_item_volume(new_list) / max(1, bounding_box_volume(new_list))

                    candidates.append({
                        "candidate": candidate,
                        "score": float(score),
                        "cx": int(cx),
                        "cy": int(cy),
                        "wc": int(wc),
                        "dc": int(dc),
                        "top_h": int(z + h),
                        "space_efficiency": float(space_efficiency),
                        "wall_contacts": int(
                            (x == 0) + (y == 0) +
                            (x + w == self.basket_size[0]) +
                            (y + d == self.basket_size[1])
                        ),
                    })

        self.cached_candidates[obj_idx] = candidates
        return candidates

    def _rank_candidates_for_object(self, obj_idx):
        if obj_idx in self.cached_ranked_candidates:
            return self.cached_ranked_candidates[obj_idx]

        candidates = list(self._build_candidates_for_object(obj_idx))
        if len(candidates) == 0:
            self.cached_ranked_candidates[obj_idx] = []
            return []

        candidates.sort(
            key=lambda c: (
                c["score"],
                c["candidate"].support_ratio,
                -c["candidate"].position[2],
                c["space_efficiency"],
            ),
            reverse=True,
        )

        ranked = candidates[: self.candidate_top_k]
        self.cached_ranked_candidates[obj_idx] = ranked
        return ranked

    def get_feasible_object_mask(self):
        mask = np.zeros(self.max_objects, dtype=np.float32)
        for i in range(min(len(self.objects), self.max_objects)):
            if i in self.remaining and len(self._rank_candidates_for_object(i)) > 0:
                mask[i] = 1.0
        return mask

    def get_candidate_mask(self, obj_idx):
        mask = np.zeros(self.candidate_top_k, dtype=np.float32)
        ranked = self._rank_candidates_for_object(obj_idx)
        for i in range(len(ranked)):
            mask[i] = 1.0
        return mask

    def get_candidate_feature_matrix(self, obj_idx):
        feat_dim = 12
        mat = np.zeros((self.candidate_top_k, feat_dim), dtype=np.float32)
        ranked = self._rank_candidates_for_object(obj_idx)
        if len(ranked) == 0:
            return mat

        scores = np.array([c["score"] for c in ranked], dtype=np.float32)
        smin, smax = float(scores.min()), float(scores.max())
        denom = max(1e-6, smax - smin)

        bx, by, bz = self.basket_size
        basket_floor = max(1.0, float(bx * by))
        basket_vol = max(1.0, float(bx * by * bz))

        for i, entry in enumerate(ranked):
            cand = entry["candidate"]
            x, y, z = cand.position
            w, d, h = cand.size
            score_norm = (entry["score"] - smin) / denom

            mat[i] = np.array([
                x / bx,
                y / by,
                z / bz,
                w / bx,
                d / by,
                h / bz,
                cand.support_ratio,
                score_norm,
                entry["space_efficiency"],
                entry["top_h"] / bz,
                (w * d) / basket_floor,
                (w * d * h) / basket_vol,
            ], dtype=np.float32)

        return mat

    def get_global_state(self):
        num_total = max(1, len(self.objects))
        placed = len(self.placements)
        max_h = used_height(self.placements)
        fill_ratio = total_item_volume(self.placements) / max(
            1, self.basket_size[0] * self.basket_size[1] * max(1, max_h)
        )
        remaining_ratio = len(self.remaining) / self.max_objects
        placed_ratio = placed / num_total
        height_ratio = max_h / self.basket_size[2]
        mean_height = float(self.heightmap.heights.mean()) / max(1.0, self.basket_size[2])
        std_height = float(self.heightmap.heights.std()) / max(1.0, self.basket_size[2])
        feasible_obj_ratio = float(self.get_feasible_object_mask().sum()) / self.max_objects

        return np.array([
            placed_ratio,
            remaining_ratio,
            fill_ratio,
            height_ratio,
            mean_height,
            std_height,
            feasible_obj_ratio,
            self.support_threshold,
        ], dtype=np.float32)

    def get_heightmap_tensor(self):
        return (self.heightmap.heights.astype(np.float32) / self.basket_size[2])[None, :, :]

    def get_object_feature_matrix(self):
        feat_dim = 10
        mat = np.zeros((self.max_objects, feat_dim), dtype=np.float32)
        bx, by, bz = self.basket_size

        for i in range(min(len(self.objects), self.max_objects)):
            obj = self.objects[i]
            sx, sy, sz = obj["size"]
            vol = sx * sy * sz
            base = sx * sy
            remaining_flag = 1.0 if i in self.remaining else 0.0
            feasible_flag = 1.0 if len(self._rank_candidates_for_object(i)) > 0 and i in self.remaining else 0.0

            mat[i] = np.array([
                sx / bx,
                sy / by,
                sz / bz,
                obj["durability"] / 5.0,
                vol / (bx * by * bz),
                base / (bx * by),
                max(sx, sy, sz) / max(self.basket_size),
                min(sx, sy, sz) / max(self.basket_size),
                remaining_flag,
                feasible_flag,
            ], dtype=np.float32)

        return mat

    def has_any_feasible_action(self):
        return bool(self.get_feasible_object_mask().sum() > 0)

    def step(self, obj_idx, candidate_idx):
        ranked = self._rank_candidates_for_object(obj_idx)
        candidate_idx = int(candidate_idx)

        if candidate_idx < 0 or candidate_idx >= len(ranked):
            return self.get_global_state(), -0.40, True, {
                "reason": "invalid_candidate_choice",
                "selector_reward": -0.20,
                "placement_reward": -0.20,
            }

        num_total = max(1, len(self.objects))
        prev_space_efficiency = float(self.prev_compact_fill)
        prev_height_ratio = float(self.prev_height_ratio)
        prev_feasible_obj_ratio = float(self.prev_feasible_obj_ratio)

        match = ranked[candidate_idx]
        candidate = match["candidate"]

        self.placements.append(candidate)
        if obj_idx in self.remaining:
            self.remaining.remove(obj_idx)

        self.heightmap.place(match["cx"], match["cy"], match["wc"], match["dc"], match["top_h"])
        self.cached_candidates = {}
        self.cached_ranked_candidates = {}

        next_state = self.get_global_state()
        done = (len(self.remaining) == 0) or (not self.has_any_feasible_action())

        total_vol = total_item_volume(self.placements)
        current_h = max(1, used_height(self.placements))
        space_efficiency = total_vol / max(1, self.basket_size[0] * self.basket_size[1] * current_h)
        height_ratio = current_h / self.basket_size[2]
        placed_ratio = len(self.placements) / num_total
        feasible_ratio = float(self.get_feasible_object_mask().sum()) / self.max_objects

        fill_gain = space_efficiency - prev_space_efficiency
        height_increase = max(0.0, height_ratio - prev_height_ratio)
        feasible_gain = feasible_ratio - prev_feasible_obj_ratio

        wall_bonus = 0.5 * float(candidate.position[0] == 0) + 0.5 * float(candidate.position[1] == 0)
        low_layer_bonus = (candidate.durability / 5.0) * (1.0 - candidate.position[2] / self.basket_size[2])
        score_norm = np.clip(match["score"] / 12.0, -1.0, 1.0)

        selector_reward = 0.0
        selector_reward += 0.55 * (1.0 / num_total)
        selector_reward += 0.35 * feasible_ratio
        selector_reward += 0.30 * feasible_gain
        selector_reward += 0.25 * space_efficiency

        placement_reward = 0.0
        placement_reward += 1.15 * fill_gain
        placement_reward += 0.24 * candidate.support_ratio
        placement_reward += 0.10 * low_layer_bonus
        placement_reward += 0.06 * wall_bonus
        placement_reward += 0.18 * score_norm
        placement_reward -= 0.26 * height_increase
        placement_reward -= 0.10 * (candidate.position[2] / self.basket_size[2])

        reward = selector_reward + placement_reward

        info = {
            "placement": candidate,
            "space_efficiency": space_efficiency,
            "used_height": current_h,
            "placed_count": len(self.placements),
            "total_count": len(self.objects),
            "placed_ratio": placed_ratio,
            "feasible_ratio": feasible_ratio,
            "selector_reward": selector_reward,
            "placement_reward": placement_reward,
        }

        if done:
            unplaced_ratio = 1.0 - placed_ratio
            all_placed_bonus = 0.35 if len(self.remaining) == 0 else 0.0
            dead_end_penalty = 0.30 * unplaced_ratio if len(self.remaining) > 0 else 0.0

            terminal_bonus = 0.0
            terminal_bonus += 2.10 * placed_ratio
            terminal_bonus += 1.80 * space_efficiency
            terminal_bonus -= 0.85 * unplaced_ratio
            terminal_bonus -= dead_end_penalty
            terminal_bonus += all_placed_bonus
            reward += terminal_bonus

            info["unplaced_ratio"] = unplaced_ratio
            info["terminal_bonus"] = terminal_bonus

        self.prev_compact_fill = space_efficiency
        self.prev_height_ratio = height_ratio
        self.prev_feasible_obj_ratio = feasible_ratio

        return next_state, float(reward), done, info


# =========================
# Networks
# =========================
class HeightMapEncoder(nn.Module):
    def __init__(self, gy, gx, hidden_dim, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        return self.proj(x)


class SetAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_vec, tokens, key_padding_mask=None):
        q = self.query_proj(query_vec).unsqueeze(1)
        kv = self.kv_norm(tokens)
        pooled, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = self.out_norm(pooled.squeeze(1) + query_vec)
        return pooled


class ObjectSelectorNetwork(nn.Module):
    def __init__(self, state_dim, object_feat_dim, gy, gx, max_objects, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.max_objects = max_objects
        self.map_encoder = HeightMapEncoder(gy, gx, hidden_dim, dropout=dropout)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.object_proj = nn.Sequential(
            nn.Linear(object_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pool = CrossAttentionPooling(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.logit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, heightmap, state_vec, object_feat):
        map_feat = self.map_encoder(heightmap)
        state_feat = self.state_proj(state_vec)
        query = map_feat + state_feat

        obj_tokens = self.object_proj(object_feat)
        key_padding_mask = torch.isclose(object_feat.abs().sum(dim=-1), torch.zeros(1, device=object_feat.device))
        for block in self.blocks:
            obj_tokens = block(obj_tokens, key_padding_mask=key_padding_mask)

        pooled_obj = self.pool(query, obj_tokens, key_padding_mask=key_padding_mask)
        context = torch.cat([map_feat, state_feat, pooled_obj], dim=-1)
        value = self.value_head(context).squeeze(-1)

        query_expand = query.unsqueeze(1).expand(-1, self.max_objects, -1)
        pooled_expand = pooled_obj.unsqueeze(1).expand(-1, self.max_objects, -1)
        logits = self.logit_head(torch.cat([query_expand, pooled_expand, obj_tokens], dim=-1)).squeeze(-1)
        return logits, value


class PlacementPolicyNetwork(nn.Module):
    def __init__(self, state_dim, object_feat_dim, candidate_feat_dim, gy, gx, top_k, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.map_encoder = HeightMapEncoder(gy, gx, hidden_dim, dropout=dropout)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.object_proj = nn.Sequential(
            nn.Linear(object_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pool = CrossAttentionPooling(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.logit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, heightmap, state_vec, object_feat, candidate_feat):
        map_feat = self.map_encoder(heightmap)
        state_feat = self.state_proj(state_vec)
        obj_feat = self.object_proj(object_feat)
        query = map_feat + state_feat + obj_feat

        cand_tokens = self.candidate_proj(candidate_feat)
        key_padding_mask = torch.isclose(candidate_feat.abs().sum(dim=-1), torch.zeros(1, device=candidate_feat.device))
        for block in self.blocks:
            cand_tokens = block(cand_tokens, key_padding_mask=key_padding_mask)

        pooled_cand = self.pool(query, cand_tokens, key_padding_mask=key_padding_mask)
        context = torch.cat([map_feat, obj_feat, pooled_cand], dim=-1)
        value = self.value_head(context).squeeze(-1)

        query_expand = query.unsqueeze(1).expand(-1, self.top_k, -1)
        pooled_expand = pooled_cand.unsqueeze(1).expand(-1, self.top_k, -1)
        logits = self.logit_head(torch.cat([query_expand, pooled_expand, cand_tokens], dim=-1)).squeeze(-1)
        return logits, value


def mask_logits(logits, mask):
    neg_large = -1e9 if logits.dtype != torch.float16 else -1e4
    return logits.masked_fill(mask < 0.5, neg_large)


# =========================
# Inference helpers
# =========================
def load_models(checkpoint_path):
    gx = BASKET_SIZE[0] // GRID_SIZE
    gy = BASKET_SIZE[1] // GRID_SIZE
    state_dim = 8
    object_feat_dim = 10

    order_model = ObjectSelectorNetwork(
        state_dim=state_dim,
        object_feat_dim=object_feat_dim,
        gy=gy,
        gx=gx,
        max_objects=MAX_OBJECTS,
        hidden_dim=HIDDEN_DIM,
        num_heads=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    placement_model = PlacementPolicyNetwork(
        state_dim=state_dim,
        object_feat_dim=object_feat_dim,
        candidate_feat_dim=CANDIDATE_FEAT_DIM,
        gy=gy,
        gx=gx,
        top_k=CANDIDATE_TOP_K,
        hidden_dim=HIDDEN_DIM,
        num_heads=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    order_model.load_state_dict(ckpt["order_model"])
    placement_model.load_state_dict(ckpt["placement_model"])
    order_model.eval()
    placement_model.eval()

    return order_model, placement_model


def get_order_distribution(order_model, env, state):
    hm_t = torch.tensor(env.get_heightmap_tensor(), dtype=torch.float32, device=device).unsqueeze(0)
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    obj_feat_t = torch.tensor(env.get_object_feature_matrix(), dtype=torch.float32, device=device).unsqueeze(0)
    order_mask_t = torch.tensor(env.get_feasible_object_mask(), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        order_logits, _ = order_model(hm_t, state_t, obj_feat_t)
        order_logits = mask_logits(order_logits, order_mask_t)
        probs = torch.softmax(order_logits, dim=-1).squeeze(0).detach().cpu().numpy()

    return probs


def get_placement_distribution(placement_model, env, state, obj_idx):
    hm_t = torch.tensor(env.get_heightmap_tensor(), dtype=torch.float32, device=device).unsqueeze(0)
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    obj_feat_t = torch.tensor(env.get_object_feature_matrix(), dtype=torch.float32, device=device).unsqueeze(0)
    chosen_feat = obj_feat_t[:, obj_idx, :]
    cand_feat_t = torch.tensor(env.get_candidate_feature_matrix(obj_idx), dtype=torch.float32, device=device).unsqueeze(0)
    cand_mask_t = torch.tensor(env.get_candidate_mask(obj_idx), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        place_logits, _ = placement_model(hm_t, state_t, chosen_feat, cand_feat_t)
        place_logits = mask_logits(place_logits, cand_mask_t)
        probs = torch.softmax(place_logits, dim=-1).squeeze(0).detach().cpu().numpy()

    return probs


def pick_index_from_probs(probs, greedy=False):
    valid_indices = np.where(probs > 0)[0]
    if len(valid_indices) == 0:
        return None
    if greedy:
        return int(np.argmax(probs))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def run_inference_once(order_model, placement_model, objects, support_threshold=0.70, greedy=True):
    env = PackingEnv(
        BASKET_SIZE,
        GRID_SIZE,
        support_threshold=support_threshold,
        max_objects=MAX_OBJECTS,
        candidate_top_k=CANDIDATE_TOP_K,
    )
    state = env.reset(objects, support_threshold=support_threshold)
    total_reward = 0.0

    while env.has_any_feasible_action():
        order_probs = get_order_distribution(order_model, env, state)
        obj_idx = pick_index_from_probs(order_probs, greedy=greedy)
        if obj_idx is None:
            break

        candidate_probs = get_placement_distribution(placement_model, env, state, obj_idx)
        cand_idx = pick_index_from_probs(candidate_probs, greedy=greedy)
        if cand_idx is None:
            break

        state, reward, done, _ = env.step(obj_idx, cand_idx)
        total_reward += reward

        if done:
            break

    return env.placements, total_reward


def objects_from_request_items(items):
    if len(items) > MAX_OBJECTS:
        raise ValueError(f"received {len(items)} items, but MAX_OBJECTS={MAX_OBJECTS}")

    objects = []

    for episode_index, item in enumerate(items):
        sx = int(item.width)
        sy = int(item.depth)
        sz = int(item.height)
        durability = int(item.durability)

        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError(
                f"invalid size at index {episode_index}: ({sx}, {sy}, {sz})"
            )

        objects.append({
            "name": item.name if item.name else f"item_{episode_index}",
            "item_id": item.item_id,
            "size": (sx, sy, sz),
            "durability": durability,
            "base_index": episode_index,
            "episode_index": episode_index,
        })

    return objects

def placements_to_response_msgs(placements):
    response_msgs = []

    bx, by, bz = BASKET_WORLD_ORIGIN

    for p in placements:
        msg = PlacementMsg()
        msg.object_index = int(p.base_index)

        sx, sy, sz = p.size  # 회전 적용된 실제 크기

        # 바구니 내부 기준 corner -> center
        local_cx = float(p.position[0] + sx / 2.0)
        local_cy = float(p.position[1] + sy / 2.0)
        local_cz = float(p.position[2] + sz / 2.0)

        # 바구니 상대좌표 -> 로봇/world 절대좌표
        msg.x = bx + local_cx
        msg.y = by - local_cy
        msg.z = bz + local_cz

        msg.roll = float(p.rotation_rpy[0])
        msg.pitch = float(p.rotation_rpy[1])
        msg.yaw = float(p.rotation_rpy[2])

        response_msgs.append(msg)

    return response_msgs
