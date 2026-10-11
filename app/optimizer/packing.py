"""3D packing into one shared cargo space, with LIFO access from the rear door.

The door is at x = 0. Later deliveries are placed first, toward the back and the
floor. An earlier delivery may sit in front of them, beside them, or on top of
them. A later carton must not sit above an earlier one, or between it and the door.
Inside one stop, identical cartons stay together and fill a row of equal height.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations

from app.domain.models import CustomerNode, LoadingRule, PackingUnit, Placement, Vehicle

_PERMS = {
    "LWH": (0, 1, 2),
    "WLH": (1, 0, 2),
    "LHW": (0, 2, 1),
    "HLW": (2, 0, 1),
    "WHL": (1, 2, 0),
    "HWL": (2, 1, 0),
}


@dataclass
class PackStop:
    store_id: str
    node_ids: list[str]
    units: list[PackingUnit]


@dataclass
class PackResult:
    ok: bool
    placements: list[Placement] = field(default_factory=list)
    failed_unit_ids: list[str] = field(default_factory=list)
    reason: str = "ok"


@dataclass
class _Box:
    unit: PackingUnit
    x: int
    y: int
    z: int
    dx: int
    dy: int
    dz: int
    rotation: str
    load_on_top_g: int = 0
    bearers: list[tuple[int, float]] = field(default_factory=list)


def boxes_overlap(a: Placement, b: Placement) -> bool:
    return (
        a.x < b.x + b.dx
        and b.x < a.x + a.dx
        and a.y < b.y + b.dy
        and b.y < a.y + a.dy
        and a.z < b.z + b.dz
        and b.z < a.z + a.dz
    )


def _orientations(unit: PackingUnit) -> list[tuple[str, int, int, int]]:
    dims = (unit.length_mm, unit.width_mm, unit.height_mm)
    names = unit.allowed_rotations or ["LWH"]
    if unit.upright_only:
        names = [name for name in names if name in ("LWH", "WLH")] or ["LWH"]
    found: list[tuple[str, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for name in names:
        perm = _PERMS.get(name)
        if perm is None:
            continue
        dx, dy, dz = dims[perm[0]], dims[perm[1]], dims[perm[2]]
        if (dx, dy, dz) in seen:
            continue
        seen.add((dx, dy, dz))
        found.append((name, dx, dy, dz))
    return found


def _overlaps_box(x: int, y: int, z: int, dx: int, dy: int, dz: int, boxes: list[_Box]) -> bool:
    for box in boxes:
        if (
            x < box.x + box.dx
            and box.x < x + dx
            and y < box.y + box.dy
            and box.y < y + dy
            and z < box.z + box.dz
            and box.z < z + dz
        ):
            return True
    return False


_PATH_BUDGET = 4096


def _aggregate_loads(boxes: list[_Box], index: int, added: int) -> dict[int, int]:
    """Total extra load per box if `added` lands on `index`, in O(edges).

    Bearers always sit below and were placed earlier, so their list indices are
    smaller; walking indices high-to-low accumulates every path's contribution
    before that node is visited. Used when the bearer DAG is too shared for
    per-path enumeration (tall mixed stacks), where the recursive version
    revisited shared bearers exponentially and blew the recursion limit.
    """
    pending = [0] * len(boxes)
    pending[index] = added
    for current in range(len(boxes) - 1, -1, -1):
        load = pending[current]
        if not load:
            continue
        for bearer_index, fraction in boxes[current].bearers:
            pending[bearer_index] += math.ceil(load * fraction)
    return {current: load for current, load in enumerate(pending) if load}


def _path_loads(boxes: list[_Box], index: int, added: int) -> dict[int, int] | None:
    """Per-path load spread, identical to the original recursive version.

    Returns None when the bearer DAG needs more than _PATH_BUDGET edge visits.
    """
    pending: dict[int, int] = {}
    stack = [(index, added)]
    visited = 0
    while stack:
        current, load = stack.pop()
        pending[current] = pending.get(current, 0) + load
        box = boxes[current]
        for bearer_index, fraction in box.bearers:
            stack.append((bearer_index, math.ceil(load * fraction)))
        visited += len(box.bearers) + 1
        if visited > _PATH_BUDGET:
            return None
    return pending


def _would_exceed(boxes: list[_Box], index: int, added: int) -> bool:
    # Per-path check (historical semantics): every propagation path is tested
    # on its own, exactly like the original recursion but without the call
    # stack. Deep shared-bearer DAGs fall back to the sound aggregate check.
    stack = [(index, added)]
    visited = 0
    while stack:
        current, load = stack.pop()
        box = boxes[current]
        if box.load_on_top_g + load > box.unit.max_stack_load_g:
            return True
        for bearer_index, fraction in box.bearers:
            stack.append((bearer_index, math.ceil(load * fraction)))
        visited += len(box.bearers) + 1
        if visited > _PATH_BUDGET:
            pending = _aggregate_loads(boxes, index, added)
            return any(
                boxes[current].load_on_top_g + load > boxes[current].unit.max_stack_load_g
                for current, load in pending.items()
            )
    return False


def _add_load(boxes: list[_Box], index: int, added: int) -> None:
    pending = _path_loads(boxes, index, added)
    if pending is None:
        pending = _aggregate_loads(boxes, index, added)
    for current, load in pending.items():
        boxes[current].load_on_top_g += load


def _support(
    x: int,
    y: int,
    z: int,
    dx: int,
    dy: int,
    unit: PackingUnit,
    boxes: list[_Box],
    rule: LoadingRule,
) -> tuple[bool, list[tuple[int, float]]]:
    if z == 0:
        return True, []
    base = dx * dy
    supporters: list[tuple[int, int]] = []
    supported = 0
    for index, box in enumerate(boxes):
        if box.z + box.dz != z:
            continue
        overlap_x = min(x + dx, box.x + box.dx) - max(x, box.x)
        overlap_y = min(y + dy, box.y + box.dy) - max(y, box.y)
        if overlap_x <= 0 or overlap_y <= 0:
            continue
        if box.unit.fragile or not box.unit.stackable:
            return False, []
        area = overlap_x * overlap_y
        supported += area
        supporters.append((index, area))
    if base <= 0 or supported < rule.min_support_ratio * base - 1e-6:
        return False, []
    bearers: list[tuple[int, float]] = []
    for index, area in supporters:
        share = math.ceil(unit.weight_g * area / supported)
        if _would_exceed(boxes, index, share):
            return False, []
        bearers.append((index, area / supported))
    return True, bearers


def _above_fragile(x: int, y: int, z: int, dx: int, dy: int, boxes: list[_Box]) -> bool:
    for box in boxes:
        if not box.unit.fragile or z < box.z + box.dz:
            continue
        overlap_x = min(x + dx, box.x + box.dx) - max(x, box.x)
        overlap_y = min(y + dy, box.y + box.dy) - max(y, box.y)
        if overlap_x > 0 and overlap_y > 0:
            return True
    return False


def _sorted_units(units: list[PackingUnit]) -> list[PackingUnit]:
    """One constructive order. Identical cartons stay together, so quantity does not add search branches."""
    return sorted(
        units,
        key=lambda unit: (
            1 if unit.fragile else 0,
            0 if unit.stackable else 1,
            -(unit.length_mm * unit.width_mm),
            -unit.weight_g,
            unit.sku_id,
            unit.height_mm,
            unit.unit_id,
        ),
    )


def _commit(
    unit: PackingUnit,
    x: int,
    y: int,
    z: int,
    shelf_h: int,
    length: int,
    width: int,
    height: int,
    boxes: list[_Box],
    rule: LoadingRule,
) -> _Box | None:
    best: tuple[tuple[int, int, int, str], _Box, list[tuple[int, float]]] | None = None
    for rotation, dx, dy, dz in _orientations(unit):
        if shelf_h and dz != shelf_h:
            continue
        if x + dx > length or y + dy > width or z + dz > height:
            continue
        if _overlaps_box(x, y, z, dx, dy, dz, boxes):
            continue
        if rule.fragile_no_stack and _above_fragile(x, y, z, dx, dy, boxes):
            continue
        supported, bearers = _support(x, y, z, dx, dy, unit, boxes, rule)
        if not supported:
            continue
        key = (0 if shelf_h else dz, dy, dx, rotation)
        if best is None or key < best[0]:
            best = (key, _Box(unit, x, y, z, dx, dy, dz, rotation), bearers)
    if best is None:
        return None
    _, placed, bearers = best
    placed.bearers = bearers
    for index, fraction in bearers:
        _add_load(boxes, index, math.ceil(unit.weight_g * fraction))
    boxes.append(placed)
    return placed


def _alternate_units(units: list[PackingUnit]) -> list[PackingUnit]:
    return sorted(
        units,
        key=lambda unit: (
            1 if unit.fragile else 0,
            0 if unit.stackable else 1,
            -unit.weight_g,
            -unit.height_mm,
            unit.sku_id,
            unit.unit_id,
        ),
    )


def _pack_walls(
    ordered: list[PackingUnit],
    length: int,
    width: int,
    height: int,
    rule: LoadingRule,
) -> tuple[list[_Box] | None, list[str]]:
    """Fill one store from a fixed order. Quantity adds placements, not search branches."""
    boxes: list[_Box] = []
    x = y = z = 0
    shelf_h = 0
    wall_d = 0
    index = 0
    guard = 0
    limit = len(ordered) * 8 + 1
    while index < len(ordered):
        guard += 1
        if guard > limit:
            return None, [unit.unit_id for unit in ordered[index:]]
        placed = _commit(ordered[index], x, y, z, shelf_h, length, width, height, boxes, rule)
        if placed is not None:
            y += placed.dy
            wall_d = max(wall_d, placed.dx)
            if shelf_h == 0:
                shelf_h = placed.dz
            index += 1
            continue
        if y > 0 and shelf_h > 0:
            y = 0
            z += shelf_h
            shelf_h = 0
            continue
        if z > 0 or wall_d > 0:
            x += max(wall_d, 1)
            y = 0
            z = 0
            shelf_h = 0
            wall_d = 0
            continue
        return None, [unit.unit_id for unit in ordered[index:]]
    return boxes, []


def _pack_block_orders(
    units: list[PackingUnit],
    max_length: int,
    width: int,
    height: int,
    rule: LoadingRule,
) -> list[list[_Box]]:
    """Every block the constructive orders produce, so attaching can try them all."""
    if not units:
        return [[]]
    for unit in units:
        if not any(dx <= max_length and dy <= width and dz <= height for _, dx, dy, dz in _orientations(unit)):
            return []
    volume = sum(unit.volume_mm3 for unit in units)
    if volume > max_length * width * height:
        return []
    primary = _sorted_units(units)
    orders = [primary]
    alternate = _alternate_units(units)
    if [unit.unit_id for unit in alternate] != [unit.unit_id for unit in primary]:
        orders.append(alternate)
    blocks: list[list[_Box]] = []
    for ordered in orders:
        boxes, _failed = _pack_walls(ordered, max_length, width, height, rule)
        if boxes is not None:
            blocks.append(boxes)
    return blocks


def _pack_block(
    units: list[PackingUnit],
    max_length: int,
    width: int,
    height: int,
    rule: LoadingRule,
) -> tuple[list[_Box] | None, int, list[str]]:
    if not units:
        return [], 0, []
    for unit in units:
        if not any(dx <= max_length and dy <= width and dz <= height for _, dx, dy, dz in _orientations(unit)):
            return None, 0, [item.unit_id for item in units]
    volume = sum(unit.volume_mm3 for unit in units)
    if volume > max_length * width * height:
        return None, 0, [unit.unit_id for unit in units]
    blocks = _pack_block_orders(units, max_length, width, height, rule)
    if not blocks:
        return None, 0, [unit.unit_id for unit in units]
    boxes = blocks[0]
    used = max((box.x + box.dx for box in boxes), default=0)
    return boxes, used, []


def later_blocks_earlier(later: Placement, earlier: Placement) -> bool:
    """True when the later delivery has to be moved before the earlier one can leave."""
    return _blocks(
        later.x,
        later.y,
        later.z,
        later.dx,
        later.dy,
        later.dz,
        earlier.x,
        earlier.y,
        earlier.z,
        earlier.dx,
        earlier.dy,
        earlier.dz,
    )


def _blocks(
    later_x: int,
    later_y: int,
    later_z: int,
    later_dx: int,
    later_dy: int,
    later_dz: int,
    x: int,
    y: int,
    z: int,
    dx: int,
    dy: int,
    dz: int,
) -> bool:
    overlaps_xy = x < later_x + later_dx and later_x < x + dx and y < later_y + later_dy and later_y < y + dy
    if overlaps_xy and later_z >= z + dz:
        return True
    overlaps_yz = y < later_y + later_dy and later_y < y + dy and z < later_z + later_dz and later_z < z + dz
    return overlaps_yz and later_x < x


def _shift_boxes(boxes: list[_Box], ox: int, oy: int, oz: int) -> list[_Box]:
    return [
        _Box(box.unit, box.x + ox, box.y + oy, box.z + oz, box.dx, box.dy, box.dz, box.rotation)
        for box in boxes
    ]


def _load_snapshot(boxes: list[_Box]) -> list[int]:
    return [box.load_on_top_g for box in boxes]


def _restore_loads(boxes: list[_Box], loads: list[int]) -> None:
    for box, load in zip(boxes, loads):
        box.load_on_top_g = load


def _link_boxes(shifted: list[_Box], existing: list[_Box], vehicle: Vehicle, rule: LoadingRule) -> bool:
    """Stack a translated group onto cargo already in the vehicle. Later cargo stays underneath or behind."""
    ground = list(existing)
    for box in sorted(shifted, key=lambda item: (item.z, item.x, item.y, item.unit.unit_id)):
        if (
            box.x < 0
            or box.y < 0
            or box.z < 0
            or box.x + box.dx > vehicle.length_mm
            or box.y + box.dy > vehicle.width_mm
            or box.z + box.dz > vehicle.height_mm
        ):
            return False
        if _overlaps_box(box.x, box.y, box.z, box.dx, box.dy, box.dz, ground):
            return False
        blocked = any(
            _blocks(later.x, later.y, later.z, later.dx, later.dy, later.dz, box.x, box.y, box.z, box.dx, box.dy, box.dz)
            for later in existing
        )
        if blocked:
            return False
        if rule.fragile_no_stack and _above_fragile(box.x, box.y, box.z, box.dx, box.dy, ground):
            return False
        supported, bearers = _support(box.x, box.y, box.z, box.dx, box.dy, box.unit, ground, rule)
        if not supported:
            return False
        box.bearers = bearers
        for index, fraction in bearers:
            _add_load(ground, index, math.ceil(box.unit.weight_g * fraction))
        ground.append(box)
    return True


def _try_attach(
    local: list[_Box],
    ox: int,
    oy: int,
    oz: int,
    existing: list[_Box],
    vehicle: Vehicle,
    rule: LoadingRule,
) -> list[_Box] | None:
    snapshot = _load_snapshot(existing)
    shifted = _shift_boxes(local, ox, oy, oz)
    if _link_boxes(shifted, existing, vehicle, rule):
        return shifted
    _restore_loads(existing, snapshot)
    return None


def _block_origins(
    span_x: int,
    span_y: int,
    span_z: int,
    existing: list[_Box],
    length: int,
    width: int,
    height: int,
) -> list[tuple[int, int, int]]:
    raw = [(length - span_x, 0, 0), (0, 0, 0)]
    for box in existing:
        raw.extend(
            (
                (box.x, box.y, box.z + box.dz),
                (box.x + box.dx - span_x, box.y, box.z + box.dz),
                (box.x, box.y + box.dy, box.z),
                (box.x, box.y + box.dy, 0),
                (box.x - span_x, box.y, 0),
                (box.x - span_x, 0, 0),
                (length - span_x, box.y + box.dy, 0),
                (0, box.y + box.dy, 0),
            )
        )
    origins: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for ox, oy, oz in raw:
        if (ox, oy, oz) in seen:
            continue
        seen.add((ox, oy, oz))
        if ox < 0 or oy < 0 or oz < 0:
            continue
        if ox + span_x > length or oy + span_y > width or oz + span_z > height:
            continue
        origins.append((ox, oy, oz))
    return origins


def _place_block(
    units: list[PackingUnit],
    existing: list[_Box],
    vehicle: Vehicle,
    rule: LoadingRule,
) -> list[_Box] | None:
    for local in _pack_block_orders(units, vehicle.length_mm, vehicle.width_mm, vehicle.height_mm, rule):
        if not local:
            continue
        span_x = max(box.x + box.dx for box in local)
        span_y = max(box.y + box.dy for box in local)
        span_z = max(box.z + box.dz for box in local)
        origins = _block_origins(span_x, span_y, span_z, existing, vehicle.length_mm, vehicle.width_mm, vehicle.height_mm)
        origins.sort(key=lambda item: (item[2], -item[0], item[1]))
        if len(units) > 40:
            origins = origins[:32]
        for ox, oy, oz in origins:
            shifted = _try_attach(local, ox, oy, oz, existing, vehicle, rule)
            if shifted is not None:
                return shifted
    return None


def _pose_candidates(
    dx: int,
    dy: int,
    dz: int,
    boxes: list[_Box],
    length: int,
    width: int,
    height: int,
) -> list[tuple[int, int, int]]:
    raw = [(0, 0, 0), (length - dx, 0, 0)]
    for box in boxes:
        raw.extend(
            (
                (box.x, box.y, box.z + box.dz),
                (box.x + box.dx - dx, box.y, box.z + box.dz),
                (box.x, box.y + box.dy - dy, box.z + box.dz),
                (box.x, box.y + box.dy, box.z),
                (box.x - dx, box.y, box.z),
                (box.x - dx, 0, 0),
                (length - dx, box.y + box.dy, 0),
                (box.x, box.y + box.dy, 0),
            )
        )
    found: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for x, y, z in raw:
        if (x, y, z) in seen or x < 0 or y < 0 or z < 0:
            continue
        seen.add((x, y, z))
        if x + dx > length or y + dy > width or z + dz > height:
            continue
        found.append((x, y, z))
    return found


def _commit_pose(
    unit: PackingUnit,
    existing: list[_Box],
    added: list[_Box],
    vehicle: Vehicle,
    rule: LoadingRule,
) -> _Box | None:
    """Best-key feasible pose. Scans in key order and stops at the first hit."""
    occupied = existing + added
    bottom_counts: Counter[int] | None = None
    if rule.align_levels:
        bottom_counts = Counter(box.z for box in occupied)
    candidates: list[tuple] = []
    for rotation, dx, dy, dz in _orientations(unit):
        for x, y, z in _pose_candidates(dx, dy, dz, occupied, vehicle.length_mm, vehicle.width_mm, vehicle.height_mm):
            key: list = []
            if rule.fragile_avoid_floor and unit.fragile:
                key.append(1 if z == 0 else 0)
            if bottom_counts is not None:
                key.append(-bottom_counts.get(z, 0))
            key.extend((z, -x, y, dz, dy, dx, rotation))
            candidates.append(tuple(key))
    candidates.sort()
    for *_, z, neg_x, y, dz, dy, dx, rotation in candidates:
        x = -neg_x
        if _overlaps_box(x, y, z, dx, dy, dz, occupied):
            continue
        blocked = any(
            _blocks(later.x, later.y, later.z, later.dx, later.dy, later.dz, x, y, z, dx, dy, dz)
            for later in existing
        )
        if blocked:
            continue
        if rule.fragile_no_stack and _above_fragile(x, y, z, dx, dy, occupied):
            continue
        supported, bearers = _support(x, y, z, dx, dy, unit, occupied, rule)
        if not supported:
            continue
        placed = _Box(unit, x, y, z, dx, dy, dz, rotation)
        placed.bearers = bearers
        for index, fraction in bearers:
            _add_load(occupied, index, math.ceil(unit.weight_g * fraction))
        return placed
    return None


def _place_items(
    units: list[PackingUnit],
    existing: list[_Box],
    vehicle: Vehicle,
    rule: LoadingRule,
) -> list[_Box] | None:
    primary = _sorted_units(units)
    orders = [primary]
    alternate = _alternate_units(units)
    if [unit.unit_id for unit in alternate] != [unit.unit_id for unit in primary]:
        orders.append(alternate)
    base = _load_snapshot(existing)
    for ordered in orders:
        _restore_loads(existing, base)
        added: list[_Box] = []
        for unit in ordered:
            placed = _commit_pose(unit, existing, added, vehicle, rule)
            if placed is None:
                added = []
                break
            added.append(placed)
        if len(added) == len(ordered):
            return added
    _restore_loads(existing, base)
    return None


def _place_in_front(
    units: list[PackingUnit],
    existing: list[_Box],
    vehicle: Vehicle,
    rule: LoadingRule,
) -> list[_Box] | None:
    if not existing:
        return None
    frontier = min(box.x for box in existing)
    local, _depth, _failed = _pack_block(units, frontier, vehicle.width_mm, vehicle.height_mm, rule)
    if not local:
        return None
    return _try_attach(local, 0, 0, 0, existing, vehicle, rule)


def _place_stop(
    units: list[PackingUnit],
    existing: list[_Box],
    vehicle: Vehicle,
    rule: LoadingRule,
) -> list[_Box] | None:
    if not existing:
        local, depth, _failed = _pack_block(units, vehicle.length_mm, vehicle.width_mm, vehicle.height_mm, rule)
        if local is not None:
            return _shift_boxes(local, vehicle.length_mm - depth, 0, 0)
        if len(units) <= 40:
            return _place_items(units, existing, vehicle, rule)
        return None
    placed = _place_block(units, existing, vehicle, rule)
    if placed is not None:
        return placed
    if len(units) <= 40:
        placed = _place_items(units, existing, vehicle, rule)
        if placed is not None:
            return placed
    return _place_in_front(units, existing, vehicle, rule)


def _commit_perch(
    unit: PackingUnit,
    occupied: list[_Box],
    vehicle: Vehicle,
    rule: LoadingRule,
) -> _Box | None:
    """First feasible pose that sits on another carton, never on the floor."""
    length, width, height = vehicle.length_mm, vehicle.width_mm, vehicle.height_mm
    platforms = [
        box
        for box in occupied
        if box.unit.stackable and not box.unit.fragile and box.unit.store_id == unit.store_id
    ]
    platforms.sort(key=lambda box: (-(box.z + box.dz), -(box.dx * box.dy), box.unit.unit_id))
    platforms = platforms[:16]
    for rotation, dx, dy, dz in _orientations(unit):
        for bearer in platforms:
            z = bearer.z + bearer.dz
            if z <= 0 or z + dz > height:
                continue
            for x, y in (
                (bearer.x, bearer.y),
                (bearer.x + bearer.dx - dx, bearer.y),
                (bearer.x, bearer.y + bearer.dy - dy),
                (0, bearer.y),
                (bearer.x, 0),
            ):
                if x < 0 or y < 0 or x + dx > length or y + dy > width:
                    continue
                if _overlaps_box(x, y, z, dx, dy, dz, occupied):
                    continue
                if any(
                    _blocks(later.x, later.y, later.z, later.dx, later.dy, later.dz, x, y, z, dx, dy, dz)
                    for later in occupied
                ):
                    continue
                if rule.fragile_no_stack and _above_fragile(x, y, z, dx, dy, occupied):
                    continue
                supported, bearers = _support(x, y, z, dx, dy, unit, occupied, rule)
                if not supported:
                    continue
                placed = _Box(unit, x, y, z, dx, dy, dz, rotation)
                placed.bearers = bearers
                for index, fraction in bearers:
                    _add_load(occupied, index, math.ceil(unit.weight_g * fraction))
                return placed
    return None


def _lift_fragile(accepted: list[_Box], vehicle: Vehicle, rule: LoadingRule) -> None:
    """Re-place fragile floor boxes on perches above other cargo.

    Walls are floor-standing strips, so a wall of fragile units (sorted last)
    puts fragile cartons at z=0, which extra_handling still prices when a store
    mixes fragile and sturdy cartons. A lift only tries
    poses on existing tops, so it stays cheap on large loads. A box that finds
    no perch keeps its original slot.
    """
    fragile_on_floor = sorted(
        (box for box in accepted if box.unit.fragile and box.z == 0),
        key=lambda box: box.unit.unit_id,
    )
    for box in fragile_on_floor:
        others = [item for item in accepted if item is not box]
        snapshot = _load_snapshot(others)
        placed = _commit_perch(box.unit, others, vehicle, rule)
        if placed is None:
            _restore_loads(others, snapshot)
            continue
        accepted.remove(box)
        accepted.append(placed)


def _as_placement(vehicle: Vehicle, box: _Box) -> Placement:
    return Placement(
        unit_id=box.unit.unit_id,
        vehicle_id=vehicle.id,
        store_id=box.unit.store_id,
        order_id=box.unit.order_id,
        x=box.x,
        y=box.y,
        z=box.z,
        dx=box.dx,
        dy=box.dy,
        dz=box.dz,
        rotation=box.rotation,
        fragile=box.unit.fragile,
        stackable=box.unit.stackable,
        max_stack_load_g=box.unit.max_stack_load_g,
        weight_g=box.unit.weight_g,
        load_on_top_g=box.load_on_top_g,
    )


_PACK_CACHE: dict[tuple, PackResult] = {}
_PACK_CACHE_LIMIT = 2048


def clear_pack_cache() -> None:
    _PACK_CACHE.clear()


def _pack_cache_key(vehicle: Vehicle, stops: list[PackStop], rule: LoadingRule) -> tuple:
    return (
        vehicle.id,
        vehicle.length_mm,
        vehicle.width_mm,
        vehicle.height_mm,
        vehicle.max_weight_g,
        rule.min_support_ratio,
        rule.max_overhang_mm,
        rule.fragile_no_stack,
        rule.fragile_avoid_floor,
        rule.align_levels,
        tuple(
            (
                stop.store_id,
                tuple(
                    (
                        unit.unit_id,
                        unit.length_mm,
                        unit.width_mm,
                        unit.height_mm,
                        unit.weight_g,
                        unit.fragile,
                        unit.stackable,
                        unit.max_stack_load_g,
                        unit.upright_only,
                        tuple(unit.allowed_rotations),
                    )
                    for unit in stop.units
                ),
            )
            for stop in stops
        ),
    )


def pack_stops(vehicle: Vehicle, stops: list[PackStop], rule: LoadingRule) -> PackResult:
    key = _pack_cache_key(vehicle, stops, rule)
    cached = _PACK_CACHE.get(key)
    if cached is not None:
        return cached
    result = _pack_stops(vehicle, stops, rule)
    if len(_PACK_CACHE) >= _PACK_CACHE_LIMIT:
        _PACK_CACHE.clear()
    _PACK_CACHE[key] = result
    return result


def _pack_stops(vehicle: Vehicle, stops: list[PackStop], rule: LoadingRule) -> PackResult:
    units = [unit for stop in stops for unit in stop.units]
    weight = sum(unit.weight_g for unit in units)
    volume = sum(unit.volume_mm3 for unit in units)
    if weight > vehicle.max_weight_g or volume > vehicle.volume_mm3:
        return PackResult(ok=False, failed_unit_ids=[unit.unit_id for unit in units], reason="capacity")
    accepted: list[_Box] = []
    placed: list[Placement] = []
    for stop in reversed(stops):
        if not stop.units:
            continue
        added = _place_stop(stop.units, accepted, vehicle, rule)
        if added is None:
            return PackResult(
                ok=False,
                placements=placed,
                failed_unit_ids=[unit.unit_id for unit in stop.units],
                reason="geometry",
            )
        accepted.extend(added)
        placed.extend(_as_placement(vehicle, box) for box in added)
    if rule.fragile_avoid_floor:
        load_snap = _load_snapshot(accepted)
        boxes_snap = list(accepted)
        _lift_fragile(accepted, vehicle, rule)
        lifted = [_as_placement(vehicle, box) for box in accepted]
        if _delivery_blocked(stops, lifted):
            accepted[:] = boxes_snap
            _restore_loads(accepted, load_snap)
        else:
            placed = lifted
    if _delivery_blocked(stops, placed):
        return PackResult(
            ok=False,
            placements=placed,
            failed_unit_ids=[unit.unit_id for stop in stops for unit in stop.units],
            reason="geometry",
        )
    return PackResult(ok=True, placements=placed)


def _delivery_blocked(stops: list[PackStop], placements: list[Placement]) -> bool:
    """True when a later stop's carton blocks an earlier stop at unload."""
    by_unit = {item.unit_id: item for item in placements}
    grouped: list[list[Placement]] = []
    for stop in stops:
        grouped.append([by_unit[unit.unit_id] for unit in stop.units if unit.unit_id in by_unit])
    for earlier_index, earlier_boxes in enumerate(grouped):
        for later_boxes in grouped[earlier_index + 1 :]:
            for earlier in earlier_boxes:
                for later in later_boxes:
                    if later_blocks_earlier(later, earlier):
                        return True
    return False


def stops_from_nodes(nodes: list[CustomerNode], units_by_id: dict[str, PackingUnit]) -> list[PackStop]:
    merged: list[PackStop] = []
    for node in nodes:
        units = [units_by_id[unit_id] for unit_id in node.unit_ids]
        if merged and merged[-1].store_id == node.store_id:
            merged[-1].node_ids.append(node.node_id)
            merged[-1].units.extend(units)
            continue
        merged.append(PackStop(store_id=node.store_id, node_ids=[node.node_id], units=units))
    return merged


def can_pack_nodes(
    nodes: list[CustomerNode],
    vehicle: Vehicle,
    rule: LoadingRule,
    units_by_id: dict[str, PackingUnit],
) -> bool:
    if not nodes:
        return True
    unit_count = sum(len(node.unit_ids) for node in nodes)
    orders = [nodes]
    if unit_count <= 40 and 1 < len(nodes) <= 4:
        orders = [list(order) for order in permutations(nodes)]
    elif len(nodes) > 1:
        orders = [nodes, list(reversed(nodes))]
    for order in orders:
        if pack_stops(vehicle, stops_from_nodes(order, units_by_id), rule).ok:
            return True
    return False


def minimal_infeasible_nodes(
    nodes: list[CustomerNode],
    vehicle: Vehicle,
    rule: LoadingRule,
    units_by_id: dict[str, PackingUnit],
) -> list[CustomerNode]:
    current = list(nodes)
    if can_pack_nodes(current, vehicle, rule, units_by_id):
        return []
    changed = True
    while changed and len(current) > 1:
        changed = False
        for index in range(len(current)):
            trial = current[:index] + current[index + 1 :]
            if not can_pack_nodes(trial, vehicle, rule, units_by_id):
                current = trial
                changed = True
                break
    return current
