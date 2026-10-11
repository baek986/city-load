"""Scaled objective shared by the route master and the packed incumbent."""

from __future__ import annotations

from collections import defaultdict

from app.domain.matrix import TravelMatrix
from app.domain.models import Dataset, PackingUnit, Placement, RoutePlan, Scales, Weights


def build_scales(dataset: Dataset, matrix: TravelMatrix) -> Scales:
    cheapest = min(vehicle.cost_per_km for vehicle in dataset.vehicles)
    cheapest_fixed = min(vehicle.fixed_cost for vehicle in dataset.vehicles)
    cost = cheapest_fixed
    travel = 0
    for store in dataset.stores:
        out_arc = matrix.get(dataset.depot.id, store.id)
        back_arc = matrix.get(store.id, dataset.depot.id)
        cost += (out_arc.distance_m + back_arc.distance_m) / 1000.0 * cheapest
        travel += out_arc.travel_time_s + back_arc.travel_time_s
    return Scales(
        cost=max(cost, 1.0),
        time=max(float(travel), 1.0),
        late=max(len(dataset.stores) * 60.0, 1.0),
        handling=max(float(len(dataset.stores)), 1.0),
        split=float(max(len(dataset.orders), 1)),
    )


def split_order_count(routes: list[RoutePlan], unit_orders: dict[str, str]) -> int:
    vehicles_by_order: dict[str, set[str]] = defaultdict(set)
    for route in routes:
        for stop in route.stops:
            for unit_id in stop.unit_ids:
                order_id = unit_orders.get(unit_id)
                if order_id is not None:
                    vehicles_by_order[order_id].add(route.vehicle_id)
    return sum(1 for vehicle_ids in vehicles_by_order.values() if len(vehicle_ids) > 1)


LAYER_MM = 50


def extra_layers_lower_bound(
    footprint_mm2: int,
    floor_mm2: int,
    volume_mm3: int = 0,
    layer_height_mm: int = 0,
) -> int:
    """Minimum extra layers: floor overflow, or volume overflow at max box height."""
    by_floor = 0
    if footprint_mm2 > 0 and floor_mm2 > 0:
        leftover = max(0, footprint_mm2 - floor_mm2)
        if leftover:
            by_floor = (leftover + floor_mm2 - 1) // floor_mm2
    by_volume = 0
    if volume_mm3 > 0 and floor_mm2 > 0 and layer_height_mm > 0:
        layer_vol = floor_mm2 * layer_height_mm
        leftover = max(0, volume_mm3 - layer_vol)
        if leftover:
            by_volume = (leftover + layer_vol - 1) // layer_vol
    return max(by_floor, by_volume)


def extra_layers_lower_bound_for_fleet(
    footprint_mm2: int,
    vehicles: list,
    volume_mm3: int = 0,
    layer_height_mm: int = 0,
) -> int:
    """Minimum extra layers if cargo may be spread across the given vehicles."""
    if not vehicles:
        return 0
    floors = [max(vehicle.length_mm * vehicle.width_mm, 1) for vehicle in vehicles]
    leftover_fp = max(0, footprint_mm2 - sum(floors))
    by_floor = 0 if leftover_fp == 0 else (leftover_fp + max(floors) - 1) // max(floors)
    by_volume = 0
    if volume_mm3 > 0 and layer_height_mm > 0:
        layer_vols = [floor * layer_height_mm for floor in floors]
        leftover_vol = max(0, volume_mm3 - sum(layer_vols))
        if leftover_vol:
            by_volume = (leftover_vol + max(layer_vols) - 1) // max(layer_vols)
    return max(by_floor, by_volume)


def extra_handling(placements: list[Placement]) -> int:
    """Geometric layers above the floor, and mixed-store fragile cartons left on the floor.

    Nearby z coordinates count as one layer (LAYER_MM bins) so a jagged wall of
    mixed SKU heights is not priced as dozens of extra handling steps.
    """
    by_vehicle: dict[str, list[Placement]] = defaultdict(list)
    for placement in placements:
        by_vehicle[placement.vehicle_id].append(placement)
    extra = 0
    for group in by_vehicle.values():
        extra += max(0, len({item.z // LAYER_MM for item in group}) - 1)
        by_store: dict[str, list[Placement]] = defaultdict(list)
        for placement in group:
            by_store[placement.store_id].append(placement)
        for items in by_store.values():
            if not any(not item.fragile for item in items):
                continue
            extra += sum(1 for item in items if item.fragile and item.z == 0)
    return extra


def score_routes(
    routes: list[RoutePlan],
    placements: list[Placement],
    split_count: int,
    weights: Weights,
    scales: Scales,
    include_extra: bool = True,
) -> float:
    cost = sum(route.travel_cost + route.fixed_cost for route in routes)
    travel_time = sum(route.travel_time_s for route in routes)
    late = sum(stop.late_minutes for route in routes for stop in route.stops)
    stops = sum(len(route.stops) for route in routes)
    handling = stops + (extra_handling(placements) if include_extra else 0)
    return (
        weights.cost * cost / scales.cost
        + weights.time * travel_time / scales.time
        + weights.late * late / scales.late
        + weights.handling * handling / scales.handling
        + weights.split * split_count / scales.split
    )


def fast_lower_bound(
    nodes,
    dataset: Dataset,
    matrix: TravelMatrix,
    weights: Weights,
    scales: Scales,
    units_by_id: dict[str, PackingUnit] | None = None,
) -> float:
    """A valid joint-score lower bound that does not run the route solver.

    Each stop is entered once, each used vehicle pays its fixed cost and a return
    to the depot, and every stop counts toward handling. Extra layers required by
    floor area or cargo height vs. volume are added for the chosen fleet.
    Lateness and splits are zero. The bound is weak, and it is available before
    CP-SAT finds its own.

    When units_by_id is provided, single-vehicle subsets that pass the weight and
    volume check are also tested for packing feasibility. A vehicle whose cargo
    bay cannot hold every stop (LIFO, support, fragile rules) is excluded from
    the fleet selection, so the bound no longer assumes a cheap vehicle the
    packer would reject.
    """
    if not nodes or not dataset.vehicles:
        return 0.0
    total_weight = sum(node.weight_g for node in nodes)
    total_volume = sum(node.volume_mm3 for node in nodes)
    total_footprint = sum(getattr(node, "footprint_mm2", 0) or 0 for node in nodes)
    vehicles = list(dataset.vehicles)
    packable_single: dict[str, bool] | None = None
    if units_by_id is not None and 2 <= len(nodes) <= 12:
        from app.optimizer.packing import can_pack_nodes

        # Identical vehicles share one packing check; only one representative per
        # cargo bay signature is tested, so 8 vans + 8 trucks cost 2 packer calls
        # instead of 16. The constructive packer scales with box count, so this
        # check is gated to small instances where it is fast and the bound is
        # used to prove optimality; large instances rely on the CP-SAT master.
        by_signature: dict[tuple, list[str]] = {}
        for vehicle in vehicles:
            signature = (vehicle.length_mm, vehicle.width_mm, vehicle.height_mm, vehicle.max_weight_g)
            by_signature.setdefault(signature, []).append(vehicle.id)
        packable_single = {}
        for signature, ids in by_signature.items():
            representative = next(v for v in vehicles if v.id == ids[0])
            if representative.max_weight_g < total_weight or representative.volume_mm3 < total_volume:
                verdict = False
            else:
                verdict = can_pack_nodes(nodes, representative, dataset.loading_rule, units_by_id)
            for vehicle_id in ids:
                packable_single[vehicle_id] = verdict

    def arc_score(origin_id: str, dest_id: str, node) -> float | None:
        arc = matrix.get(origin_id, dest_id)
        best: float | None = None
        for vehicle in vehicles:
            if node.weight_g > vehicle.max_weight_g or node.volume_mm3 > vehicle.volume_mm3:
                continue
            value = (
                weights.cost * (arc.distance_m / 1000.0 * vehicle.cost_per_km) / scales.cost
                + weights.time * arc.travel_time_s / scales.time
            )
            if best is None or value < best:
                best = value
        return best

    inbound_total = 0.0
    returns: list[float] = []
    origins = [dataset.depot.id, *[node.store_id for node in nodes]]
    for node in nodes:
        inbound: float | None = None
        for origin_id in origins:
            if origin_id == node.store_id:
                continue
            value = arc_score(origin_id, node.store_id, node)
            if value is None:
                continue
            if inbound is None or value < inbound:
                inbound = value
        inbound_total += inbound or 0.0
        back = arc_score(node.store_id, dataset.depot.id, node)
        if back is not None:
            returns.append(back)
    returns.sort()
    mst = _mst_travel(nodes, dataset, matrix, weights, scales)
    stop_handling = weights.handling * len(nodes) / scales.handling

    best_lb: float | None = None
    for mask in range(1, 1 << len(vehicles)):
        chosen = [vehicle for index, vehicle in enumerate(vehicles) if mask & (1 << index)]
        if sum(vehicle.max_weight_g for vehicle in chosen) < total_weight:
            continue
        if sum(vehicle.volume_mm3 for vehicle in chosen) < total_volume:
            continue
        if any(
            not any(node.weight_g <= vehicle.max_weight_g and node.volume_mm3 <= vehicle.volume_mm3 for vehicle in chosen)
            for node in nodes
        ):
            continue
        if packable_single is not None and len(chosen) == 1:
            if not packable_single.get(chosen[0].id, True):
                continue
        layer_h = max((getattr(node, "max_extent_mm", 0) or 0) for node in nodes)
        extra = extra_layers_lower_bound_for_fleet(total_footprint, chosen, total_volume, layer_h)
        radial = inbound_total + sum(returns[: len(chosen)])
        travel = max(radial, mst)
        cost = weights.cost * sum(vehicle.fixed_cost for vehicle in chosen) / scales.cost
        handling = stop_handling + weights.handling * extra / scales.handling
        value = cost + travel + handling
        if best_lb is None or value < best_lb:
            best_lb = value
    return best_lb or 0.0


def _mst_travel(nodes, dataset: Dataset, matrix: TravelMatrix, weights: Weights, scales: Scales) -> float:
    """Cheapest-vehicle spanning tree. A fleet's tours cost at least this much to connect."""
    points = [dataset.depot.id, *[node.store_id for node in nodes]]
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges: list[tuple[float, int, int]] = []
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            arc = matrix.get(points[left], points[right])
            weight = min(
                weights.cost * (arc.distance_m / 1000.0 * vehicle.cost_per_km) / scales.cost
                + weights.time * arc.travel_time_s / scales.time
                for vehicle in dataset.vehicles
            )
            edges.append((weight, left, right))
    edges.sort()
    total = 0.0
    linked = 0
    for weight, left, right in edges:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        parent[left_root] = right_root
        total += weight
        linked += 1
        if linked == len(points) - 1:
            break
    return total


def on_time_rate(routes: list[RoutePlan]) -> float:
    stops = [stop for route in routes for stop in route.stops]
    if not stops:
        return 1.0
    return sum(1 for stop in stops if stop.late_minutes <= 1e-9) / len(stops)
