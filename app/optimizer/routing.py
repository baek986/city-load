"""Feasible route construction and schedule evaluation.

Clarke-Wright savings proposes merges. Cheapest insertion assigns the fixed fleet.
Both are only an initial upper bound; the CP-SAT master proves the route cost.
When cargo shapes and river banks are provided, merges also prefer stores whose
boxes tessellate and stores that share a bank of the river.
"""

from __future__ import annotations

from app.domain.matrix import TravelMatrix
from app.domain.models import CustomerNode, Dataset, Location, RoutePlan, StopPlan, Vehicle
from app.optimizer.shape import CargoShape, combine_shapes, complementarity, route_bank


def _capacity_ok(nodes: list[CustomerNode], vehicle: Vehicle) -> bool:
    weight = sum(node.weight_g for node in nodes)
    volume = sum(node.volume_mm3 for node in nodes)
    return weight <= vehicle.max_weight_g and volume <= vehicle.volume_mm3


def _simulate(
    nodes: list[CustomerNode],
    vehicle: Vehicle,
    matrix: TravelMatrix,
    depot: Location,
) -> RoutePlan | None:
    if not nodes:
        return RoutePlan(
            vehicle_id=vehicle.id,
            stops=[],
            distance_m=0,
            travel_time_s=0,
            travel_cost=0.0,
            fixed_cost=0.0,
            depart_depot_s=depot.window_open_s,
            return_depot_s=depot.window_open_s,
            total_weight_g=0,
            total_volume_mm3=0,
        )
    time = depot.window_open_s
    previous = depot.id
    distance = 0
    travel_time = 0
    stops: list[StopPlan] = []
    for sequence, node in enumerate(nodes, start=1):
        arc = matrix.get(previous, node.store_id)
        time += arc.travel_time_s
        distance += arc.distance_m
        travel_time += arc.travel_time_s
        if time < node.window_open_s:
            time = node.window_open_s
        if time > node.window_close_s:
            return None
        late_seconds = 0
        if node.soft_deadline_s is not None and time > node.soft_deadline_s:
            late_seconds = time - node.soft_deadline_s
        arrival = time
        time += node.service_time_s
        stops.append(
            StopPlan(
                store_id=node.store_id,
                node_ids=[node.node_id],
                unit_ids=list(node.unit_ids),
                arrival_s=arrival,
                depart_s=time,
                late_minutes=late_seconds / 60.0,
                sequence=sequence,
            )
        )
        previous = node.store_id
    back = matrix.get(previous, depot.id)
    time += back.travel_time_s
    distance += back.distance_m
    travel_time += back.travel_time_s
    if time > depot.window_close_s:
        return None
    return RoutePlan(
        vehicle_id=vehicle.id,
        stops=stops,
        distance_m=distance,
        travel_time_s=travel_time,
        travel_cost=distance / 1000.0 * vehicle.cost_per_km,
        fixed_cost=vehicle.fixed_cost,
        depart_depot_s=depot.window_open_s,
        return_depot_s=time,
        total_weight_g=sum(node.weight_g for node in nodes),
        total_volume_mm3=sum(node.volume_mm3 for node in nodes),
    )


def route_distance(nodes: list[CustomerNode], matrix: TravelMatrix, depot_id: str) -> int:
    if not nodes:
        return 0
    previous = depot_id
    distance = 0
    for node in nodes:
        distance += matrix.get(previous, node.store_id).distance_m
        previous = node.store_id
    distance += matrix.get(previous, depot_id).distance_m
    return distance


def _geo_scale_m(nodes: list[CustomerNode], dataset: Dataset, matrix: TravelMatrix) -> float:
    if not nodes:
        return 1.0
    radial = sum(matrix.get(dataset.depot.id, node.store_id).distance_m for node in nodes) / len(nodes)
    return max(radial, 1.0)


def _insertion_cost(
    nodes: list[CustomerNode],
    position: int,
    node: CustomerNode,
    vehicle: Vehicle,
    matrix: TravelMatrix,
    depot: Location,
    shapes: dict[str, CargoShape] | None = None,
    banks: dict[str, str] | None = None,
    shape_scale: float = 0.0,
    bank_scale: float = 0.0,
) -> float | None:
    trial = nodes[:position] + [node] + nodes[position:]
    if not _capacity_ok(trial, vehicle):
        return None
    plan = _simulate(trial, vehicle, matrix, depot)
    if plan is None:
        return None
    cost = plan.travel_cost + plan.fixed_cost
    if shapes is not None and node.node_id in shapes and nodes:
        current_shape = combine_shapes([shapes[item.node_id] for item in nodes if item.node_id in shapes])
        cost -= complementarity(shapes[node.node_id], current_shape) * (shape_scale / 1000.0) * vehicle.cost_per_km
    if banks is not None and nodes:
        current_bank = route_bank(nodes, banks)
        node_bank = banks.get(node.node_id)
        if current_bank is not None and node_bank is not None and node_bank != current_bank:
            cost += (bank_scale / 1000.0) * vehicle.cost_per_km
    return cost


def cheapest_insertion(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    *,
    shapes: dict[str, CargoShape] | None = None,
    banks: dict[str, str] | None = None,
    vehicle_ids: set[str] | None = None,
) -> dict[str, list[CustomerNode]] | None:
    vehicles = sorted(
        (vehicle for vehicle in dataset.vehicles if vehicle_ids is None or vehicle.id in vehicle_ids),
        key=lambda vehicle: vehicle.id,
    )
    if not vehicles:
        return None
    routes: dict[str, list[CustomerNode]] = {vehicle.id: [] for vehicle in dataset.vehicles}
    ordered = sorted(
        nodes,
        key=lambda node: (
            -matrix.get(dataset.depot.id, node.store_id).distance_m,
            node.node_id,
        ),
    )
    vehicle_by_id = {vehicle.id: vehicle for vehicle in vehicles}
    radial = _geo_scale_m(nodes, dataset, matrix)
    shape_scale = 0.20 * radial
    bank_scale = 0.35 * radial
    for node in ordered:
        best: tuple[float, str, int] | None = None
        for vehicle in vehicles:
            current = routes[vehicle.id]
            for position in range(len(current) + 1):
                cost = _insertion_cost(
                    current,
                    position,
                    node,
                    vehicle,
                    matrix,
                    dataset.depot,
                    shapes=shapes,
                    banks=banks,
                    shape_scale=shape_scale,
                    bank_scale=bank_scale,
                )
                if cost is None:
                    continue
                candidate = (cost, vehicle.id, position)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None
        _, vehicle_id, position = best
        routes[vehicle_id].insert(position, node)
        if _simulate(routes[vehicle_id], vehicle_by_id[vehicle_id], matrix, dataset.depot) is None:
            return None
    return routes


def _merge_ends(left: list[CustomerNode], right: list[CustomerNode], left_id: str, right_id: str) -> list[CustomerNode] | None:
    if left[-1].node_id == left_id and right[0].node_id == right_id:
        return left + right
    if left[-1].node_id == left_id and right[-1].node_id == right_id:
        return left + list(reversed(right))
    if left[0].node_id == left_id and right[0].node_id == right_id:
        return list(reversed(left)) + right
    if left[0].node_id == left_id and right[-1].node_id == right_id:
        return list(reversed(left)) + list(reversed(right))
    return None


def clarke_wright(
    nodes: list[CustomerNode],
    dataset: Dataset,
    matrix: TravelMatrix,
    *,
    shapes: dict[str, CargoShape] | None = None,
    banks: dict[str, str] | None = None,
    vehicle_ids: set[str] | None = None,
) -> dict[str, list[CustomerNode]] | None:
    if not nodes:
        return {vehicle.id: [] for vehicle in dataset.vehicles}
    pool = [vehicle for vehicle in dataset.vehicles if vehicle_ids is None or vehicle.id in vehicle_ids]
    if not pool:
        return None
    largest = max(pool, key=lambda vehicle: (vehicle.volume_mm3, vehicle.max_weight_g))
    routes: list[list[CustomerNode]] = [[node] for node in nodes]
    index_of = {node.node_id: index for index, node in enumerate(nodes)}
    radial = _geo_scale_m(nodes, dataset, matrix)
    savings: list[tuple[int, str, str]] = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            depot_left = matrix.get(dataset.depot.id, left.store_id).distance_m
            depot_right = matrix.get(dataset.depot.id, right.store_id).distance_m
            between = matrix.get(left.store_id, right.store_id).distance_m
            saving = depot_left + depot_right - between
            if shapes is not None and left.node_id in shapes and right.node_id in shapes:
                saving += int(round(complementarity(shapes[left.node_id], shapes[right.node_id]) * 0.20 * radial))
            if banks is not None and banks.get(left.node_id) != banks.get(right.node_id):
                saving -= int(round(0.35 * radial))
            savings.append((saving, left.node_id, right.node_id))
    savings.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _saving, left_id, right_id in savings:
        left_route = index_of[left_id]
        right_route = index_of[right_id]
        if left_route == right_route:
            continue
        merged = _merge_ends(routes[left_route], routes[right_route], left_id, right_id)
        if merged is None or not _capacity_ok(merged, largest):
            continue
        if _simulate(merged, largest, matrix, dataset.depot) is None:
            continue
        routes[left_route] = merged
        routes[right_route] = []
        for node in merged:
            index_of[node.node_id] = left_route
    active = [route for route in routes if route]
    vehicles = sorted(pool, key=lambda vehicle: (vehicle.volume_mm3, vehicle.max_weight_g, vehicle.id))
    if len(active) > len(vehicles):
        return None
    assignment: dict[str, list[CustomerNode]] = {vehicle.id: [] for vehicle in dataset.vehicles}
    used: set[str] = set()
    for route in sorted(active, key=lambda item: -sum(node.volume_mm3 for node in item)):
        placed = False
        for vehicle in reversed(vehicles):
            if vehicle.id in used or not _capacity_ok(route, vehicle):
                continue
            if _simulate(route, vehicle, matrix, dataset.depot) is None:
                continue
            assignment[vehicle.id] = route
            used.add(vehicle.id)
            placed = True
            break
        if not placed:
            return None
    return assignment


def materialize_routes(
    assignment: dict[str, list[CustomerNode]],
    dataset: Dataset,
    matrix: TravelMatrix,
) -> list[RoutePlan] | None:
    vehicles = {vehicle.id: vehicle for vehicle in dataset.vehicles}
    plans: list[RoutePlan] = []
    for vehicle_id, nodes in assignment.items():
        if not nodes:
            continue
        plan = _simulate(nodes, vehicles[vehicle_id], matrix, dataset.depot)
        if plan is None:
            return None
        plans.append(plan)
    return plans


def total_distance(routes: list[RoutePlan]) -> int:
    return sum(route.distance_m for route in routes)
