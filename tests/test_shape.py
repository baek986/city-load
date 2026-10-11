"""Cargo-shape complementarity and the river-bank split."""

from app.domain.units import build_units, group_units_by_store
from app.network.roads import BLOCK_M, bank_of
from app.optimizer.routing import cheapest_insertion, clarke_wright
from app.optimizer.shape import complementarity, shape_from_units, shapes_from_nodes
from app.domain.matrix import TravelMatrix
from tests.test_packing import _unit


def test_bank_of_puts_the_depot_west_of_the_bridge() -> None:
    assert bank_of(0) == "west"
    assert bank_of(BLOCK_M) == "west"
    assert bank_of(BLOCK_M * 2) == "east"
    assert bank_of(4000) == "east"


def test_pharmacy_day_has_stores_on_both_banks() -> None:
    from app.demo.scenarios import pharmacy_day

    dataset = pharmacy_day()
    west = sum(1 for store in dataset.stores if bank_of(store.x_m) == "west")
    east = sum(1 for store in dataset.stores if bank_of(store.x_m) == "east")
    assert west >= 3
    assert east >= 3


def test_fragile_load_complements_a_sturdy_floor() -> None:
    sturdy = shape_from_units([_unit("a", 400, 400, 200, fragile=False)])
    glass = shape_from_units([_unit("b", 400, 400, 200, fragile=True)])
    mixed = complementarity(sturdy, glass)
    alike = complementarity(glass, glass)
    assert mixed > alike
    assert 0 <= alike <= mixed <= 1


def test_short_and_tall_boxes_complement_more_than_two_short_loads() -> None:
    short = shape_from_units([_unit("s", 400, 400, 100)])
    tall = shape_from_units([_unit("t", 400, 400, 400)])
    assert complementarity(short, tall) > complementarity(short, short)


def test_construction_with_shapes_still_covers_every_stop() -> None:
    from app.demo.scenarios import preview_day

    dataset = preview_day()
    matrix = TravelMatrix(dataset.arcs)
    units = build_units(dataset)
    nodes = group_units_by_store(dataset, units)
    units_by_id = {unit.unit_id: unit for unit in units}
    shapes = shapes_from_nodes(nodes, units_by_id)
    inserted = cheapest_insertion(nodes, dataset, matrix, shapes=shapes)
    saved = clarke_wright(nodes, dataset, matrix, shapes=shapes)
    assert inserted is not None
    assert saved is not None
    for assignment in (inserted, saved):
        covered = {node.node_id for sequence in assignment.values() for node in sequence}
        assert covered == {node.node_id for node in nodes}
