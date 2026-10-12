"""Planning errors with stable codes for the API."""


class PlanError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        unit_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.unit_ids = list(unit_ids or [])
        self.store_ids = list(store_ids or [])


class MissingTravelArcError(PlanError):
    def __init__(self, from_id: str, to_id: str) -> None:
        super().__init__(
            "missing_travel_arc",
            f"Missing travel arc {from_id} -> {to_id}",
        )
        self.from_id = from_id
        self.to_id = to_id


class CargoExceedsVehicleError(PlanError):
    def __init__(self, unit_ids: list[str]) -> None:
        super().__init__(
            "cargo_exceeds_vehicle",
            "At least one package does not fit in any vehicle",
            unit_ids=unit_ids,
        )
