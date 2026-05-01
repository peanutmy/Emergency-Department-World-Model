"""Order storage for the deterministic ED workflow layer."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


OrderType = Literal["drug", "intervention", "lab"]
OrderPriority = Literal["stat", "urgent", "routine"]
OrderStatus = Literal[
    "ordered",
    "validated",
    "pending_execution",
    "in_progress",
    "completed",
    "failed",
    "cancelled",
]

SUPPORTED_ORDER_TYPES = {"drug", "intervention", "lab"}


@dataclass
class ClinicalOrder:
    """Structured clinician order tracked outside EMSim physiology."""

    order_id: str = ""
    ordered_by: str = "clinician"
    order_type: str = "drug"
    payload: dict[str, Any] = field(default_factory=dict)
    priority: str = "stat"
    status: str = "ordered"
    ordered_at_s: float = 0.0
    started_at_s: float | None = None
    completed_at_s: float | None = None

    def __post_init__(self) -> None:
        self.payload = deepcopy(self.payload)
        self.ordered_at_s = float(self.ordered_at_s)
        if self.started_at_s is not None:
            self.started_at_s = float(self.started_at_s)
        if self.completed_at_s is not None:
            self.completed_at_s = float(self.completed_at_s)


class OrderManager:
    """In-memory store for hand-typed structured orders."""

    def __init__(self, orders: list[ClinicalOrder] | None = None):
        self.orders: dict[str, ClinicalOrder] = {}
        self._next_order_number = 1
        for order in orders or []:
            self.add_order(order)

    def create_order(
        self,
        *,
        order_type: str,
        payload: dict[str, Any] | None = None,
        ordered_by: str = "clinician",
        priority: str = "stat",
        ordered_at_s: float = 0.0,
        order_id: str | None = None,
        status: str = "ordered",
    ) -> ClinicalOrder:
        order = ClinicalOrder(
            order_id=order_id or self._make_order_id(),
            ordered_by=ordered_by,
            order_type=order_type,
            payload=payload or {},
            priority=priority,
            status=status,
            ordered_at_s=ordered_at_s,
        )
        return self.add_order(order)

    def submit_order(
        self,
        order: ClinicalOrder | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClinicalOrder:
        """Create or store an order from a dataclass, dict, or keyword fields."""

        if isinstance(order, ClinicalOrder):
            return self.add_order(order)

        order_kwargs = dict(order or {})
        order_kwargs.update(kwargs)
        order_type = order_kwargs.get("order_type")
        if order_type is None:
            raise ValueError("order_type is required")
        payload = order_kwargs.get("payload")
        if payload is None:
            metadata_keys = {
                "order_id",
                "ordered_by",
                "order_type",
                "type",
                "priority",
                "status",
                "ordered_at_s",
            }
            payload = {
                key: deepcopy(value)
                for key, value in order_kwargs.items()
                if key not in metadata_keys
            }
        return self.create_order(
            order_type=order_type,
            payload=payload,
            ordered_by=order_kwargs.get("ordered_by", "clinician"),
            priority=order_kwargs.get("priority", "stat"),
            ordered_at_s=order_kwargs.get("ordered_at_s", 0.0),
            order_id=order_kwargs.get("order_id"),
            status=order_kwargs.get("status", "ordered"),
        )

    submit = submit_order

    def add_order(self, order: ClinicalOrder | dict[str, Any]) -> ClinicalOrder:
        if isinstance(order, dict):
            return self.submit_order(order)
        if not order.order_id:
            order.order_id = self._make_order_id()
        if order.order_id in self.orders:
            raise ValueError(f"duplicate order_id {order.order_id!r}")
        if order.order_type not in SUPPORTED_ORDER_TYPES:
            raise ValueError(f"unsupported order_type {order.order_type!r}")
        if order.status != "ordered":
            raise ValueError("incoming orders must have status 'ordered'")
        self.orders[order.order_id] = order
        return order

    def get_order(self, order_id: str) -> ClinicalOrder:
        return self.orders[order_id]

    def list_orders(self, status: str | None = None) -> list[ClinicalOrder]:
        if status is None:
            return list(self.orders.values())
        return [order for order in self.orders.values() if order.status == status]

    def set_status(
        self,
        order_id: str,
        status: str,
        *,
        now_s: float | None = None,
    ) -> ClinicalOrder:
        order = self.get_order(order_id)
        order.status = status
        if now_s is not None and status == "in_progress" and order.started_at_s is None:
            order.started_at_s = float(now_s)
        if now_s is not None and status in {"completed", "failed", "cancelled"}:
            order.completed_at_s = float(now_s)
        return order

    def _make_order_id(self) -> str:
        order_id = f"order-{self._next_order_number}"
        self._next_order_number += 1
        return order_id
