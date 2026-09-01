"""Training losses for 4RC."""

from .geometry import GeometryLoss
from .tcp_tracking import TCPTrackingLoss

__all__ = ["GeometryLoss", "TCPTrackingLoss"]
