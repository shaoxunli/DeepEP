"""Full-CLOS host scheduling. Core counts remain operation inputs."""

import logging
from dataclasses import dataclass, replace
from functools import lru_cache

from .config import Config, _MAX_NUM_AIV


def detect_pod_hosts(group, comm, size, enabled, transport):
    """Collect EP-local physical host evidence only for the measured topology."""
    if size != 64:
        return None
    from ..._card_adapter import host_identity
    from .buffer import _bootstrap_all_gather

    try:
        host = host_identity() if enabled is True and transport == "udma" else None
    except (OSError, ValueError):
        host = None
    evidence = _bootstrap_all_gather(group, comm, size, (enabled, transport, host))
    if any(type(item[0]) is not bool for item in evidence):
        raise TypeError("udma_auto_policy must be bool on every EP rank")
    if any(item[:2] != evidence[0][:2] for item in evidence):
        raise ValueError("EP ranks must agree on udma_auto_policy and transport")
    return [item[2] for item in evidence] if enabled and transport == "udma" else None


def _is_measured_pod(world_size, hosts):
    if world_size != 64 or hosts is None or len(hosts) != 64 or not all(hosts):
        return False
    groups = [hosts[i : i + 8] for i in range(0, 64, 8)]
    return all(len(set(g)) == 1 for g in groups) and len({g[0] for g in groups}) == 8


@lru_cache(maxsize=128)
def _warn_ignored(name, supplied, effective):
    logging.getLogger(__name__).warning(
        "UDMA auto policy: ignoring explicit %s=%s; effective %s=%s. "
        "Set udma_auto_policy=False to enable manual tuning.",
        name,
        supplied,
        name,
        effective,
    )


def _check_override(name, supplied, effective, rank):
    if rank == 0 and supplied is not None and supplied != effective:
        _warn_ignored(name, repr(supplied), repr(effective))


def validate_manual_options(
    schedule,
    dispatch_order,
    combine_order,
    ranks_per_group,
    peers_per_group_per_worker,
    workers,
    qps,
    peers=None,
    wqes=None,
    *,
    operation_config=False,
):
    """Require all knobs used by this schedule; V1 P/WQE live in Config."""
    required = {"udma_schedule": schedule}
    if schedule in (None, "peer"):
        required.update(
            udma_dispatch_peer_order=dispatch_order,
            udma_combine_peer_order=combine_order,
            udma_workers_per_peer=workers,
            udma_qps_per_worker=qps,
        )
        if any(order is not None and order != "rank" for order in (dispatch_order, combine_order)):
            required["udma_ranks_per_group"] = ranks_per_group
        if any(order in ("group_round_robin", "group_window") for order in (dispatch_order, combine_order)):
            required["udma_peers_per_group_per_worker"] = peers_per_group_per_worker
        if not operation_config and any(
            order not in ("group_round_robin", "group_window") for order in (dispatch_order, combine_order)
        ):
            required["udma_peers_per_worker"] = peers
    if not operation_config:
        required["udma_wqes_per_visit"] = wqes
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("manual UDMA policy requires explicit parameters: " + ", ".join(missing))


def validate_manual_config(config, schedule, dispatch_order, combine_order):
    required = ["udma_wqes_per_visit"]
    if schedule == "peer" and any(
        order not in ("group_round_robin", "group_window") for order in (dispatch_order, combine_order)
    ):
        required.append("udma_peers_per_worker")
    flags = {"udma_wqes_per_visit": "_udma_wqes_explicit", "udma_peers_per_worker": "_udma_peers_explicit"}
    missing = [name for name in required if config is None or not getattr(config, flags[name], False)]
    if missing:
        raise ValueError("manual UDMA policy requires explicit Config parameters: " + ", ".join(missing))


@dataclass(frozen=True)
class UdmaPolicy:
    world_size: int
    peers_per_worker: int
    workers_per_peer: int
    qps_per_worker: int
    wqes_per_visit: int
    version: str = "clos-v2"
    peer_order: str = "rank"
    ranks_per_group: int = 0
    cross_lane0_weight: int = 1

    @property
    def shape_policy(self):
        return self.peer_order == "group_window"

    @property
    def send_workers(self):
        if self.shape_policy:
            return ((self.ranks_per_group - 1) // self.peers_per_worker + 1) * self.workers_per_peer
        return ((self.world_size - 2) // self.peers_per_worker + 1) * self.workers_per_peer

    def signature(self):
        return (
            self.version,
            self.world_size,
            self.peers_per_worker,
            self.workers_per_peer,
            self.qps_per_worker,
            self.wqes_per_visit,
            self.peer_order,
            self.ranks_per_group,
            self.cross_lane0_weight,
        )

    def validate_num_aiv(self, num_aiv):
        """Check the supplied count without changing it or the selected schedule."""
        if isinstance(num_aiv, bool) or not isinstance(num_aiv, int):
            raise TypeError("num_aiv/num_sms must be an integer")
        if num_aiv < 0 or num_aiv % 2 or num_aiv > _MAX_NUM_AIV:
            raise ValueError(f"num_aiv/num_sms must be zero or an even count <= {_MAX_NUM_AIV}")
        effective = 32 if num_aiv == 0 else num_aiv
        if self.send_workers > effective or self.send_workers * self.qps_per_worker > 32:
            raise ValueError(
                f"UDMA policy requires {self.send_workers} sending AIVs and "
                f"{self.send_workers * self.qps_per_worker} QP slots; got {effective} AIVs "
                "with a 32-QP budget. Increase the operation core count or use manual tuning."
            )

    @lru_cache(maxsize=512, typed=True)
    def for_num_aiv(self, num_aiv):
        """Fit peer ownership to the operation budget without changing QP lanes.

        PeerMask provisions all P=1/2/4/8 mappings for this immutable W/Q.
        W is at most two in auto mode, so every supported even core budget
        can retain W; only the peer count may need to grow. No runtime state
        changes, including when different handles use different core counts.
        """
        if isinstance(num_aiv, bool) or not isinstance(num_aiv, int):
            raise TypeError("num_aiv/num_sms must be an integer")
        if num_aiv < 0 or num_aiv % 2 or num_aiv > _MAX_NUM_AIV:
            raise ValueError(f"num_aiv/num_sms must be zero or an even count <= {_MAX_NUM_AIV}")
        budget = min(32 // self.qps_per_worker, 32 if num_aiv == 0 else num_aiv)
        policy = self
        while policy.send_workers > budget and policy.peers_per_worker < 8:
            policy = replace(policy, peers_per_worker=policy.peers_per_worker * 2)
        policy.validate_num_aiv(num_aiv)
        return policy

    def for_row_bytes(self, num_aiv, payload_bytes, scale_bytes=0):
        """Resolve from actual transfer rows; no tensor readback or QP changes.

        The 8 KiB boundary interpolates the measured 5632/7392/14336-byte
        EP64 rows. It is not a claim of optimality for every intermediate shape.
        """
        for name, value in (("payload_bytes", payload_bytes), ("scale_bytes", scale_bytes)):
            if type(value) is not int or value < (1 if name == "payload_bytes" else 0):
                raise ValueError(f"{name} must be a {'positive' if name == 'payload_bytes' else 'nonnegative'} integer")
        policy = self.for_num_aiv(num_aiv)
        if policy.shape_policy:
            return replace(policy, wqes_per_visit=8 if payload_bytes + scale_bytes <= 8192 else 4)
        return policy

    def config(self, num_aiv=0, *, payload_bytes=None, scale_bytes=0):
        policy = (
            self.for_num_aiv(num_aiv)
            if payload_bytes is None
            else self.for_row_bytes(num_aiv, payload_bytes, scale_bytes)
        )
        return Config(num_aiv, udma_peers_per_worker=policy.peers_per_worker, udma_wqes_per_visit=policy.wqes_per_visit)

    def resolve_config(self, config, default_num_aiv, *, rank=0, payload_bytes=None, scale_bytes=0):
        from ..._config import Config as PublicConfig

        if config is None:
            return self.config(default_num_aiv, payload_bytes=payload_bytes, scale_bytes=scale_bytes)
        if isinstance(config, PublicConfig):
            config = config._impl
        if not isinstance(config, Config):
            raise TypeError("config must be a Config")
        policy = (
            self.for_num_aiv(config.num_aiv)
            if payload_bytes is None
            else self.for_row_bytes(config.num_aiv, payload_bytes, scale_bytes)
        )
        for name, flag, effective in (
            ("udma_peers_per_worker", "_udma_peers_explicit", policy.peers_per_worker),
            ("udma_wqes_per_visit", "_udma_wqes_explicit", policy.wqes_per_visit),
        ):
            if getattr(config, flag, True):
                _check_override(name, getattr(config, name), effective, rank)
        return Config(
            config.num_aiv,
            config.num_max_nvl_chunked_send_tokens,
            config.num_max_nvl_chunked_recv_tokens,
            config.num_max_rdma_chunked_send_tokens,
            config.num_max_rdma_chunked_recv_tokens,
            udma_peers_per_worker=policy.peers_per_worker,
            udma_wqes_per_visit=policy.wqes_per_visit,
        )


def resolve_udma_policy(
    enabled,
    world_size,
    transport,
    schedule,
    dispatch_order,
    combine_order,
    ranks_per_group,
    peers_per_group_per_worker,
    workers=None,
    qps=None,
    peers=None,
    wqes=None,
    *,
    rank=0,
    operation_config=False,
    host_ids=None,
):
    """Auto is authoritative. Manual mode requires its relevant inputs."""
    if not isinstance(enabled, bool):
        raise TypeError("udma_auto_policy must be a bool")
    # Native transport='auto' currently selects MTE, not UDMA.
    if transport != "udma":
        return None
    if not enabled:
        validate_manual_options(
            schedule,
            dispatch_order,
            combine_order,
            ranks_per_group,
            peers_per_group_per_worker,
            workers,
            qps,
            peers,
            wqes,
            operation_config=operation_config,
        )
        return None
    if isinstance(world_size, bool) or not isinstance(world_size, int) or not 2 <= world_size <= 1024:
        raise ValueError("automatic UDMA policy requires EP between 2 and 1024")
    w, q = (2 if world_size <= 8 else 1), 1
    p, visit = (1, 8) if world_size <= 16 else (4, 32)
    while ((world_size - 2) // p + 1) * w * q > 32 and p < 8:
        p *= 2
    if ((world_size - 2) // p + 1) * w * q > 32:
        raise ValueError("automatic UDMA policy cannot fit the 32-QP budget with P<=8")
    pod = _is_measured_pod(world_size, host_ids)
    if pod:
        p, w, q, visit = 1, 2, 1, 8
    order, group = ("group_window", 8) if pod else ("rank", None)
    for name, supplied, effective in (
        ("udma_schedule", schedule, "peer"),
        ("udma_dispatch_peer_order", dispatch_order, order),
        ("udma_combine_peer_order", combine_order, order),
        ("udma_ranks_per_group", ranks_per_group, group),
        ("udma_peers_per_group_per_worker", peers_per_group_per_worker, p if pod else None),
        ("udma_workers_per_peer", workers, w),
        ("udma_qps_per_worker", qps, q),
        ("udma_peers_per_worker", peers, p),
        ("udma_wqes_per_visit", wqes, visit),
    ):
        _check_override(name, supplied, effective, rank)
    return UdmaPolicy(
        world_size,
        p,
        w,
        q,
        visit,
        version="clos-pod-bytes-v1" if pod else "clos-v2",
        peer_order=order,
        ranks_per_group=group or 0,
        cross_lane0_weight=4 if pod else 1,
    )
