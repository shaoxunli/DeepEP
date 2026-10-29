"""Platform-named adapters over the pinned Clos V1/V2 implementation."""

from ascend_deepep.buffer import Buffer as SourceBuffer
from ascend_deepep.elastic import ElasticBuffer as SourceElasticBuffer
from ._compat import platform_keywords


class Buffer(SourceBuffer):
    __init__ = platform_keywords(
        SourceBuffer.__init__,
        {
            "num_scaleup_bytes": "num_nvl_bytes",
            "num_scaleout_bytes": "num_rdma_bytes",
            "allow_memory_transport_for_low_latency_mode": "allow_nvlink_for_low_latency_mode",
            "allow_multinode_memory_transport": "allow_mnnvl",
        },
    )
    dispatch = platform_keywords(SourceBuffer.dispatch, {"num_tokens_per_scaleout_rank": "num_tokens_per_rdma_rank"})
    set_num_compute_units = staticmethod(SourceBuffer.set_num_sms)


class ElasticBuffer(SourceElasticBuffer):
    __init__ = platform_keywords(SourceElasticBuffer.__init__, {"num_device_timeout_secs": "num_gpu_timeout_secs"})
    dispatch = platform_keywords(SourceElasticBuffer.dispatch, {"num_compute_units": "num_sms"})
    combine = platform_keywords(SourceElasticBuffer.combine, {"num_compute_units": "num_sms"})
    get_theoretical_num_compute_units = platform_keywords(
        SourceElasticBuffer.get_theoretical_num_sms,
        {
            "scaleup_gbs": "nvlink_gbs",
            "scaleout_gbs": "rdma_gbs",
            "compute_unit_read_gbs": "sm_read_gbs",
            "compute_unit_write_gbs": "sm_write_gbs",
        },
    )
    get_theoretical_num_qps = platform_keywords(
        SourceElasticBuffer.get_theoretical_num_qps, {"num_compute_units": "num_sms"}
    )


def buffer_class(name):
    return {"Buffer": Buffer, "ElasticBuffer": ElasticBuffer}[name]
