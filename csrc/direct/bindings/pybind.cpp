/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <chrono>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/types.h"
#include "runtime/kernel_trace.h"
#include "runtime/shmem_instance.h"

namespace {
uint32_t TracePhysicalDeviceId(int device)
{
    using Query = aclError (*)(int32_t, int32_t*);
    auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByUserDevId"));
    if (!query) query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
    int32_t physical_device_id = -1;
    if (!query || query(device, &physical_device_id) != ACL_SUCCESS || physical_device_id < 0)
        throw std::runtime_error("physical device ID unavailable for ACL device " + std::to_string(device));
    return static_cast<uint32_t>(physical_device_id);
}

std::string TraceDeviceUuid(int device)
{
    using Query = aclError (*)(int32_t, aclrtUuid*);
    const auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "aclrtDeviceGetUuid"));
    aclrtUuid uuid{};
    if (!query || query(device, &uuid) != ACL_SUCCESS)
        throw std::runtime_error("device UUID unavailable for ACL device " + std::to_string(device));
    constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    for (const unsigned char byte : uuid.bytes) {
        hex += digits[byte >> 4];
        hex += digits[byte & 0xf];
    }
    if (hex.find_first_not_of('0') == std::string::npos)
        throw std::runtime_error("empty device UUID for ACL device " + std::to_string(device));
    return hex;
}

int ProbePythonGil()
{
#ifdef Py_GIL_DISABLED
    return -1;
#else
    return PyGILState_Check(); // Observe only; never acquire or release the GIL.
#endif
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.attr("kernel_trace_diagnostic_version") = 4;
    m.attr("host_dispatch_probe_version") = 3;
    m.attr("kernel_trace_metadata_version") = 2;
    m.def("_configure_trace_test_mode", &ascend_deepep::trace::ConfigureTraceTestMode,
          pybind11::arg("metadata_only") = false);
    m.def("check_host_dispatch_gil_probe", []() {
        const int held = ProbePythonGil();
        int released;
        { pybind11::gil_scoped_release release; released = ProbePythonGil(); }
        pybind11::dict result;
        result["held_control"] = held;
        result["released_control"] = released;
        return result;
    });
    m.def("configure_host_dispatch_probe", [](int64_t capacity, bool device_timing) {
        if (capacity < 0) throw std::runtime_error("negative Host dispatch probe capacity");
        ascend_deepep::trace::ConfigureHostDispatchProbe(static_cast<std::size_t>(capacity), &ProbePythonGil, device_timing);
    }, pybind11::arg("capacity"), pybind11::arg("device_timing") = false);
    m.def("take_host_dispatch_probe", []() {
        auto report = ascend_deepep::trace::TakeHostDispatchProbe();
        static constexpr const char* names[] = {"native_entry", "notify_begin", "notify_end", "wait_begin",
            "first_pending", "first_count_ready", "all_counts_ready", "wait_end", "outputs_begin", "outputs_end",
            "dispatch_begin", "dispatch_end", "native_exit"};
        static_assert(sizeof(names) / sizeof(names[0]) == ascend_deepep::trace::kDispatchProbePoints);
        pybind11::list samples;
        for (const auto& row : report.samples) {
            pybind11::dict item, ticks, gil;
            item["ordinal"] = row.ordinal;
            item["rank"] = row.rank;
            item["cached"] = row.cached;
            item["wait_path"] = row.wait_path;
            item["api_call_id"] = row.api_call_id;
            item["success"] = row.success;
            pybind11::dict device;
            device["sequence"] = row.notify_sequence;
            device["done_seq"] = row.notify_done;
            static constexpr const char* device_names[] = {"entry", "counts_ready", "publish_begin", "publish_end"};
            for (unsigned i = 0; i < 4; ++i) device[device_names[i]] = row.notify_ticks[i];
            static constexpr const char* sync_names[] = {"entry_collective", "channel_counts_join", "channel_prefix_join",
                "counts_collective", "gather_join", "expert_reuse_join", "expert_ready_join", "final_join"};
            pybind11::dict sync_ns, sync_calls;
            for (unsigned i = 0; i < 8; ++i) {
                sync_ns[sync_names[i]] = row.notify_ticks[4 + i];
                sync_calls[sync_names[i]] = row.notify_ticks[12 + i];
            }
            device["sync_ns"] = sync_ns;
            device["sync_calls"] = sync_calls;
            item["notify_device"] = device;
            for (unsigned i = 0; i < ascend_deepep::trace::kDispatchProbePoints; ++i) {
                ticks[names[i]] = row.ticks[i];
                gil[names[i]] = row.gil[i];
            }
            item["steady_ns"] = ticks;
            item["gil_held"] = gil;
            samples.append(item);
        }
        pybind11::dict result;
        result["version"] = 1;
        result["dropped"] = report.dropped;
        result["samples"] = samples;
        return result;
    });
    using ascend_deepep::trace::KernelTrace;
    using ascend_deepep::trace::TraceRecord;
    struct Snapshot { ascend_deepep::trace::TraceMetadataSnapshot value; };
    pybind11::class_<Snapshot>(m, "TraceMetadataSnapshot");
    m.def("make_trace_metadata", [](pybind11::dict fields, const Snapshot& parent, bool recorder) {
        auto value = std::make_shared<ascend_deepep::trace::TraceMetadata>();
        value->parent = parent.value;
        if (recorder && !parent.value)
            value->op_counters = std::make_shared<ascend_deepep::trace::KernelOpCounters>();
        for (const auto& item : fields) {
            auto key = pybind11::cast<std::string>(item.first);
            if (pybind11::isinstance<pybind11::str>(item.second)) value->labels[key] = pybind11::cast<std::string>(item.second);
            else value->values[key] = pybind11::cast<int64_t>(item.second);
        }
        return Snapshot{value};
    }, pybind11::arg("fields"), pybind11::arg("parent") = Snapshot{}, pybind11::arg("recorder") = false);
    m.def("push_trace_metadata", [](const Snapshot& base, int64_t api, int64_t handle, int64_t dispatch_count) {
        auto value = ascend_deepep::trace::MakeCallMetadata(base.value, api, handle, dispatch_count);
        return Snapshot{ascend_deepep::trace::ExchangeTraceMetadata(value)};
    }, pybind11::arg("base"), pybind11::arg("api"), pybind11::arg("handle"), pybind11::arg("dispatch_count") = -1);
    m.def("restore_trace_metadata", [](const Snapshot& previous) {
        ascend_deepep::trace::ExchangeTraceMetadata(previous.value);
    });
    auto to_dict = [](const TraceRecord& record) {
        pybind11::dict result;
        for (const auto& item : record.labels) result[pybind11::str(item.first)] = item.second;
        result["event"] = record.event;
        if (!record.op_name.empty()) result["op_name"] = record.op_name;
        if (!record.kernel_name.empty()) result["kernel_name"] = record.kernel_name;
        for (const auto& item : record.values) result[pybind11::str(item.first)] = item.second;
        return result;
    };
    pybind11::class_<KernelTrace, std::shared_ptr<KernelTrace>>(m, "KernelTrace")
        .def_property_readonly("identity", [](const KernelTrace& trace) {
            return reinterpret_cast<uintptr_t>(&trace);
        })
        .def_property_readonly("poll_interval_ms", &KernelTrace::PollIntervalMs)
        .def("calibration", [to_dict](const KernelTrace& trace) { return to_dict(trace.Calibration()); })
        .def(
            "device_identity",
            [](const KernelTrace& trace) {
                const int device = trace.Device();
                return std::pair{TracePhysicalDeviceId(device), TraceDeviceUuid(device)};
            },
            pybind11::call_guard<pybind11::gil_scoped_release>())
        .def(
            "drain_production_batches",
            [](KernelTrace& trace, uint32_t bus_id, bool closing) {
                std::map<std::string, std::string> batches;
                {
                    pybind11::gil_scoped_release release;
                    batches = trace.DrainProductionBatches(bus_id, closing);
                }
                pybind11::list result;
                for (const auto& [comm_id, jsonl] : batches)
                    result.append(pybind11::make_tuple(pybind11::str(comm_id), pybind11::bytes(jsonl)));
                return result;
            },
            pybind11::arg("bus_id"), pybind11::arg("closing") = false)
        .def("drain_trace_json", [](KernelTrace& trace, bool closing) {
            std::string text;
            {
                pybind11::gil_scoped_release release;
                auto records = trace.Drain(closing);
                text = ascend_deepep::trace::EncodeTraceRecords(records);
            } // Native records and maps are destroyed before reacquiring GIL.
            return text;
        }, pybind11::arg("closing") = false)
        .def("drain_trace", [to_dict](KernelTrace& trace, bool closing) {
            std::vector<TraceRecord> records;
            {
                pybind11::gil_scoped_release release;
                records = trace.Drain(closing);
            }
            pybind11::list result;
            for (const auto& record : records) result.append(to_dict(record));
            return result;
        }, pybind11::arg("closing") = false)
        .def("synchronize", &KernelTrace::Synchronize, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("drain_trace_diagnostic", [](KernelTrace& trace, bool include_records, bool timing, bool closing,
                                         bool return_text) {
            using Clock = std::chrono::steady_clock;
            const auto now = [timing]() -> int64_t {
                return timing ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now().time_since_epoch()).count() : 0;
            };
            std::vector<TraceRecord> records;
            std::map<std::string, int64_t> counters{{"polls", 1}, {"records", 0}, {"launches", 0},
                {"ends", 0}, {"launch_errors", 0}, {"summaries", 0}, {"completed", 0},
                {"incomplete", 0}, {"dropped", 0}, {"encoded_bytes", 0}, {"returned_text_bytes", 0}};
            int64_t drain_ns{}, serialize_ns{};
            std::string text;
            {
                pybind11::gil_scoped_release release;
                const auto begin = now();
                records = trace.Drain(closing);  // Same done_seq validation and slot recycling as production.
                drain_ns = now() - begin;
                counters["records"] = records.size();
                for (const auto& record : records) {
                    if (record.event == "launch") ++counters["launches"];
                    else if (record.event == "end") ++counters["ends"];
                    else if (record.event == "launch_error") ++counters["launch_errors"];
                    else if (record.event == "summary") {
                        ++counters["summaries"];
                        for (const char* key : {"completed", "incomplete", "dropped"})
                            counters[key] += record.values.at(key);
                    }
                }
                if (include_records) {
                    const auto encode_begin = now();
                    text = ascend_deepep::trace::EncodeTraceRecords(records);
                    serialize_ns = now() - encode_begin;
                    counters["encoded_bytes"] = text.size();
                    if (!return_text) {
                        // Keep encoding and recycling identical, but free the payload before reacquiring the GIL.
                        std::string{}.swap(text);
                    }
                }
                counters["returned_text_bytes"] = text.size();
                records.clear();
            }
            const auto begin = now();
            pybind11::str converted(text);
            const auto convert_ns = now() - begin;
            pybind11::dict result;
            for (const auto& item : counters) result[pybind11::str(item.first)] = item.second;
            result["drain_ns"] = drain_ns;
            result["convert_ns"] = convert_ns;
            result["text"] = converted;
            result["serialize_ns"] = serialize_ns;
            return result;
        }, pybind11::arg("include_records"), pybind11::arg("timing") = false, pybind11::arg("closing") = false,
           pybind11::arg("return_text") = true);
    m.def("get_kernel_trace", &KernelTrace::Open);
    m.def("get_shmem_unique_id" , &GetShmemUniqueId);
    m.def("get_shmem_bootstrap", &GetShmemBootstrap);
    // Keep the Python dtype contract tied to the native V1 dispatch path.
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<ascend_deepep::topk_idx_t>::value);
}
