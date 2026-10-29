#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <torch/custom_class.h>

std::vector<int64_t> GetShmemUniqueId();
std::tuple<int64_t, std::vector<int64_t>> GetShmemBootstrap(const std::string& key);

class ShmemInstance : public torch::jit::CustomClassHolder {
public:
    ShmemInstance(int64_t rank, int64_t world_size, std::string ip_port);
    ShmemInstance(int64_t rank, int64_t world_size, std::string ip_port, int64_t data_op_engine_type,
                  bool use_1g_huge_pages = false, std::vector<int64_t> unique_id = {}, std::string bootstrap_key = {},
                  bool v1_shared_jetty = false, bool v1_token_schedule = false);

    ~ShmemInstance();

    ShmemInstance(const ShmemInstance&) = delete;
    ShmemInstance& operator=(const ShmemInstance&) = delete;

    void Init();
    void Init(int64_t local_mem_size);
    void Init(int64_t local_mem_size, int64_t workspace_size);
    void Barrier();
    void WaitHandle(aclrtStream stream) const;
    void Synchronize(aclrtStream stream) const;
    void Destroy();

    uint64_t FftsAddr() const;
    bool Initialized() const;
    bool IsAvailable() const;
    int64_t Rank() const;
    int64_t WorldSize() const;
    void* WorkspacePtr() const;
    bool Use1gHugePages() const;
    bool SharedJetty() const
    {
        return shared_jetty_;
    }
    bool TokenSchedule() const
    {
        return token_schedule_;
    }

private:
    void InitRuntime();
    void AllocateHugePageWorkspace();
    void ReleaseHugePageWorkspace();
    void DestroyImpl(bool synchronize);
    void DestroyNoThrow() noexcept;

    int rank_;
    int world_size_;
    uint64_t local_mem_size_;
    uint64_t workspace_size_;
    uint64_t workspace_allocation_size_;
    std::string ip_port_;
    std::vector<int64_t> unique_id_;
    std::string bootstrap_key_;
    int64_t data_op_engine_type_;
    bool use_1g_huge_pages_;
    bool shared_jetty_;
    bool token_schedule_;
    bool initialized_;
    bool acquired_global_runtime_;
    bool workspace_source_mapped_;

    aclrtDrvMemHandle workspace_handle_;
    void* workspace_source_ptr_;
    void* workspace_ptr_;

    uint64_t ffts_addr_;
};
