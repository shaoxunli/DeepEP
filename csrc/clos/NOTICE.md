# Ascend Clos backend provenance

Imported from https://gitcode.com/lishaoxun/ascend_deepep, branch `deepep`,
commit `c09040328c853a011dfe37c74463c44723db4476`.
The imported implementation is licensed under CANN Open Software License
Agreement Version 2.0; see `third-party/ascend-deepep.LICENSE` at repository root.
Retained copyright notices apply to their respective files.

`upstream.json` records the original kernel/include hashes and the minimal
CATLASS/TLA resource-header dependency closure. The latter is pinned to
`48f7461afcff7c9282c6372c19d8768d9c6e0d9d` and retains its CANN license notices.
It does not replace the existing MegaMoE CATLASS submodule.

The device sources and layouts are retained unchanged. Host integration adds
the cross-backend SHMEM ownership check. Python adaptation adds platform keyword
aliases, backend selection, event compatibility and a shared lifetime lock.
