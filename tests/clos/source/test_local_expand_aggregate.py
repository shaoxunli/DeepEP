"""Boundary checks for local expand/aggregate, using an initialized native Buffer.

The caller owns device selection and Buffer lifetime. No SHMEM communication is
issued here. Returned bit-pattern digests allow an old/new kernel comparison,
including floating-point exceptional values whose CPU encoding can differ.
"""

import hashlib

import torch


def _bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8)


def _equal(actual, expected):
    assert torch.equal(_bits(actual), _bits(expected)), (actual.shape, expected.shape)


def _expand_reference(x, routes, weights, experts, alignment):
    counts = [(routes == expert).sum().item() for expert in range(experts)]
    padded = [(count + alignment - 1) // alignment * alignment for count in counts]
    offsets = [0]
    for count in padded:
        offsets.append(offsets[-1] + count)
    mapping = torch.full(routes.shape, -1, dtype=torch.int32)
    output = torch.zeros((offsets[-1], x.size(1)), dtype=x.dtype)
    values = torch.zeros(offsets[-1], dtype=torch.float32)
    for expert in range(experts):
        cursor = offsets[expert]
        for token, slot in (routes == expert).nonzero().tolist():
            mapping[token, slot] = cursor
            output[cursor] = x[token]
            values[cursor] = weights[token, slot]
            cursor += 1
    return (
        output,
        values,
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(counts, dtype=torch.int32),
        mapping,
        padded,
    )


def check_local_stages(native, device, num_aiv):
    checks = 0
    digests = {}
    two_route_digests = {}
    # Empty cores, mapping tile boundaries, odd row sizes, hidden chunk tails,
    # short expert tiles, multiple expert tiles, duplicate and absent routes.
    shapes = [
        (0, 17, 3, 7, 1),
        (1, 31, 3, 7, 8),
        (17, 64, 8, 32, 1),
        (129, 128, 8, 32, 8),
        (17, 6144, 8, 249, 1),
        (9, 15425, 3, 1024, 8),
        (3, 17, 128, 1024, 1),
    ]
    for dtype in (torch.bfloat16, torch.float16):
        for tokens, hidden, topk, experts, alignment in shapes:
            x = ((torch.arange(tokens * hidden).reshape(tokens, hidden) % 23 - 11).float() / 8).to(dtype)
            routes = torch.full((tokens, topk), -1, dtype=torch.int64)
            for token in range(tokens):
                for k in range(min(token % 4, topk)):
                    routes[token, k] = (token * 17 + k // 2) % experts
            weights = torch.arange(tokens * topk).reshape(tokens, topk).float() / 128
            expected, values, offsets, counts, mapping, padded = _expand_reference(
                x, routes, weights, experts, alignment
            )
            result = native.expand(
                x.to(device),
                None,
                routes.to(device),
                weights.to(device),
                padded,
                alignment,
                0,
                None,
                None,
                None,
                num_aiv,
                None,
            )
            payload, _, got_values, got_offsets, got_counts, got_mapping = result
            _equal(got_offsets, offsets)
            _equal(got_counts, counts)
            _equal(got_mapping, mapping)
            valid = mapping[mapping >= 0].long()
            _equal(payload[valid.to(device)], expected[valid])
            _equal(got_values[valid.to(device)], values[valid])
            cached = native.expand(
                (x + 1).to(device),
                None,
                None,
                weights.to(device),
                [],
                alignment,
                expected.size(0),
                got_offsets,
                got_counts,
                got_mapping,
                num_aiv,
                None,
            )
            expected_cached = _expand_reference(x + 1, routes, weights, experts, alignment)[0]
            _equal(cached[0][valid.to(device)], expected_cached[valid])
            for weighted in (False, True):
                route_weights = torch.full((expected.size(0),), 0.5, dtype=torch.float32) if weighted else None
                actual, restored = native.aggregate(
                    payload,
                    None if route_weights is None else route_weights.to(device),
                    got_values,
                    got_mapping,
                    num_aiv,
                    None,
                )
                ref = torch.zeros_like(x, dtype=torch.float32)
                restored_ref = torch.zeros_like(weights)
                for token in range(tokens):
                    for k in range(topk):
                        slot = int(mapping[token, k])
                        if slot >= 0:
                            ref[token] += expected[slot].float() * (0.5 if weighted else 1.0)
                            restored_ref[token, k] = values[slot]
                _equal(actual, ref.to(dtype))
                _equal(restored, restored_ref)
                checks += 1
            checks += 1

        # Exhaust all 16-bit payload patterns, including both zero signs,
        # subnormals, infinities and NaN payloads, on the single-route path.
        raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).reshape(64, 1024)
        mapping = torch.full((64, 8), -1, dtype=torch.int32)
        mapping[:, 3] = torch.arange(64, dtype=torch.int32)
        out, _ = native.aggregate(raw.to(device), None, None, mapping.to(device), num_aiv, None)
        # Use raw digests for exact old/new comparison rather than treating
        # CPU NaN canonicalization as the device's bit-pattern oracle.
        digests[str(dtype)] = hashlib.sha256(_bits(out).numpy().tobytes()).hexdigest()
        finite_normal = torch.isfinite(raw.float()) & ((raw.float().abs() >= torch.finfo(dtype).tiny) | (raw == 0))
        _equal(
            out.cpu()[finite_normal],
            (torch.zeros_like(raw, dtype=torch.float32) + raw.float()).to(dtype)[finite_normal],
        )
        checks += 1

        # Also compare two contributors bit-for-bit with the baseline. The
        # second contributor changes zero signs, NaN payloads and cancellation.
        paired = raw.flip(0).contiguous()
        pair_mapping = mapping.clone()
        pair_mapping[:, 6] = torch.arange(64, 128, dtype=torch.int32)
        out, _ = native.aggregate(
            torch.cat((raw, paired)).to(device), None, None, pair_mapping.to(device), num_aiv, None
        )
        two_route_digests[str(dtype)] = hashlib.sha256(_bits(out).numpy().tobytes()).hexdigest()
        normal = (
            finite_normal
            & torch.isfinite(paired.float())
            & ((paired.float().abs() >= torch.finfo(dtype).tiny) | (paired == 0))
        )
        _equal(
            out.cpu()[normal],
            (torch.zeros_like(raw, dtype=torch.float32) + raw.float() + paired.float()).to(dtype)[normal],
        )
        checks += 1

        # Exercise every contributor count, duplicate slots, mapping tile
        # boundaries, and both sides of the register-path chunk boundary.
        for hidden in (63, 64, 65, 6144, 10815, 10816, 10817):
            tokens, topk = (num_aiv * 129 + 1 if hidden == 63 else 137), 8
            mapping = torch.full((tokens, topk), -1, dtype=torch.int32)
            for token in range(tokens):
                for k in range(token % 9):
                    mapping[token, k] = (token * 3 + k // 2) % 19
            x = ((torch.arange(19 * hidden).reshape(19, hidden) % 61 - 30).float() / 16).to(dtype)
            # Cancellation and a small contribution expose reassociated sums.
            x[0] = 16384
            x[1] = -16384
            x[2] = 0.03125
            reference = torch.zeros((tokens, hidden), dtype=torch.float32)
            for k in range(topk):
                valid = mapping[:, k] >= 0
                reference[valid] += x[mapping[valid, k].long()].float()
            actual, _ = native.aggregate(x.to(device), None, None, mapping.to(device), num_aiv, None)
            _equal(actual, reference.to(dtype))
            checks += 1
    return dict(checks=checks, num_aiv=num_aiv, single_route_all_bits=digests, two_route_all_bits=two_route_digests)
