"""Scale-aware, one-ULP audit for the unchanged first DPO policy/reference."""
import math
import struct


def float32(value):
    return struct.unpack('!f', struct.pack('!f', value))[0]


def float32_ulp_distance(left, right):
    """Exact representable-step distance, treating signed zero as equal.

    Both arguments must already be finite binary32 values from recorded tensors;
    silently rounding arbitrary decimal values would weaken this audit.
    """
    def ordered(value):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite float32 value required')
        rounded = float32(value)
        if rounded != value:
            raise ValueError('Recorded value is not exactly representable as float32')
        bits = struct.unpack('!I', struct.pack('!f', rounded))[0]
        return 0x80000000 - (bits & 0x7fffffff) if bits & 0x80000000 else 0x80000000 + bits
    return abs(ordered(left) - ordered(right))


def inspect_initial_dpo(row, beta=0.1):
    """Require each sequence score within one ULP and consistent DPO arithmetic.

    A scalar margin threshold cannot represent float32 resolution at arbitrary
    summed-log-probability magnitudes. This does not certify identical execution
    paths: checkpoint hashes and immutable reference metadata remain separate.
    """
    details = {'criterion': 'each sequence logp differs by at most one binary32 ULP; recorded ratios, margin, beta scaling and logistic loss agree',
               'maximum_sequence_ulp_distance': 1,
               'legacy_margin_limit': 1e-5,
               'legacy_loss_log2_abs_difference': abs(row.get('loss', math.inf) - math.log(2)),
               'legacy_margin_passed': abs(row.get('relative_margin', math.inf)) < 1e-5}
    try:
        pc, pr = row['policy_chosen_logp'], row['policy_rejected_logp']
        rc, rr = row['reference_chosen_logp'], row['reference_rejected_logp']
        distances = {'chosen': float32_ulp_distance(pc, rc), 'rejected': float32_ulp_distance(pr, rr)}
        policy_ratio, reference_ratio = float32(pc - pr), float32(rc - rr)
        relative = float32(policy_ratio - reference_ratio)
        scaled = float32(beta * relative)
        expected_loss = float32(math.log1p(math.exp(-scaled)))
        expected = {'policy_logratio': policy_ratio, 'ref_logratio': reference_ratio,
                    'relative_margin': relative, 'dpo_scaled_margin': scaled, 'loss': expected_loss}
        arithmetic = {key: float32_ulp_distance(row[key], value) for key, value in expected.items()}
        details.update(sequence_scores={'policy_chosen': pc, 'reference_chosen': rc,
                                        'policy_rejected': pr, 'reference_rejected': rr},
                       sequence_ulp_distances=distances, expected_float32_arithmetic=expected,
                       arithmetic_ulp_distances=arithmetic)
        details['passed'] = all(value <= 1 for value in distances.values()) and all(value <= 1 for value in arithmetic.values())
    except (KeyError, ValueError, OverflowError, struct.error) as error:
        details.update(passed=False, error=type(error).__name__ + ': ' + str(error))
    return details
