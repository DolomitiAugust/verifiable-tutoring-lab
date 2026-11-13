"""Numerical-boundary tests; synthetic rows are not experimental results."""
import math
import struct
import unittest
from .dpo_numeric_audit import float32, float32_ulp_distance, inspect_initial_dpo


def adjacent_negative(value, steps):
    bits = struct.unpack('!I', struct.pack('!f', value))[0]
    return struct.unpack('!f', struct.pack('!I', bits + steps))[0]


def fixture(policy_rejected=-143.99761962890625):
    row = {'policy_chosen_logp': -148.46653747558594,
           'reference_chosen_logp': -148.46653747558594,
           'policy_rejected_logp': policy_rejected,
           'reference_rejected_logp': -143.9976348876953}
    row['policy_logratio'] = float32(row['policy_chosen_logp'] - row['policy_rejected_logp'])
    row['ref_logratio'] = float32(row['reference_chosen_logp'] - row['reference_rejected_logp'])
    row['relative_margin'] = float32(row['policy_logratio'] - row['ref_logratio'])
    row['dpo_scaled_margin'] = float32(0.1 * row['relative_margin'])
    row['loss'] = float32(math.log1p(math.exp(-row['dpo_scaled_margin'])))
    return row


class NumericAuditTests(unittest.TestCase):
    def test_exact_identity(self):
        self.assertTrue(inspect_initial_dpo(fixture(-143.9976348876953))['passed'])

    def test_one_sequence_ulp_with_consistent_math(self):
        result = inspect_initial_dpo(fixture())
        self.assertTrue(result['passed'])
        self.assertFalse(result['legacy_margin_passed'])
        self.assertEqual(result['sequence_ulp_distances'], {'chosen': 0, 'rejected': 1})

    def test_two_sequence_ulps_rejected_even_with_consistent_loss(self):
        row = fixture(adjacent_negative(-143.9976348876953, -2))
        self.assertFalse(inspect_initial_dpo(row)['passed'])

    def test_inconsistent_margin_rejected(self):
        row = fixture(); row['relative_margin'] = 0.0
        self.assertFalse(inspect_initial_dpo(row)['passed'])

    def test_inconsistent_loss_rejected(self):
        row = fixture(); row['loss'] = float32(0.69)
        self.assertFalse(inspect_initial_dpo(row)['passed'])

    def test_nonfinite_rejected(self):
        row = fixture(); row['policy_chosen_logp'] = float('nan')
        self.assertFalse(inspect_initial_dpo(row)['passed'])

    def test_missing_fields_rejected(self):
        self.assertFalse(inspect_initial_dpo({'loss': math.log(2), 'relative_margin': 0})['passed'])

    def test_unrepresentable_decimal_rejected(self):
        with self.assertRaises(ValueError): float32_ulp_distance(0.1, float32(0.1))

    def test_signed_zero_and_cross_zero(self):
        self.assertEqual(float32_ulp_distance(-0.0, 0.0), 0)
        self.assertEqual(float32_ulp_distance(-(2**-149), 2**-149), 2)

    def test_ulp_is_scale_aware(self):
        for value in (-1.0, -128.0, -1024.0):
            self.assertEqual(float32_ulp_distance(value, adjacent_negative(value, 1)), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
