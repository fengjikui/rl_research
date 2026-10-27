import unittest
import torch
from core import clipped_loss, group_advantages
from guard import pressure


class Checks(unittest.TestCase):
    def test_groups_do_not_mix_tasks(self):
        a, fraction = group_advantages(torch.tensor([0., 1., 0., 1., 5., 5., 5., 5.]), 4)
        torch.testing.assert_close(a[4:], torch.zeros(4))
        self.assertAlmostEqual(float(a[:4].sum()), 0.)
        self.assertEqual(fraction, .5)

    def test_old_policy_and_advantage_are_frozen(self):
        new = torch.tensor([[0., 0.]], requires_grad=True)
        old = torch.tensor([[0., 0.]], requires_grad=True)
        a = torch.tensor([1.], requires_grad=True)
        loss = clipped_loss(new, old, a, torch.tensor([[1., 0.]]))
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([[-1., 0.]]))
        self.assertIsNone(old.grad); self.assertIsNone(a.grad)

    def test_thermal_pressure_overrides_low_cpu(self):
        s = dict(thermal_state=0, cpu_percent=5., available_gib=12., task_rss_gib=2.)
        self.assertFalse(pressure(s))
        self.assertTrue(pressure(dict(s, thermal_state=1)))
        self.assertTrue(pressure(dict(s, available_gib=5.)))


if __name__ == '__main__': unittest.main()
