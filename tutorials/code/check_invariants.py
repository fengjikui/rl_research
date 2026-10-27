"""Checks mathematical/data boundaries where a silent bug invalidates learning."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from common import Catch, setup


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pg = load('03_policy_gradient')
lm = load('05_tiny_lm')
pretrained = load('07_pretrained_grpo')


class Invariants(unittest.TestCase):
    def test_real_terminal_and_reset(self):
        env = Catch(2)
        env.ball[:] = [0, 4]; env.player[:] = [4, 0]
        for i in range(5):
            _, r, done, terminal = env.step([0, 2])
            self.assertEqual(bool(done.all()), i == 4)
        np.testing.assert_array_equal(r, [1, 1])
        np.testing.assert_array_equal(env.row, [0, 0])
        self.assertEqual(float(terminal[:, 0, 5].sum()), 2.)

    def test_gae_never_bootstraps_across_terminal(self):
        r = torch.tensor([[0.], [1.]])
        v = torch.zeros_like(r)
        adv, ret = pg.gae(r, v, torch.tensor([[0.], [1.]]), torch.tensor([999.]), 1., 1.)
        torch.testing.assert_close(ret, torch.ones_like(ret))
        torch.testing.assert_close(adv, ret)

    def test_gae_bootstraps_nonterminal_collector_boundary(self):
        _, ret = pg.gae(torch.tensor([[0.]]), torch.tensor([[.2]]),
                        torch.tensor([[0.]]), torch.tensor([.7]), 1., 1.)
        torch.testing.assert_close(ret, torch.tensor([[.7]]))

    def test_causal_mask_prevents_future_leak(self):
        setup(0)
        model = lm.TinyLM().eval()
        a = torch.tensor([[0, 1, 4, 2]])
        b = torch.tensor([[0, 1, 4, 3]])
        torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3])

    def test_eos_mask_includes_first_eos_only(self):
        ids = torch.tensor([[1, 2, 9, 9], [1, 2, 3, 4], [9, 9, 9, 9]])
        expected = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 1.], [1., 0., 0., 0.]])
        torch.testing.assert_close(pretrained.completion_mask(ids, 9), expected)

    def test_reward_and_disjoint_splits(self):
        self.assertEqual(pretrained.exact_reward(' 0123\n', '0123'), 1.)
        self.assertEqual(pretrained.exact_reward('The answer is 0123', '0123'), 0.)
        self.assertEqual(pretrained.exact_reward('123', '0123'), 0.)
        for task in ['reverse', 'add']:
            data = pretrained.make_data(task)
            sets = [{r['id'] for r in data[s]} for s in ['train', 'validation', 'test']]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_ppo_clips_only_the_improving_side(self):
        ratio = torch.tensor([1.5, .5, .5, 1.5], requires_grad=True)
        adv = torch.tensor([1., -1., 1., -1.])
        objective = torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv)
        objective.sum().backward()
        torch.testing.assert_close(ratio.grad, torch.tensor([0., 0., 1., -1.]))


if __name__ == '__main__': unittest.main()
