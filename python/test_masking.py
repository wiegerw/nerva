#!/usr/bin/env python3

from unittest import TestCase

import torch

from testing.masking import create_mask


class TestMasking(TestCase):
    def test_mask(self):
        W = torch.zeros(100, 200)
        density = 0.23
        non_zero_count = int(density * W.numel())
        mask = create_mask(W, non_zero_count)
        print(mask)

        # Count the number of elements equal to 1
        count = (mask == 1).sum().item()
        self.assertEqual(non_zero_count, count)


if __name__ == '__main__':
    import unittest
    unittest.main()
