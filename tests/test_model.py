import unittest

import torch

from traffic_detection.model import PacketCNN


class PacketCNNTests(unittest.TestCase):
    def test_forward_and_backward(self):
        model = PacketCNN(num_classes=7)
        packets = torch.zeros((3, 30), dtype=torch.int16)
        packets[0, :4] = torch.tensor([1448, -31, -1448, 92], dtype=torch.int16)
        packets[1, :3] = torch.tensor([-100, -200, 64], dtype=torch.int16)
        logits = model(packets)
        self.assertEqual(tuple(logits.shape), (3, 7))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
