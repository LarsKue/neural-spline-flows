
import torch
import FrEIA.framework as ff

from .dense import DenseFlow


class TrajectoryFlow(DenseFlow):
    def training_step(self, batch, batch_idx):
        z = batch

        all_nlls = 0
        for node in self.inn.node_list:
            if not isinstance(node, (ff.InputNode, ff.OutputNode)):
                (z,), j = node.module.forward((z,))

                nll = -(self.distribution.log_prob(z) + j)

                all_nlls = all_nlls + nll
