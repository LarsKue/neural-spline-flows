import FrEIA.framework as ff
import FrEIA.modules as fm
import torch.nn as nn

from additive import AdditiveCoupling
from distributions import StandardNormal
from .base import BaseFlow

from spline import RationalQuadraticSpline


class DenseFlow(BaseFlow):
    """
    Dense Flow used for 1D input
    """

    @property
    def default_hparams(self):
        return super().default_hparams | dict(
            input_shape=(2,),
            steps=4,
            subnet_widths=[64, 64],
            coupling_type="affine",
            coupling_args={},
            distribution="normal",
            dropout=0.0,
        )

    def configure_inn(self):
        nodes = []
        input_node = ff.InputNode(*self.hparams.input_shape, name=f"Input")
        nodes.append(input_node)

        for step in range(self.hparams.steps):
            match self.hparams.coupling_type.lower():
                case "additive":
                    coupling = ff.Node(
                        inputs=nodes[-1],
                        module_type=AdditiveCoupling,
                        module_args=dict(
                            subnet_constructor=self.configure_subnet,
                            **self.hparams.coupling_args,
                        ),
                        name=f"Additive({step})"
                    )
                case "affine":
                    coupling = ff.Node(
                        inputs=nodes[-1],
                        module_type=fm.GLOWCouplingBlock,
                        module_args=dict(
                            subnet_constructor=self.configure_subnet,
                            **self.hparams.coupling_args,
                        ),
                        name=f"Affine({step})"
                    )
                case "spline":
                    coupling = ff.Node(
                        inputs=nodes[-1],
                        module_type=RationalQuadraticSpline,
                        module_args=dict(
                            subnet_constructor=self.configure_subnet,
                            **self.hparams.coupling_args,
                        ),
                        name=f"Spline({step})"
                    )
                case _:
                    raise ValueError(f"Unsupported Coupling Type: {self.hparams.coupling_type}")

            nodes.append(coupling)

        output_node = ff.OutputNode(nodes[-1], name=f"Output")
        nodes.append(output_node)

        return ff.GraphINN(nodes)

    def configure_subnet(self, in_features, out_features):

        widths = self.hparams.subnet_widths

        subnet = nn.Sequential()

        input_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=widths[0]),
            nn.ReLU(),
        )

        subnet.add_module("input_layer", input_layer)

        for i in range(1, len(widths)):
            layer = nn.Sequential(
                nn.Dropout(p=self.hparams.dropout),
                nn.Linear(in_features=widths[i - 1], out_features=widths[i]),
                nn.ReLU(),
            )

            subnet.add_module(f"hidden_layer_{i}", layer)

        output_layer = nn.Linear(in_features=widths[-1], out_features=out_features)

        subnet.add_module("output_layer", output_layer)

        # initialize such that the coupling performs an identity transform
        subnet[-1].weight.data.fill_(0.0)
        subnet[-1].bias.data.fill_(0.0)

        return subnet

    def configure_distribution(self):
        match self.hparams.distribution:
            case "normal":
                distribution = StandardNormal(self.hparams.input_shape)
            case _:
                raise ValueError(f"Unsupported Distribution: {self.hparams.distribution}")
        return distribution
