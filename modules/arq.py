
import FrEIA.framework as ff
import FrEIA.modules as fm

from .dense import DenseFlow
from .image import ImageFlow


class DenseARQ(DenseFlow):
    @property
    def default_hparams(self):
        return super().default_hparams | dict(
            affine_steps=5,
            spline_steps=1,
        )

    def configure_inn(self):
        nodes = []
        input_node = ff.InputNode(*self.hparams.input_shape, name=f"Input")
        nodes.append(input_node)

        for step in range(self.hparams.steps):
            for i in range(self.hparams.affine_steps):
                node = ff.Node(
                    inputs=nodes[-1],
                    module_type=fm.GLOWCouplingBlock,
                    module_args=dict(
                        subnet_constructor=self.configure_subnet,
                        **self.hparams.coupling_args,
                    ),
                    name=f"Affine({step}:{i})"
                )

                nodes.append(node)

            for i in range(self.hparams.spline_steps):
                node = ff.Node(
                    inputs=nodes[-1],
                    module_type=fm.RationalQuadraticSpline,
                    module_args=dict(
                        subnet_constructor=self.configure_subnet,
                        **self.hparams.spline_args,
                    ),
                    name=f"Spline({step}:{i})"
                )

                nodes.append(node)

        output_node = ff.OutputNode(nodes[-1], name=f"Output")
        nodes.append(output_node)

        return ff.GraphINN(nodes)


class ImageARQ(ImageFlow):
    @property
    def default_hparams(self):
        return super().default_hparams | dict(
            affine_steps=5,
            spline_steps=1,
        )

    def configure_inn(self):
        nodes = []
        input_node = ff.InputNode(*self.hparams.input_shape, name=f"Input")
        nodes.append(input_node)

        for step in range(self.hparams.steps):
            for i in range(self.hparams.affine_steps):
                node = ff.Node(
                    inputs=nodes[-1],
                    module_type=fm.GLOWCouplingBlock,
                    module_args=dict(
                        subnet_constructor=self.configure_subnet,
                        **self.hparams.coupling_args,
                    ),
                    name=f"Affine({step}:{i})"
                )

                nodes.append(node)

            for i in range(self.hparams.spline_steps):
                node = ff.Node(
                    inputs=nodes[-1],
                    module_type=fm.RationalQuadraticSpline,
                    module_args=dict(
                        subnet_constructor=self.configure_subnet,
                        **self.hparams.spline_args,
                    ),
                    name=f"Spline({step}:{i})"
                )

                nodes.append(node)

        output_node = ff.OutputNode(nodes[-1], name=f"Output")
        nodes.append(output_node)

        return ff.GraphINN(nodes)
