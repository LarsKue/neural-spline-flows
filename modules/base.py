import pytorch_lightning as lightning
import torch
from torch.utils.data import DataLoader


class BaseFlow(lightning.LightningModule):
    """
    Defines the base class for flows, trained by lightning
    """

    def __init__(self, train_data=None, val_data=None, test_data=None, **hparams):
        super().__init__()
        # merge hparams with merge operator (python 3.9+)
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams, ignore=["train_data", "val_data", "test_data"])

        self.inn = self.configure_inn()
        self.distribution = self.configure_distribution()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    @property
    def default_hparams(self):
        return dict(
            batch_size=1,
            optimizer="adam",
            learning_rate=1e-3,
            lr_warmup_milestones=[],
            lr_warmup_gamma=10.0,
            lr_milestones=[],
            lr_gamma=0.1,
        )

    def log_likelihood(self, x):
        z, logabsdet = self.inn.forward(x)
        assert isinstance(z, torch.Tensor)

        # print(f"Jacobian Mean: {logabsdet.mean(dim=0)}")

        log_prob = self.distribution.log_prob(z)

        # print(f"Log Prob Mean: {log_prob.mean(dim=0)}")

        return log_prob + logabsdet

    def generate(self, shape=torch.Size((1,)), temperature=1.0):
        """ Generate a number of samples by sampling randomly from the latent distribution """
        z = temperature * self.distribution.sample(shape)
        x, _ = self.inn.forward(z, rev=True)

        return x

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step and log the NLL
        :param batch: input train batch
        :param batch_idx: unused
        :return: the NLL for backpropagation
        """
        nll = -self.log_likelihood(batch).mean(dim=0)
        self.log("training_nll", nll)

        return nll

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step and log the NLL
        :param batch: input validation batch
        :param batch_idx: unused
        :return: None
        """
        nll = -self.log_likelihood(batch).mean(dim=0)
        self.log("validation_nll", nll)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step (call only right before publishing)
        :param batch: input test batch
        :param batch_idx: unused
        :return: None
        """
        nll = -self.log_likelihood(batch).mean(dim=0)
        self.log("test_nll", nll)

    def configure_optimizers(self):
        """
        Configure optimizers and LR schedulers to be used in training
        :return: Dict containing the configuration
        """
        match self.hparams.optimizer:
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
            case _:
                raise ValueError(f"Unsupported Optimizer: {self.hparams.optimizer}")

        lr_warmup = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams.lr_warmup_milestones,
            gamma=self.hparams.lr_warmup_gamma
        )
        lr_step = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams.lr_milestones,
            gamma=self.hparams.lr_gamma
        )
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            lr_warmup,
            lr_step,
        ])

        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def configure_callbacks(self):
        """
        Configure and return train callbacks for Lightning
        """
        return [
            lightning.callbacks.ModelCheckpoint(monitor="validation_nll", save_last=True),
            lightning.callbacks.LearningRateMonitor(),
        ]

    def configure_inn(self):
        """
        Configure and return the inn used by this module
        :return:
        """
        raise NotImplementedError

    def configure_subnet(self, in_features, out_features):
        """
        Configure and return the subnetwork used by couplings to predict [s, t]
        """
        raise NotImplementedError

    def configure_distribution(self):
        """
        Configure and return the latent distribution used by this module
        """
        raise NotImplementedError

    def train_dataloader(self):
        """
        Configure and return the train dataloader
        """
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        """
        Configure and return the validation dataloader
        """
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        """
        Configure and return the test dataloader
        """
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )