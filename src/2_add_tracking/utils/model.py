from torch import Tensor, nn


class ConvNet(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config
        self.conv1 = nn.Conv2d(
            self.model_config["conv1"]["in_channels"],
            self.model_config["conv1"]["out_channels"],
            kernel_size=self.model_config["conv1"]["kernel_size"],
            padding=self.model_config["conv1"]["padding"],
        )
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(self.model_config["max_pool"])
        self.conv2 = nn.Conv2d(
            self.model_config["conv2"]["in_channels"],
            self.model_config["conv2"]["out_channels"],
            kernel_size=self.model_config["conv2"]["kernel_size"],
            padding=self.model_config["conv2"]["padding"],
        )
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(self.model_config["max_pool"])
        self.fc = nn.Linear(
            self.model_config["fc"]["in_features"],
            self.model_config["fc"]["out_features"],
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.reshape(-1, self.model_config["fc"]["in_features"])
        out = self.fc(out)
        return out
