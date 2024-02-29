from torch import nn
from model.functions import ReverseLayerF

class FeatureExtractor(nn.Module):
    def __init__(self, in_channel=8):
        super(FeatureExtractor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=16, stride=4, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fla = nn.Flatten()

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fla(x)
        return x


class ClassClassifier(nn.Module):
    def __init__(self):
        super(ClassClassifier, self).__init__()
        self.l1 = nn.Linear(256, 1024)
        self.l2 = nn.Linear(1024, 5)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.l1 = nn.Linear(256, 1024)
        self.l2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Sharenet = FeatureExtractor()
        self.GasClassification = ClassClassifier()
        self.DomainDiscrimination = DomainClassifier()

    def forward(self, sou_data, tar_data, alpha):
        sou_feature = self.Sharenet(sou_data)
        sou_reverse_feature = ReverseLayerF.apply(sou_feature, alpha)
        sou_class_output = self.GasClassification(sou_feature)
        sou_domain_output = self.DomainDiscrimination(sou_reverse_feature)

        tar_feature = self.Sharenet(tar_data)
        tar_reverse_feature = ReverseLayerF.apply(tar_feature, alpha)
        tar_class_output = self.GasClassification(tar_feature)
        tar_domain_output = self.DomainDiscrimination(tar_reverse_feature)

        return sou_class_output, sou_domain_output, tar_domain_output

    def predict(self, x):
        x = self.Sharenet(x)
        label_cla = self.GasClassification(x)
        return label_cla