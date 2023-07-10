import torchvision
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

class Vgg19PercLoss(nn.Module):
    def __init__(self, device):
        super(Vgg19PercLoss, self).__init__()
        vgg19 = torchvision.models.vgg19(weights = "DEFAULT")
        vgg19.to(device)
        self.feature_extractor = create_feature_extractor(
            model = vgg19,
            return_nodes = ["features.35"]
        )
        self.feature_extractor.eval()
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        for params in self.feature_extractor.parameters():
            params.requires_grad = False
            
    def forward(self, candidate, target):
        candidate_norm = self.normalize(candidate)
        target_norm = self.normalize(target)

        candidate_output = self.feature_extractor(candidate_norm)["features.35"]
        target_output = self.feature_extractor(target_norm)["features.35"]

        return nn.functional.mse_loss(candidate_output,target_output)
    

class GeneratorLoss(nn.Module):
    def __init__(self, device):
        super(GeneratorLoss, self).__init__()

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pixelwise_loss = nn.functional.mse_loss
        self.perceptual_loss = Vgg19PercLoss(device).forward

    def forward(self, true_label, predicted_label, sr_image, hr_image):
        adv_loss = self.adversarial_loss(predicted_label, true_label)
        pixel_loss = self.pixelwise_loss(sr_image, hr_image)
        perceptual_loss = self.perceptual_loss(sr_image, hr_image)
        return {
            "adversarial_loss": adv_loss,
            "pixel_loss": pixel_loss,
            "perceptual_loss": perceptual_loss
        }