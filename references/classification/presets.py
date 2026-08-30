import cv2
import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode


def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


class VisionTypeTransformation:
    """
    Simulate different types of colour vision.

    The image is converted from RGB to CIELAB colour space, where:
        L* : lightness
        a* : red-green opponent axis
        b* : yellow-blue opponent axis

    Simulated colour deficiencies are obtained by setting one or both
    chromatic channels to their neutral value (128 in OpenCV's LAB
    representation) before converting the image back to RGB.
    """

    def __init__(self, vision_type):
        print("Vision type:", vision_type)
        self.vision_type = vision_type

    def __call__(self, image):
        if self.vision_type == "trichromat":
            return image

        # Convert to HWC uint8 numpy array expected by OpenCV
        if isinstance(image, torch.Tensor):
            # Tensor is CHW uint8 at this point (after PILToTensor, before float conversion)
            img_np = image.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(image)

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

        if self.vision_type == "red-green":
            # Neutral a* channel (no red-green discrimination)
            lab[:, :, 1] = 128
        elif self.vision_type == "yellow-blue":
            # Neutral b* channel (no yellow-blue discrimination)
            lab[:, :, 2] = 128
        elif self.vision_type == "monochromat":
            # Neutral both chromatic channels
            lab[:, :, 1] = 128
            lab[:, :, 2] = 128

        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if isinstance(image, torch.Tensor):
            return torch.from_numpy(rgb).permute(2, 0, 1)
        else:
            from PIL import Image

            return Image.fromarray(rgb)


class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
        vision_type="trichromat",
    ):
        T = get_module(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))

        if backend == "pil":
            transforms.append(T.PILToTensor())
        transforms.append(VisionTypeTransformation(vision_type))
        if backend == "pil":
            # Convert back to PIL for colour augmentations that expect PIL input
            transforms.append(T.ToPILImage())

        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(T.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.extend(
            [
                T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
        vision_type="trichromat",
    ):
        T = get_module(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.PILToTensor())
        transforms.append(VisionTypeTransformation(vision_type))
        if backend == "pil":
            transforms.append(T.ToPILImage())

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
