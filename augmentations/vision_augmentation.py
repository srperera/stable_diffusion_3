import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2


#############################################################################################
# Function: Get Image Processor
#############################################################################################
def get_image_processor(img_size):
    return transforms.Compose(
        [
            transforms_v2.RGB(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
