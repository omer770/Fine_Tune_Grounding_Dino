# Grounding DINO Fine-tuning 🦖


We have expanded on the original DINO  repository 
https://github.com/IDEA-Research/GroundingDINO 
by introducing the capability to train the model with image-to-text grounding. This capability is essential in applications where textual descriptions must align with regions of an image. For instance, when the model is given a caption "a cat on the sofa," it should be able to localize both the "cat" and the "sofa" in the image.

## Features:

- **Fine-tuning DINO**: This extension works allows you to fine-tune DINO on your custom dataset.
- **Bounding Box Regression**: Uses Generalized IoU and Smooth L1 loss for improved bounding box prediction.
- **Position-aware Logit Losses**: The model not only learns to detect objects but also their positions in the captions.
- **NMS**: We also implemented phrase based NMS to remove redundant boxes of same objects


## Installation:
See original Repo for installation of required dependencies essentially we need to install prerequisits and 
