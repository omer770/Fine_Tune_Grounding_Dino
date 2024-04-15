from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
from pathlib import Path
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

weights_Dir = Path('weights/')
filePaths = [file for file in weights_Dir.iterdir() if file.name.startswith('model_weights')]
model_weights= str(filePaths[-1])
print("choosen weights: ",model_weights)
def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.3):
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))

    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list


def process_image(
        model_config="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights=model_weights,
        image_path="data/test_images/img.jpg",
        text_prompt="building . pool . carvan .circular stucture",
        #text_prompt=" carvan ",
        box_threshold=0.60,
        text_threshold=0.75
):
    model_weight = 'weights/groundingdino_swint_ogc.pth'
    model = load_model(model_config, model_weight)
    state_dict = torch.load(model_weights,map_location= device)
    model.load_state_dict(state_dict)
    #model.load_state_dict(torch.load(state_dict_path))
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    print(f"Original boxes size {boxes.shape}")
    boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
    print(f"NMS boxes size {boxes.shape}")

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("results/result.jpg", annotated_frame)


if __name__ == "__main__":
    #model_weights="weights/groundingdino_swint_ogc.pth"
    model_weights = model_weights
    process_image(model_weights=model_weights)
