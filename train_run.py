from groundingdino.util.train_utils import load_model, load_image,train_image, annotate
import cv2
import os
import json
import csv
import torch
from collections import defaultdict
import torch.optim as optim
from pathlib import Path
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# Model
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
for param in model.parameters():
  param.requires_grad = False
for param in model.backbone.parameters():
  param.requires_grad = True
for param in model.bbox_embed.parameters():
  param.requires_grad = True
weights_Dir = Path('/content/drive/MyDrive/Colab_zip/GroundingDINO/weights')
weights_Dir.mkdir(parents=True, exist_ok=True)
filePaths = [file for file in weights_Dir.iterdir() if file.name.startswith('model_weights')]
try:
  latest_weigths = str(filePaths[-1])
  model.load_state_dict(torch.load(latest_weigths,map_location= device))
  print("choosen weights: ",latest_weigths)
  times = str(int(latest_weigths.split('_')[-2])+1).zfill(2)
except:
  latest_weigths= None
  print("choosen weights: ",latest_weigths)
  times = '00'
# Dataset paths
images_files=sorted(os.listdir("data/images"))
#ann_file="data/annotations/annotation.csv"
ann_file="data/annotations/annotation_sub.csv"

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (numpyarray): input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (str):  Input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)



def read_dataset(ann_file):
    ann_Dict= defaultdict(lambda: defaultdict(list))
    with open(ann_file) as file_obj:
        ann_reader= csv.DictReader(file_obj)  
        # Iterate over each row in the csv file
        # using reader object
        for row in ann_reader:
            #print(row)
            img_n=os.path.join("data/images",row['image_name'])
            x1=int(row['bbox_x'])
            y1=int(row['bbox_y'])
            x2=x1+int(row['bbox_width'])
            y2=y1+int(row['bbox_height'])
            label=row['label_name']
            ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
            ann_Dict[img_n]['captions'].append(label)
    return ann_Dict


def train(model, ann_file, epochs=1,times = times,device= device, save_path= str(weights_Dir/'model_weights'),save_epoch=10):
    # Read Dataset
    ann_Dict = read_dataset(ann_file)
    #model = model.to(device)
    # Add optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Ensure the model is in training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0  # Track the total loss for this epoch
        for idx, (IMAGE_PATH, vals) in enumerate(ann_Dict.items()):
            image_source, image = load_image(IMAGE_PATH)
            bxs = vals['boxes']
            captions = vals['captions']

            # Zero the gradients
            optimizer.zero_grad()
            
            # Call the training function for each image and its annotations
            loss = train_image(
                model=model,
                image_source=image_source,
                image=image,
                caption_objects=captions,
                box_target=bxs,
            )
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()  # Accumulate the loss
            print(f"Processed image {idx+1}/{len(ann_Dict)}, Loss: {loss.item()}")

        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(ann_Dict)}")
        if (epoch%save_epoch)==0:
            epock = str(epoch).zfill(3)
            # Save the model's weights after each epoch
            torch.save(model.state_dict(), f"{save_path}_{times}_{epock}.pth")
            print(f"Model weights saved to {save_path}_{times}_{epock}.pth")



if __name__=="__main__":
    train(model=model, ann_file=ann_file, epochs=50,device= device, save_path=str(weights_Dir/'model_weights'))
