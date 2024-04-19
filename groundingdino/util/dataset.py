import json,os
import pandas as pd
import shutil
import random

def print_structure(d, indent=0):
    """Print the structure of a dictionary or list."""

    # If the input is a dictionary
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + str(key))
            print_structure(value, indent+1)

    # If the input is a list
    elif isinstance(d, list):
        print('  ' * indent + "[List of length {} containing:]".format(len(d)))
        if d:
            print_structure(d[0], indent+1)
def cat_replace(id):
  if id in [11,13,14,15]: id = 10
  if id == 16: id =  2
  return id

def annotation_coco_2_pd_converter(
  path_2_ann_json:str,
  path_2_images:str,
  list_class_to_train:list = None,
  list_class_not_to_train:list = None,
  size_per_class:int = 50,
  path_2_csv:str = None,
  path_2_sub_csv:str = None
  ):
    cat_id_2_name = {}
    cat_name_2_id = {}
    img_id_2_width = {}
    img_id_2_height = {}
    img_id_2_name = {}
    img_name_2_id = {}
    df_dicts = {}
    json_file = path_2_ann_json

    with open(json_file, 'r') as openfile:
      # Reading from json file
      ann_json_object = json.load(openfile)

    for imgs in ann_json_object['images']:
      img_id_2_name[imgs['id']] = imgs['file_name']
      img_name_2_id[imgs['file_name']]= imgs['id']
      img_id_2_width[imgs['id']] = imgs['width']
      img_id_2_height[imgs['id']] = imgs['height']
    for cats in ann_json_object['categories']:
      #print(cats['id'],cats['name'])
      cat_name = cats['name']
      cat_id_2_name[cats['id']] = cat_name.lower()
      cat_name_2_id[cat_name.lower()]= cats['id']
    
    df_ann = pd.DataFrame(ann_json_object['annotations'])
    df_ann2 = df_ann.copy()
    df_ann2['image_name'] = df_ann2['image_id'].map(img_id_2_name)
    df_ann2['image_width'] = df_ann2['image_id'].map(img_id_2_width)
    df_ann2['image_height'] = df_ann2['image_id'].map(img_id_2_height)
    df_ann2['label_name'] =  df_ann2['category_id'].map(cat_replace).map(cat_id_2_name)
    df_ann3 = df_ann2.join(pd.DataFrame(df_ann2.bbox.tolist(), index= df_ann2.index,columns=['bbox_x','bbox_y','bbox_width','bbox_height']))
    
    for col in ['bbox_x','bbox_y','bbox_width','bbox_height']:
      df_ann3[col] = df_ann3[col].astype(float).astype(int)
    df_ann4 = df_ann3.loc[:,['label_name','bbox_x','bbox_y','bbox_width','bbox_height','image_name','image_width','image_height']]
    cats = list(df_ann4.label_name.unique())
    #df_ann4['image_name'] = str(df_ann4['image_name']).replace("buildings/batch_11/","")
    if list_class_not_to_train:
      for discard in list_class_not_to_train: cats.remove(discard)
    if list_class_to_train:
      cat = list_class_to_train
    for cat in cats:
      df_sub_cat = df_ann4[df_ann4['label_name'] == cat]
      try: df_dicts[cat] = df_sub_cat.iloc[random.sample(range(0, len(df_sub_cat)), size_per_class),:]
      except:
        #size_per_class_min = min(len(df_sub_cat), size_per_class)
        df_dicts[cat] = df_sub_cat.iloc[random.choices(range(0, len(df_sub_cat)), k = size_per_class),:]
    df_all = pd.concat([df_dicts[cat] for cat in cats], axis=0).sample(frac=1)
    
    if path_2_csv:
      df_ann4.to_csv(path_2_csv,index=False)
    else:
      df_ann4.to_csv('/content/sample_data/annotation.csv',index=False)
    if path_2_sub_csv:
      df_all.to_csv(path_2_sub_csv,index=False)
    else:
      df_ann4.to_csv('/content/sample_data/annotation_sub.csv',index=False)
    return df_ann4.head(10),df_all.head(10),df_ann4.shape

if __name__ == "__main__":
    #model_weights="weights/groundingdino_swint_ogc.pth"
  annotation_coco_2_pd_converter()
