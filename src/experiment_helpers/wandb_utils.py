from collections import deque
from datasets import load_dataset
import itertools
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import wandb

def get_run_id(project_name,file_path):
    run_id=None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.find(f"https://wandb.ai/jlbaker361/{project_name}/runs/")!=-1:
                    run_id=line[line.rfind("/")+1:].strip()
                if line.find("CANCELLED AT")!=-1:
                    print("Cancelled", file_path)
    except:
        pass
    if run_id is None:
        print("couldn't find run id!")
    return run_id

def get_grouping_dict(project_name:str,file_path_format:str,key_value_dict:dict, grouping_keys:list)-> dict:
    assert all(key in key_value_dict for key in grouping_keys)
    grouping_keys.sort()
    relevant_values=[key_value_dict[key] for key in grouping_keys]
    combinations = ["_".join(t) for t in  list(itertools.product(*relevant_values))]
    grouping_dict={
        c:[] for c in combinations
    }

    print(grouping_dict)

    all_values=[]
    for k,v_list in key_value_dict.items():
        all_values.append([(k,v) for v in v_list])
    all_combinations=list(itertools.product(*all_values))
    all_combo_dicts=[
        {key:value for (key,value) in combination} for combination in all_combinations
    ]
    
    for combo_dict in all_combo_dicts:
        #print(combo_dict)
        file_path=file_path_format
        for key,value in combo_dict.items():
            file_path=file_path.replace("{"+key+"}",value)
        run_id=get_run_id(project_name,file_path)
        
        group_id="_".join([combo_dict[key] for key in grouping_keys])
        print('\t',file_path,run_id,group_id)
        grouping_dict[group_id].append(run_id)
    
    return grouping_dict

def make_tables(run_id_dict:dict,key_list:list,project:str):
    api=wandb.Api(timeout=60)
    for name,run_id_list in run_id_dict.items():
        metric_dict={
            key:[] for key in key_list
        }
        for run_id in run_id_list:
            try:
                run=api.run(f"jlbaker361/{project}/{run_id}")
                for key in key_list:
                    try:
                        history=run.history(keys=[key])
                        metric_dict[key].append(history[key][0])
                    except:
                        print(f"couldnt find key {key} for name {name} and {run_id}")
            except:
                pass
        #print(metric_dict)
        print(name.replace("_", " "), "&", " & ".join([f"{round(np.mean(metric_dict[key]),4)}" for key in metric_dict.keys()]),"\\\\")


def make_grid(project_name:str,
              methods_to_target_dict:dict,
              target_column:str, 
              image_column:str,
              size:tuple,
              suffix_list:list,
              gd:dict,
              prompt_id:str)->list:
    api=wandb.Api(timeout=60)
    grid=[]
    data=load_dataset(f"jlbaker361/{project_name}",split="train")
    for method,target_list in methods_to_target_dict.items():
        for target in target_list:
            new_row=[]
            for line in data:
                if line[target_column]==target:
                    new_row.append(line[image_column].resize(size))
                    break
            for suffix in suffix_list:
                key=f"{method}_{suffix}"
                run_id_list= gd[key]
                for normal_run in run_id_list:
                    run=api.run(f"jlbaker361/{project_name}/{normal_run}")
                    for file in run.files():
                        if file.mimetype=="image/png":
                            query_string=f"media/images/{target}/{method}_{prompt_id}"
                            if file.name.find(query_string)!=-1:
                                image_path=file.name
                                print(query_string, image_path)
                                file.download(exist_ok=True)
                                with Image.open(image_path) as img:
                                    # Resize image
                                    img = img.resize(size)
                                    new_row.append(img)
                                    break
            grid.append(new_row)
            print(target, new_row)
    return grid

def create_image_grid_with_labels(images, row_labels, col_labels, image_size, label_font_size=20,margin=10, horiz_margin=50):
    """
    Create a grid of images with axis labels.

    :param images: 2D list of PIL Image objects.
    :param row_labels: List of labels corresponding to the rows of images.
    :param col_labels: List of labels corresponding to the columns of images.
    :param image_size: Tuple (width, height) representing the size of each image.
    :param label_font_size: Font size for the labels.
    :return: PIL Image object representing the grid.
    """
    # Find unique row and column labels
    unique_row_labels = row_labels
    unique_col_labels = col_labels

    # Create mapping from labels to grid positions
    row_map = {label: idx for idx, label in enumerate(unique_row_labels)}
    col_map = {label: idx for idx, label in enumerate(unique_col_labels)}

    # Calculate the grid size
    num_rows = len(unique_row_labels)
    num_cols = len(unique_col_labels)

    img_width, img_height = image_size

    # Calculate the total size including labels
    label_padding = label_font_size + margin
    grid_width = (num_cols) * (img_width+margin) + label_padding+horiz_margin
    grid_height = (num_rows )* (img_height+margin) + label_padding

    # Create a new blank image with a white background
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid_img)

    # Load a default fon
    try:
        font= ImageFont.truetype("times_new_roman.ttf", label_font_size)
    except:
        font = ImageFont.load_default()
    # Place each image in the correct position based on the labels
    for i in range(len(images)):
        for j in range(len(images[i])):
            
            
            img = images[i][j]
            print("\t",img)
            # Get the position from labels
            row_idx = i
            col_idx = j
            # Calculate the position where the image will be pasted
            x = col_idx * (img_width +margin) + label_padding +horiz_margin
            y = row_idx * (img_height+margin) + label_padding
            # Resize the image if it's not the correct size
            img = img.resize(image_size)
            # Paste the image into the grid
            grid_img.paste(img, (x, y))
            print(x,y)

    # Draw labels for columns
    for col_idx,label in enumerate(col_labels):
        x = col_idx * (img_width+margin) + label_padding + img_width // 2 +horiz_margin //2
        y = margin  # Position for the column label
        draw.text((x, y), str(label), fill='black', font=font, anchor='mm')

    # Draw labels for rows
    for row_idx,label in enumerate(row_labels):
        x = horiz_margin  # Position for the row label
        y = row_idx * (img_height+margin) + label_padding + img_height // 2
        draw.text((x, y), str(label), fill='black', font=font, anchor='mm')

    return grid_img