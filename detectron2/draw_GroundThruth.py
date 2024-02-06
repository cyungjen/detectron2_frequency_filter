import json
import cv2
import os
filename_root = os.listdir(f"datasets/source_images")
for filename in filename_root:
    with open(f"SFU_ground_truth/{filename}_val.json") as f:
        annotations = json.load(f)


    categories = {category['id']: category['name'] for category in annotations['categories']}

    output_folder = f"output/ground_truth/{filename}/"
    os.makedirs(output_folder, exist_ok=True) 

    for image_info in annotations['images']:
        image_id = image_info['id']
        image_path = f"datasets/source_images/{filename}/imgs/" + image_info['file_name']
        
        image = cv2.imread(image_path)

        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

        
        for ann in image_annotations:
            bbox = ann['bbox']  
            x, y, w, h = map(int, bbox)
            category_id = ann['category_id']
            category_name = categories[category_id]
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  
        output_path = os.path.join(output_folder, image_info['file_name'].replace('.jpg', '.png'))
        cv2.imwrite(output_path, image)


