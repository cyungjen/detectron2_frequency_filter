#import the COCO Evaluator to use the COCO Metrics
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import time
import os 
import logging
import cv2

logging.basicConfig(filename=f'log/logger.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
'''
Choose a dataset to run mAP
'''
dataset_name = 'filtered_images_triplet'
# folder_root = os.listdir(f"datasets/{dataset_name}")
folder_root = ['BlowingBubbles_416x240_50']
for folder in folder_root:
    folder_path = os.listdir(f"datasets/{dataset_name}/{folder}")
    for path in folder_path:
        logging.info(path)

        register_coco_instances("my_dataset_test", {}, f"SFU_ground_truth/{folder}_val.json", f"datasets/{dataset_name}/{folder}/{path}")
        
        #load the config file, configure the threshold value, load weights 
        cfg = get_cfg()
        cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "model_final_68b088.pkl"
        print(cfg.INPUT)

        # Create predictor
        predictor = DefaultPredictor(cfg)
        # print(predictor.model)

        #Call the COCO Evaluator function and pass the Validation Dataset
        evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "my_dataset_test")

        '''
        打開以下程式碼將會繪製預測結果（bbox)到圖片上並儲存
        '''
        '''//////////////////////////繪製預測結果///////////////////////////'''
        # 獲取數據集的元數據（例如，類別名稱）
        # dataset_metadata = MetadataCatalog.get("my_dataset_test")

        # for batch in val_loader:
        #     for data in batch:
        #         img = data["image"].numpy().transpose(1, 2, 0)
        #         file_name = data["file_name"]

        #         # 使用 predictor 預測
        #         outputs = predictor(img)

        #         v = Visualizer(img[:, :, ::-1],
        #                     metadata=dataset_metadata, 
        #                     scale=0.5)
                
        #         # 將預測結果繪制到圖像上
        #         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        #         # 獲取繪制了預測結果的圖像
        #         result_img = out.get_image()[:, :, ::-1]

        #         # save img
        #         if not os.path.exists(f"./output/{dataset_name}/{folder}/{path}"):
        #             os.makedirs(f"./output/{dataset_name}/{folder}/{path}")  
        #         save_path = os.path.join(f"./output/{dataset_name}/{folder}/{path}/", os.path.basename(file_name))
        #         cv2.imwrite(save_path, result_img)

        #         print(f"Processed {file_name}, saved to {save_path}")
        '''//////////////////////////繪製預測結果///////////////////////////'''

        #Use the created predicted model in the previous step
        start_time = time.time()
        inference_on_dataset(predictor.model, val_loader, evaluator)
        end_time = time.time()
        total_time = end_time - start_time
        print(total_time)
        #remove the registered dataset 
        DatasetCatalog.remove("my_dataset_test")
        MetadataCatalog.remove("my_dataset_test")
