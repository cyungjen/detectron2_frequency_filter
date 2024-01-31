#import the COCO Evaluator to use the COCO Metrics
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
import time
import os 
import logging
logging.basicConfig(filename=f'log/logger.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
'''
Choose a dataset to run mAP
'''
folder = "Traffic_2560x1600_30"
folder_path = os.listdir(f"datasets/filtered_images/{folder}")
for path in folder_path:
    logging.info(path)

    register_coco_instances("my_dataset_test", {}, f"SFU_ground_truth/{folder}_val.json", f"datasets/filtered_images/{folder}/{path}")
    
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

    #Use the created predicted model in the previous step
    start_time = time.time()
    inference_on_dataset(predictor.model, val_loader, evaluator)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    #remove the registered dataset 
    DatasetCatalog.remove("my_dataset_test")
    MetadataCatalog.remove("my_dataset_test")
