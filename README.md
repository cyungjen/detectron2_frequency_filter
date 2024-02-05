# 設定root
export PYTHONPATH="${PYTHONPATH}:/root to detectron2_main"
***
# inference 1張照片
python demo.py --config-file configs/COCO-InstanceSegmentation/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
  --input input1.jpg \
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/faster_rcnn_X_101_32x8d_FPN_3x/137849600/model_final_68b088.pkl
***
# 資料集路徑
## img
/datasets/source_images/sequence name/imgs/
### e.g.
/datasets/source_images/BasketballDrive_1920x1080_50/imgs/
## annotation
/annotation folder/
***
# 生成不同bitrate應對的圖片資料夾
## Entropy coding使用RLC->Huffman coding
python detectron2/test_all_QF_bitrate_RLC.py
## Entropy coding使用triplet->Huffman coding
python detectron2/test_all_QF_bitrate_triplet.py
***
# Run mAP
python detectron2/run_mAP.py
***
# Draw ground truth
python detectron2/draw_GroundThruth.py
