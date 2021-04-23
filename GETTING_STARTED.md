## Data Preparation

### COCO dataset
1. Download and extract COCO 2017 train and val images with annotations from [http://cocodataset.org](http://cocodataset.org).
2. Download panoptic annotations from COCO website.
3. Download the pre-processing panopitc segmentation annotations from here([Google Drive](https://drive.google.com/file/d/1Yej9MyoPH9B3N7HfWNyG4U_yRHoUknV5/view?usp=sharing)).
4. prepare the data as the following structure:

```
detectron2/
  datasets/
    coco/
      {train,val}2017/
      panoptic_{train,val}2017/  # png annotations
      annotations/
        panoptic_{train,val}2017.json
        panoptic_{train,val}2017_trans/  # pre-processing panoptic segmentation png annotations
```

### ADE20K dataset
1. Download and extract the ADE20K dataset train and val images from [http://sceneparsing.csail.mit.edu/](http://sceneparsing.csail.mit.edu/).
2. Download the annotations for panoptic segmentation from here ([Google Drive](https://drive.google.com/file/d/1bFQ9rpG2raxhQSgTk0vqyujcvSHW0QmZ/view?usp=sharing)).
3. prepare the data as the following structure:
```
detectron2/
  datasets/
    ADE20K_2017/
      images/
        training/
        validation/
      new_segment_anno_continuous/
        training/
        validation/
      ade_{train,val}_things_only.json
      panoptic_ade20k_val_iscrowd.json
```

## Getting Started

### Training & Evaluation in Command Line

To train a model, first ```cd detectron2``` and then 
run 
```
python tools/train_net.py --num-gpus 4 \
  --config-file ./configs/ADE/panoptic_fpn_bs8_R_50_1x_G.yaml
```

To eval a model, use
```
./train_net.py \
  --config-file  ./configs/ADE/panoptic_fpn_bs8_R_50_1x_G.yaml \
  --eval-only MODEL.WEIGHTS /path/to/model_weights
```

### Models

**Model weights**

|Datasets |Google Drive|
|--------|--------------|
|ADE20K |[Download link](https://drive.google.com/file/d/16ScCWm4lZJvz6gL5e7i27apbA9C22g5J/view?usp=sharing) |
|COCO |[Download link](https://drive.google.com/file/d/16ScCWm4lZJvz6gL5e7i27apbA9C22g5J/view?usp=sharing) |