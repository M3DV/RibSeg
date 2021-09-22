# RibSeg

## Intro

Manual rib inspections in computed tomography (CT) scans are clinically critical but labor-intensive, as 24 ribs are typically elongated and oblique in 3D volumes. Automatic rib segmentation methods can speed up the process through rib measurement and visualization. However, prior arts mostly use in-house labeled datasets that are publicly unavailable and work on dense 3D volumes that are computationally inefficient. To address these issues, we develop a labeled rib segmentation benchmark, named RibSeg, including 490 CT scans (11,719 individual ribs) from a public dataset. For ground truth generation, we used existing morphology-based algorithms and manually refined its results. Then, considering the sparsity of ribs in 3D volumes, we thresholded and sampled sparse voxels from the input and designed a point cloud-based baseline method for rib segmentation. The proposed method achieves state-of-the-art segmentation performance (Dice≈95%≈95%) with significant efficiency (10∼40×10∼40× faster than prior arts). 

## Dataset

The RibSeg Dataset contains annotations for both rib segmentation and centerline.

<img src="D:\MICCAI\RibSeg\readme_pic\10_s.png" style="zoom:33%;" /><img src="D:\MICCAI\RibSeg\readme_pic\10_c.png" style="zoom:33%;" />

Over view of RibSeg dataset:

| Subset      | No. of CT Scans | No. of Individual Ribs |
| ----------- | --------------- | ---------------------- |
| Training    | 320             | 7,670                  |
| Development | 50              | 1,187                  |
| Test        | 120             | 2,862                  |



## Model Training

For training data, please download the source CT scans from RibFrac Dataset to **./data/ribfrac** directory:

### RibFrac Dataset:

training set part1: https://zenodo.org/record/3893508#.YUtisbj0kac 

training set part2: https://zenodo.org/record/3893498#.YUti2bj0kac

test set: https://zenodo.org/record/3993380#.YUti67j0kac

validation set: https://zenodo.org/record/3893496#.YUtjCLj0kac

### RibSeg Dataset

For annotations, download RibSeg dataset to **./data/ribseg** directory

please refer to https://zenodo.org/record/5336592#.YUtkIbj0kac

### Data Preparation

run data_prepare.py to create data for training.

Based on RibFrac dataset and RibSeg dataset, we binarized the CT scans and the annotations for rib segmentation to **./data/pn** for the convenience of training PointNet++. 

### Model 

You can train your model through the following command line:

```
python train_ribseg.py --model pointnet2_part_seg_msg --log_dir <model_directory>
```

You can test your model through the following command line:

```
python test_ribseg.py --log_dir <model_directory>
```

You can conduct inference through the following command line:

```
python inference.py --log_dir <model_directory>
```

You can run our model through the following command line:

```
python inference.py --log_dir c2
```

