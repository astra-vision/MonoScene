# MonoScene: Monocular 3D Semantic Scene Completion

**MonoScene: Monocular 3D Semantic Scene Completion**] \[[arXiv + supp](https://arxiv.org/abs/2112.00726)\] | \[[Project page](https://cv-rits.github.io/MonoScene/)\] \
[Anh-Quan Cao](https://anhquancao.github.io),
[Raoul de Charette](https://team.inria.fr/rits/membres/raoul-de-charette/)  
Inria, Paris, France

If you find this work useful, please cite our [paper](https://arxiv.org/abs/2112.00726):
```
@misc{cao2021monoscene,
      title={MonoScene: Monocular 3D Semantic Scene Completion}, 
      author={Anh-Quan Cao and Raoul de Charette},
      year={2021},
      eprint={2112.00726},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Teaser


|SemanticKITTI | KITTI-360 <br/>(Trained on SemanticKITTI) |
|:------------:|:------:|
|<img src="./teaser/SemKITTI.gif"  />|<img src="./teaser/KITTI-360.gif" />|


<p align="center">
  <b>NYUv2</b>
</p>
<p align="center">
  <img src="./teaser/NYUv2.gif" style="width:48%"/>
</p>

# Table of Content
- [Preparing MonoScene](#preparing-monoscene)
  - [Installation](#installation)  
  - [Datasets](#datasets)
  - [Pretrained models](#pretrained-models)
- [Running MonoScene](#running-monoscene)
  - [Training](#training)
  - [Evaluating](#evaluating)
- [Inference & Visualization](#inference--visualization)
  - [Inference](#inference)
  - [Visualization](#visualization)
- [License](#license)


# Preparing MonoScene

## Installation

1. Create conda environment:

```
$ conda create -y -n monoscene python=3.7
$ conda activate monoscene
```
2. This code was implemented with python 3.7, pytorch 1.7.1 and CUDA 10.2. Please install [PyTorch](https://pytorch.org/): 

```
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the additional dependencies:

```
$ cd MonoScene/
$ pip install -r requirements.txt
```

4. Install tbb:

```
$ conda install tbb
```

5. Finally, install MonoScene:

```
$ pip install -e ./
```


## Datasets


### SemanticKITTI

1. You need to download

      - The **Semantic Scene Completion dataset v1.1** (SemanticKITTI voxel data (700 MB)) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download)
      -  The **KITTI Odometry Benchmark calibration data** (Download odometry data set (calibration files, 1 MB)) and the **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).


2. Create a folder to store SemanticKITTI preprocess data at `/path/to/kitti/preprocess/folder`.

3. Store paths in environment variables for faster access (**Note: folder 'dataset' is in /path/to/semantic_kitti**):

```
$ export KITTI_PREPROCESS=/path/to/kitti/preprocess/folder
$ export KITTI_ROOT=/path/to/semantic_kitti 
```

4. Preprocess the data to generate labels at a lower scale, which are used to compute the ground truth relation matrices:

```
$ cd MonoScene/
$ python monoscene/data/semantic_kitti/preprocess.py kitti_root=$KITTI_ROOT kitti_preprocess_root=$KITTI_PREPROCESS
```

### NYUv2

1. Download the NYUv2 dataset using [this script](https://github.com/shurans/sscnet/blob/master/download_data.sh).

2. Create a folder to store NYUv2 preprocess data at `/path/to/NYU/preprocess/folder`.

3. Store paths in environment variables for faster access:

```
$ export NYU_PREPROCESS=/path/to/NYU/preprocess/folder
$ export NYU_ROOT=/path/to/NYU/depthbin 
```

4. Preprocess the data to generate labels at a lower scale, which are used to compute the ground truth relation matrices:

```
$ cd MonoScene/
$ python monoscene/data/NYU/preprocess.py NYU_root=$NYU_ROOT NYU_preprocess_root=$NYU_PREPROCESS

```

### KITTI-360

1. We only perform inference on KITTI-360. You can download either the **Perspective Images for Train & Val (128G)** or the **Perspective Images for Test (1.5G)** at [http://www.cvlibs.net/datasets/kitti-360/download.php](http://www.cvlibs.net/datasets/kitti-360/download.php).

2. Create a folder to store KITTI-360 data at `/path/to/KITTI-360/folder`.

3. Store paths in environment variables for faster access:

```
$ export KITTI_360_ROOT=/path/to/KITTI-360
```

## Pretrained models

Download MonoScene pretrained models [on SemanticKITTI](https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/monoscene_kitti.ckpt) and [on NYUv2](https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/monoscene_nyu.ckpt), then put them in the folder `/path/to/MonoScene/trained_models`.


# Running MonoScene

## Training

To train MonoScene with SemanticKITTI, type:

### SemanticKITTI

1. Create folders to store training logs at **/path/to/kitti/logdir**.

2. Store in an environment variable:

```
$ export KITTI_LOG=/path/to/kitti/logdir
```

3. Train MonoScene using 4 GPUs with batch_size of 4 (1 item per GPU) on Semantic KITTI:

```
$ cd MonoScene/
$ python monoscene/scripts/train_monoscene.py \
    dataset=kitti \
    enable_log=true \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS\
    kitti_logdir=$KITTI_LOG \
    n_gpus=4 batch_size=4    
```

### NYUv2

1. Create folders to store training logs at **/path/to/NYU/logdir**.

2. Store in an environment variable:

```
$ export NYU_LOG=/path/to/NYU/logdir
```

3.  Train MonoScene using 2 GPUs with batch_size of 4 (2 item per GPU) on NYUv2:
```
$ cd MonoScene/
$ python monoscene/scripts/train_monoscene.py \
    dataset=NYU \
    NYU_root=$NYU_ROOT \
    NYU_preprocess_root=$NYU_PREPROCESS \
    logdir=$NYU_LOG \
    n_gpus=2 batch_size=4

```


## Evaluating 

### SemanticKITTI

To evaluate MonoScene on SemanticKITTI validation set, type:

```
$ cd MonoScene/
$ python monoscene/scripts/eval_monoscene.py \
    dataset=kitti \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS \
    n_gpus=1 batch_size=1
```

### NYUv2

To evaluate MonoScene on NYUv2 test set, type:

```
$ cd MonoScene/
$ python monoscene/scripts/eval_monoscene.py \
    dataset=NYU \
    NYU_root=$NYU_ROOT\
    NYU_preprocess_root=$NYU_PREPROCESS \
    n_gpus=1 batch_size=1
```

# Inference & Visualization

## Inference

Please create folder **/path/to/monoscene/output** to store the MonoScene outputs and store in environment variable:

```
export MONOSCENE_OUTPUT=/path/to/monoscene/output
```

### NYUv2

To generate the predictions on the NYUv2 test set, type:

```
$ cd MonoScene/
$ python monoscene/scripts/generate_output.py \
    +output_path=$MONOSCENE_OUTPUT \
    dataset=NYU \
    NYU_root=$NYU_ROOT \
    NYU_preprocess_root=$NYU_PREPROCESS \
    n_gpus=1 batch_size=1
```

### Semantic KITTI

To generate the predictions on the Semantic KITTI validation set, type:

```
$ cd MonoScene/
$ python monoscene/scripts/generate_output.py \
    +output_path=$MONOSCENE_OUTPUT \
    dataset=kitti \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS \
    n_gpus=1 batch_size=1
```

### KITTI-360

Here we use the sequence **2013_05_28_drive_0009_sync**, you can use other sequences. To generate the predictions on KITTI-360, type:

```
$ cd MonoScene/
$ python monoscene/scripts/generate_output.py \
    +output_path=$MONOSCENE_OUTPUT \
    dataset=kitti_360 \
    +kitti_360_root=$KITTI_360_ROOT \
    +kitti_360_sequence=2013_05_28_drive_0009_sync  \
    n_gpus=1 batch_size=1
```

## Visualization

We use mayavi to visualize the predictions. Please install mayavi following the [official installation instruction](https://docs.enthought.com/mayavi/mayavi/installation.html). Then, use the following commands to visualize the outputs on respective datasets.

### NYUv2 

```
$ cd MonoScene/
$ python monoscene/scripts/visualization/NYU_vis_pred.py +file=/path/to/output/file.pkl
```

### Semantic KITTI 

```
$ cd MonoScene/
$ python monoscene/scripts/visualization/kitti_vis_pred.py +file=/path/to/output/file.pkl +dataset=kitt
```


### KITTI-360

```
$ cd MonoScene/ 
$ python monoscene/scripts/visualization/kitti_vis_pred.py +file=/path/to/output/file.pkl +dataset=kitti_360
```



# License
MonoScene is released under the [Apache 2.0 license](./LICENSE).
