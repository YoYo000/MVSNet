# MVSNet
<p align="center"><img width="100%" src="doc/network.png" /></p>


## About
[MVSNet](https://arxiv.org/abs/1804.02505) is a deep learning architecture for depth map inference from unstructured multi-view images. If you find this project useful for your research, please cite:
```
@article{yao2018mvsnet,
  title={MVSNet: Depth Inference for Unstructured Multi-view Stereo},
  author={Yao, Yao and Luo, Zixin and Li, Shiwei and Fang, Tian and Quan, Long},
  journal={arXiv preprint arXiv:1804.02505},
  year={2018}
}
```

## How to Use

### Installation

* check out the source code ```git clone https://github.com/YoYo000/MVSNet```
* Install cuda 9.0 and cudnn 7.0
* Install Tensorflow and other dependencies by ```sudo pip install -r requirements.txt```

### Training

* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1dsGZxR9vQy4dY3wNzSwUWop-ZmQqouk1/view?usp=sharing) (see the paper), and upzip it as the ``MVS_TRANING`` folder.
* Enter the ``MVSNet/mvsnet`` folder, in ``train.py``, set ``dtu_data_root`` to your ``MVS_TRANING`` path.
* Create a log folder and a model folder in wherever you like to save the training outputs. Set the ``log_dir`` and ``save_dir`` in ``train.py`` correspondingly.
* Train the network ``python train.py``

### Testing

* Download the test data for [scan9](https://drive.google.com/file/d/17ZoojQSubtzQhLCWXjxDLznF2vbKz81E/view?usp=sharing) and unzip it as the ``TEST_DATA_FOLDER`` folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.
* Download the pre-trained MVSNet [model](https://drive.google.com/file/d/1i20LF9q3Pti6YoT1Q-5Li-VNu55SBPQS/view?usp=sharing) and upzip it as ``MODEL_FOLDER``.
* Enter the ``MVSNet/mvsnet`` folder, in ``test.py``, set ``pretrained_model_ckpt_path`` to ``MODEL_FOLDER/model.ckpt``
* Depth map inference for this test data by ``python test.py --dense_folder TEST_DATA_FOLDER``.
* Inspect the .pfm format outputs in ``TEST_DATA_FOLDER/depths_mvsnet`` using ``python visualize.py .pfm``


### Todo

* File formats 
* Post-processing.
