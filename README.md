# DeepLab with PyTorch <!-- omit in toc --> 

This is an unofficial **PyTorch** implementation of **DeepLab v2** [[1](##references)] with a **ResNet-101** backbone. **COCO-Stuff** dataset [[2](##references)] and **PASCAL VOC** dataset [[3]()] are supported. The initial weights (`.caffemodel`) officially provided by the authors are can be converted/used without building the Caffe API. DeepLab v3/v3+ models with the identical backbone are also included (although not tested). [```torch.hub``` is supported](#torchhub).

## Performance

Pretrained models are provided for each training set. Note that the 2D interpolation ways are different from the original, which leads to a bit better results.

### COCO-Stuff

<table>
    <tr>
        <th>Train set</th>
        <th>Eval set</th>
        <th>CRF?</th>
        <th>Code</th>
        <th>Pixel<br>Accuracy</th>
        <th>Mean<br>Accuracy</th>
        <th>Mean IoU</th>
        <th>FreqW IoU</th>
    </tr>
    <tr>
        <td rowspan="3">
            10k <i>train</i> &dagger;<br>
            (<a href='https://drive.google.com/file/d/1Cgbl3Q_tHPFPyqfx2hx-9FZYBSbG5Rhy/view?usp=sharing'>Model</a>)
        </td>
        <td rowspan="3">10k <i>val</i> &dagger;</td>
        <td rowspan="2"></td>
        <td>Original [<a href="#references">2</a>]</td>
        <td><strong>65.1</strong></td>
        <td><strong>45.5</strong></td>
        <td><strong>34.4</strong></td>
        <td><strong>50.4</strong></td>
    </tr>
    <tr>
        <td>Ours</td>
        <td><strong>65.8</td>
        <td><strong>45.7</strong></td>
        <td><strong>34.8</strong></td>
        <td><strong>51.2</strong></td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>Ours</td>
        <td>67.1</td>
        <td>46.4</td>
        <td>35.6</td>
        <td>52.5</td>
    </tr>
    <tr>
        <td rowspan="4">
            164k <i>train</i><br>
            (<a href='https://drive.google.com/file/d/18kR928yl9Hz4xxuxnYgg7Hpi36hM8J2d/view?usp=sharing'>Model</a>)
        </td>
        <td rowspan="2">10k <i>val</i></td>
        <td></td>
        <td>Ours</td>
        <td>68.4</td>
        <td>55.6</td>
        <td>44.2</td>
        <td>55.1</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>Ours</td>
        <td>69.2</td>
        <td>55.9</td>
        <td>45.0</td>
        <td>55.9</td>
    </tr>
    <tr>
        <td rowspan="2">164k <i>val</i></td>
        <td></td>
        <td>Ours</td>
        <td>66.8</td>
        <td>51.2</td>
        <td>39.1</td>
        <td>51.5</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>Ours</td>
        <td>67.6</td>
        <td>51.5</td>
        <td>39.7</td>
        <td>52.3</td>
    </tr>
</table>

&dagger; Images and labels are pre-warped to square-shape 513x513
## Setup

### Requirements

* Python 2.7+/3.6+
* Anaconda environement

Then setup from `conda_env.yaml`. Please modify cuda option as needed (default: `cudatoolkit=10.0`)

```console
$ conda env create -f configs/conda_env.yaml
$ conda activate deeplab-pytorch
```

### Datasets

Setup instruction is provided in each link.

* [COCO-Stuff 10k/164k](data/datasets/cocostuff/README.md)


### Single image

```
Usage: demo.py single [OPTIONS]

  Inference from a single image

Options:
  -c, --config-path FILENAME  Dataset configuration file in YAML  [required]
  -m, --model-path PATH       PyTorch model to be loaded  [required]
  -i, --image-path PATH       Image to be processed  [required]
  --cuda / --cpu              Enable CUDA if available [default: --cuda]
  --crf                       CRF post-processing  [default: False]
  --help                      Show this message and exit.
```

### Several images

```
Usage: demo.py test [OPTIONS]

  Inference from a single folder

Options:
  -c, --config-path FILENAME  Dataset configuration file in YAML  [required]
  -m, --model-path PATH       PyTorch model to be loaded  [required]
  -i, --folder-path PATH      Folder of images to be processed  [required]
  -o, --output-path PATH      Folder to store results [required]
  --cuda / --cpu              Enable CUDA if available [default: --cuda]
  --crf                       CRF post-processing  [default: False]
  --help                      Show this message and exit.
```


## References

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv
paper](https://arxiv.org/abs/1606.00915)

2. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)

1. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)
