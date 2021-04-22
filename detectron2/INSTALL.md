## Installation

The codebases are built on top of Detectron2, and need to be built from source.

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.6 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization


### Build Detectron2 from Source

gcc & g++ ≥ 5.4 are required. [ninja](https://ninja-build.org/) is recommended for faster build.
After having them, run:
```
# To install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```

To __rebuild__ detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the
old build first. You often need to rebuild detectron2 after reinstalling PyTorch.


#### Common Installation Issues

Check [offical page](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for common installation issues. 

