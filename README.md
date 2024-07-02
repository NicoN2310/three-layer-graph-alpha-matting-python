# Three Layer Graph Alpha Matting in Python

This repository contains a Python implementation of the paper "Three-layer graph framework with the sumD feature for alpha matting" [[1]](#1) as part of my project work at Heinrich-Heine-University Duesseldorf.

## Getting Started

### Requirements

* numpy  >= 2.0.0
* pillow >= 10.3.0
* numba  >= 0.60.0
* scipy  >= 1.14.0

### Example usage via testrun.py

```bash
git clone https://github.com/NicoN2310/three-layer-graph-alpha-matting-python

cd three-layer-graph-alpha-matting-python

python testrun.py
```

This will take the input image and trimap and generate the alpha matte and the resulting cutout. Paths of the files can be set in [testrun.py](testrun.py).

## Trimap Construction

To use the underlying alpha matting algorithm you need a trimap. A trimap is a mask for the image, roughly classifying it into three areas:

* __Foreground:__ Pixels of value __255__ - these pixels will always be included, use it to mark areas where you are sure that they belong to the image part you want to extract
* __Background:__ Pixels of value __0__ - these pixels will always be ignored, use it to mark areas where you are sure that they do not belong to the image part you want to extract
* __Uncertain:__ Pixels of value __128__ - these pixels indicate the "border" region between background and foreground. The alpha matting algorithm will estimate here which pixels belong to the foreground and background.

To create such trimaps for your own images you can use image editing tools like the [GIMP](https://www.gimp.org/downloads/) or you could use the [interactive tool](https://github.com/pymatting/pymatting-interactive-tool) from the [PyMatting Library](https://github.com/pymatting/pymatting).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Many thanks to the authors of the original paper for allowing me to publish my code this way!

## References

<a id="1">[1]</a>
Chao Li, Ping Wang, Xiangyu Zhu, Huali Pi, Three-layer graph framework with the sumD feature for alpha matting, Computer Vision and Image Understanding, Volume 162, 2017, Pages 34-45, ISSN 1077-3142