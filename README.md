# Towards Skin Cancer Self-Monitoring through an Optimized MobileNet with Coordinate Attention
This project contains the code implemented in the paper *Towards Skin Cancer Self-Monitoring through an Optimized MobileNet with Coordinate Attention*, presented at the 25th Euromicro Conference on Digital System Design (DSD).

# Prerequisites
1. [PyTorch](https://anaconda.org/pytorch/pytorch)
    * Version: 1.9 or above
2. [Torchvision](https://anaconda.org/pytorch/torchvision)
    * Version: 0.12
3. [tensorboard](https://anaconda.org/anaconda/tensorboard)
    * Version: 2.6
4. [tqdm](https://anaconda.org/anaconda/tqdm)
5. [torchinfo](https://anaconda.org/conda-forge/torchinfo)
    * Version: 1.6.5
5. Others:
    * Pandas, Matplotlib, Seaborn


# Modules
* [AttentionMap](https://github.com/SolidusAbi/AttentionMap)
    * Dependenceis: 
        * [Sparse](https://github.com/SolidusAbi/Sparse)

In order to **download the submodules** in the cloning process, use the following instruction:
``` Bash
git clone --recurse-submodules git@github.com:HIRIS-Lab/DermaModelOptimization.git
```
# References

If you find our library useful in your research, please consider citing us:
```
@inproceedings{castro2022towards,
  title={Towards Skin Cancer Self-Monitoring through an Optimized MobileNet with Coordinate Attention},
  author={Castro-Fernandez, Maria and Hernandez, Abian and Fabelo, Himar and Balea-Fernandez, Francisco J and Ortega, Samuel and Callico, Gustavo M},
  booktitle={2022 25th Euromicro Conference on Digital System Design (DSD)},
  pages={607--614},
  year={2022},
  organization={IEEE Computer Society},
  keywords = {performance evaluation;neural networks;graphics processing units;melanoma;computer architecture;sensitivity and specificity;skin},
  doi = {10.1109/DSD57027.2022.00087},
  url = {https://doi.ieeecomputersociety.org/10.1109/DSD57027.2022.00087}
}
```

# TODO
* [X] Dataset definition
* [ ] Architecture design
    * With and without attention blocks
* [ ] InvertedResidual documentation
