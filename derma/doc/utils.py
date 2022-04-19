import numpy as np
import pandas as pd
import seaborn as sns
from torch import Tensor
from torch.nn import Module
from captum.attr import GuidedGradCam, LayerGradCam, LayerAttribution
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

def GradCamAttribute(model: Module, layer: Module, inputs: Tensor, target: Tensor):
    model.eval()
    guided_gc = GuidedGradCam(model, layer)
    guided_gc.relu_attributions = False

    attribution = guided_gc.attribute(inputs, target)

    return attribution


def plot_attribution(attribution: np.ndarray, img: np.ndarray, figsize=(8,8)) -> Figure:
    x, y = np.meshgrid(np.arange(attribution.shape[1]), np.arange(attribution.shape[0]))
    df = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'att': attribution.ravel()})
    f, ax = plt.subplots(1, figsize=figsize)
    sns.kdeplot(x=df['x'], y=df['y'], weights=df['att'], cmap='mako', shade=True, bw_adjust=0.5, alpha=0.5, thresh=0, ax=ax)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(''), ax.set_ylabel('')
    ax.imshow(img)

    return f