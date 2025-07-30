# RGB-NIR Image Registration for Off-road environments

The main aim of the thesis was to explore the available methodologies for multi-modal/multispectral image registration, and compare the results of the available classical and deep-learning methods in order to understand the efficacy of the existing techniques, and tailoring it too suit for off-road environments.

**Image registration** is the process of aligning the images from different spectral bands or modalities in order to maximize the usable information.

Inspired by the extensive use of classical approaches for multi-modal image registration in the field of medicine, and some of the application of multi-spectral image registration in remote sensing and aerial images, we choose Histogram matching, Mutual information optimization, Feature matching, Template matching, and Fourier based approaches to compare their performances. Further, as the exploration can not be complete with out the inclusion of deep learning approaches, we extensively test NeMAR model and carry out a preliminary exploration of MURF model.

## Quick summary of results

* Some of the classical approaches show a marginal improvement in the quantitative metrics to evaluate image registration.
* NeMARmodel does not show any improvement in quantitative metrics. However, a shift in the images were observed in the initial epochs while training the NeMAR model.
* MURF model, in our observation, struggles to handle non-linear shifts between the RGB and NIR images.


## References:

* NeMAR: https://github.com/moabarar/nemar
* MURF: https://github.com/hanna-xu/MURF
