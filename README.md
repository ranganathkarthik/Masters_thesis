# RGB-NIR Image Registration for Off-road environments

To obtain a good quality fusion, the images from different spectral bands are to be perfectly aligned. But, the images captured from different spectral bands show high amounts of discrepancy in terms of the information available, positions of the salient feature, and spatial alignment. This misalignment between the images gives rise to a unique challenge in case of off-road environment perception. Correcting this misalignment becomes a deciding factor between a well aligned good quality image fusion and a poorly fused image.

The process of aligning the images from different spectral bands or modalities in order to maximize the usable information is called Image Registration.

The main aim of the thesis was to explore the available methodologies for multi-modal/multispectral image registration, and compare the results of the available classical and deep-learning methods in order to understand the efficacy of the existing techniques, and tailoring it too suit for off-road environments.

Inspired by the extensive use of classical approaches for multi-modal image registration in the field of medicine, and some of the application of multi-spectral image registration in remote sensing and aerial images, we chose Histogram matching, Mutual information optimization, Feature matching, Template matching, and Fourier based approaches to compare their performances. Further, as the exploration can not be complete with out the inclusion of deep learning approaches, we extensively tested NeMAR model and carried out a preliminary exploration of MURF model.

## Quick summary of results

* Some of the classical approaches show a marginal improvement in the quantitative metrics used to evaluate image registration.
* NeMAR model does not show any improvement in quantitative metrics. However, a shift in the images were observed in the initial epochs while training the NeMAR model.
* MURF model, in our observation, struggles to handle non-linear shifts between the RGB and NIR images.


## References:

* NeMAR: https://github.com/moabarar/nemar
* MURF: https://github.com/hanna-xu/MURF
