# MulHiST
**Multiple Histological Staining for Thick Biological Samples via Unsupervised Image-to-Image Translation** (to appear in MICCAI2023) \
Lulin Shi, Yan Zhang, Ivy H. M. Wong, Claudia T. K. Lo, and Terence T. W. Wong \
[Translational and Advanced Bioimaging Laboratory](https://ttwwong.wixsite.com/tabhkust), \
Department of Chemical and Biological Engineering, \
Hong Kong University of Science and Technology, Hong Kong, China 

> Abstract: _The conventional histopathology paradigm can provide the gold standard for clinical diagnosis, which, however, suffers from lengthy processing time and requires costly laboratory equipment. Recent advancements made in deep learning for computational histopathology have sparked lots of efforts in achieving a rapid chemical-free staining technique. Yet, existing approaches are limited to well-prepared thin sections, and invalid in handling more than one stain. In this paper, we present a multiple histological staining model for thick tissues (MulHiST), without any laborious sample preparation, sectioning, and staining process. We use the grey-scale light-sheet microscopy image of thick tissues as model input and transfer it into different histologically stained versions, including hematoxylin and eosin (H&E), Masson’s trichrome (MT), and periodic acid-Schiff (PAS). This is the first work that enables the automatic and simultaneous generation of multiple histological staining for thick biological samples. Moreover, we empirically demonstrate that the AdaIN-based generator offers an advantage over other configurations to achieve higher-quality multi-style image generation. Extensive experiments also indicated that multi-domain data fusion is conducive to the model capturing shared pathological features. We believe that the proposed MulHiST can potentially be applied in clinical rapid pathology and will significantly improve the current histological workflow._


# Acknowledgements
The implementation of the whole architecture is from [StarGAN](https://github.com/yunjey/stargan).
