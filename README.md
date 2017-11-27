# Awesome Transfer Learning
A list of awesome papers and other cool resources on transfer learning and domain adaptation! Don't hesitate to suggest some resources I could have forgotten.

# Tutorials and Blogs

* [Transfer Learning − Machine Learning's Next Frontier](http://ruder.io/transfer-learning/index.html)

# Papers

Papers are ordered by theme and inside each theme by publication date (submission date for arXiv papers).

## Surveys

* [A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf) (2009)
* [A Survey of transfer learning](https://link.springer.com/article/10.1186/s40537-016-0043-6) (2016)
* [Domain Adaptation for Visual Applications: A Comprehensive Survey](https://arxiv.org/pdf/1702.05374.pdf) (2017)

## General

* [Adapting Visual Category Models to New Domains](https://scalable.mpi-inf.mpg.de/files/2013/04/saenko_eccv_2010.pdf) (2010)

## Adversarial Domain Adaptation

* **DANN**: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/pdf/1505.07818.pdf) (2015)
* **DRCN**: [Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1607.03516.pdf) (2016)
* **DSN**: [Domain Separation Networks](https://arxiv.org/pdf/1608.06019.pdf) (2016)
* **DIAT**: [Deep Identity-aware Transfer of Facial Attributes](https://arxiv.org/pdf/1610.05586.pdf) (2016)
* **DTN**: [Unsupervised Cross-domain Image Generation](https://arxiv.org/pdf/1611.02200.pdf) (2016)
* [Learning to Pivot with Adversarial Networks](https://arxiv.org/pdf/1611.01046.pdf) (2016)
* **Pix2pix**: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) (2016)
* **SimGAN**: [Learning from Simulated and Unsupervised Images through Adversarial Training (2016)](https://arxiv.org/pdf/1612.07828.pdf) (2016)
* **ADDA**: [Adaptative Discriminative Domain Adaptation](https://arxiv.org/pdf/1702.05464.pdf) (2017)
* **CycleGAN**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593) (2017)
* **DiscoGAN**: [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/pdf/1703.05192.pdf) (2017)
* **DualGAN**: [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/pdf/1704.02510.pdf) (2017)
* **GenToAdapt**: [Generate To Adapt: Aligning Domains using Generative Adversarial Networks](https://arxiv.org/pdf/1704.01705.pdf) (2017)
* **SBADA-GAN**: [From source to target and back: symmetric bi-directional adaptive GAN](https://arxiv.org/pdf/1705.08824.pdf) (2017)
* **WDGRL**: [Wasserstein Distance Guided Representation Learning for Domain Adaptation](https://arxiv.org/pdf/1707.01217.pdf) (2017)
* **CyCADA**: [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](https://arxiv.org/pdf/1707.01217.pdf) (2017)
* **StarGAN**: [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/pdf/1711.09020.pdf) (2017)

## Optimal Transport

* [Theoretical Analysis of Domain Adaptation with Optimal Transport](https://arxiv.org/pdf/1610.04420.pdf) (2016)
* **JDOT**: [Joint distribution optimal transportation for domain adaptation](https://arxiv.org/pdf/1705.08848.pdf) (2017)

## Kernel Embdedding

* **DAN**: [Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/pdf/1502.02791.pdf) (2015)
* **RTN**: [Unsupervised Domain Adaptation with Residual Transfer Networks](https://arxiv.org/pdf/1602.04433.pdf) (2016)

## Autoencoder approach

* **MCAE**: [Learning Classifiers from Synthetic Data Using a Multichannel Autoencoder](https://arxiv.org/pdf/1503.03163.pdf) (2015)
* **SMCAE**: [Learning from Synthetic Data Using a Stacked Multichannel Autoencoder](https://arxiv.org/pdf/1509.05463.pdf) (2015)

# Datasets

## Image-to-image

* [MNIST](http://yann.lecun.com/exdb/mnist/) vs [SVHN](http://ufldl.stanford.edu/housenumbers/) vs [USPS](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps): digit images
* [NYU Depth Dataset V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html): labeled paired images taken with two different cameras (normal and depth)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): faces of celebrities, offering the possibility to perform gender or hair color translation for instance
* [Office-Caltech dataset](https://people.eecs.berkeley.edu/~jhoffman//domainadapt/): images of office objects from 10 common categories shared by the Office-31 and Caltech-256 datasets. There are in total four domains: Amazon, Webcam, DSLR and Caltech.
* [Cityscapes dataset](https://www.cityscapes-dataset.com/): street scene photos (source) and their annoted version (target)
* [UnityEyes](http://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) vs [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/): simulated vs real gaze images (eyes)
* [CycleGAN datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/): horse2zebra, apple2orange, cezanne2photo, monet2photo, ukiyoe2photo, vangogh2photo, summer2winter
* [pix2pix dataset](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/): edges2handbags, edges2shoes, facade, maps


## Text-to-text

* [Amazon review benchmark dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/): sentiment analysis for four kinds (domains) of reviews: books, DVDs, electronics, kitchen
* [ECML/PKDD Spam Filtering](http://www.ecmlpkdd2006.org/challenge.html#download): emails from 3 different inboxes, that can represent the 3 domains.
* [20 Newsgroup](http://qwone.com/~jason/20Newsgroups/): collection of newsgroup documents across 6 top categories and 20 subcategories. Subcategories can play the role of the domains, as describe in [this article](https://arxiv.org/pdf/1707.01217.pdf).

# Libraries

No good library for the moment. If you're interested in a project of creating a generic transfer learning/domain adaptation library, please let me know.
