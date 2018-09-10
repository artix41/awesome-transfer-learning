# Awesome Transfer Learning
A list of awesome papers and cool resources on transfer learning, domain adaptation and domain-to-domain translation in general! As you will notice, this list is currently mostly focused on domain adaptation (DA) and domain-to-domain translation, but don't hesitate to suggest resources in other subfields of transfer learning. I accept pull requests.

# Table of Contents

* [Tutorial and Blogs](#tutorials-and-blogs)
* [Papers](#papers)
  * [Surveys](#surveys)
  * [Deep Transfer Learning](#deep-transfer-learning)
    * [Fine-tuning approach](#fine-tuning-approach)
    * [Feature extraction (embedding) approach](#feature-extraction-embedding-approach)
    * [Policy transfer for RL](#policy-transfer-for-rl)
    * [Few-shot transfer learning](#few-shot-transfer-learning)
    * [Meta transfer learning](#meta-transfer-learning)
    * [Applications](#applications)
  * [Unsupervised Domain Adaptation](#unsupervised-domain-adaptation)
    * [Theory](#theory)
    * [Adversarial methods](#adversarial-methods)
    * [Optimal Transport](#optimal-transport)
    * [Embedding methods](#embedding-methods)
    * [Kernel methods](#kernel-methods)
    * [Autoencoder approach](#autoencoder-approach)
    * [Subspace learning](#subspace-learning)
    * [Self-ensembling methods](#self-ensembling-methods)
    * [Other](#other)
  * [Semi-supervised Domain Adaptation](#semi-supervised-domain-adaptation)
    * [General methods](#general-methods)
    * [Subspace learning](#subspace-learning)
    * [Copulas methods](#copulas-methods)
  * [Few-shot Supervised Domain Adaptation](#few-shot-supervised-domain-adaptation)
    * [Adversarial methods](#adversarial-methods)
    * [Embedding methods](#embedding-methods)
  * [Applied Domain Adaptation](#applied-domain-adaptation)
    * [Physics](#physics)
* [Datasets](#datasets)
  * [Image-to-image](#image-to-image)
  * [Text-to-text](#text-to-text)
* [Results](#results)
  * [Digits transfer](digits-transfer)
* [Challenges](#challenges)
* [Libraries](#libraries)
   

# Tutorials and Blogs 

* [Transfer Learning − Machine Learning's Next Frontier](http://ruder.io/transfer-learning/index.html)
* [A Little Review of Domain Adaptation in 2017](https://artix41.github.io/static/domain-adaptation-in-2017/)

# Papers

Papers are ordered by theme and inside each theme by publication date (submission date for arXiv papers). If the network or algorithm is given a name in a paper, this one is written in bold before the paper's name.

## Surveys

* [A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf) (2009)
* [Transfer Learning for Reinforcement Learning Domains: A Survey](http://www.jmlr.org/papers/volume10/taylor09a/taylor09a.pdf) (2009)
* [A Survey of transfer learning](https://link.springer.com/article/10.1186/s40537-016-0043-6) (2016)
* [Domain Adaptation for Visual Applications: A Comprehensive Survey](https://arxiv.org/pdf/1702.05374.pdf) (2017)
* [Deep Visual Domain Adaptation: A Survey](https://arxiv.org/pdf/1802.03601.pdf) (2018)

## Deep Transfer Learning

Transfer of deep learning models.

### Fine-tuning approach

* [Do Better ImageNet Models Transfer Better?](https://arxiv.org/pdf/1805.08974.pdf) (2018)

### Feature extraction (embedding) approach

* [CNN Features off-the-shelf: an Astounding Baseline for Recognition](https://www.cv-foundation.org//openaccess/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf) (2014)
* [Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/pdf/1804.08328v1.pdf) (2018)

### Multi-task learning

* [Learning without forgetting](https://arxiv.org/pdf/1606.09282) (2016)

### Policy transfer for RL

* [Reinforcement Learning to Train Ms. Pac-Man Using Higher-order Action-relative Inputs](https://www.rug.nl/research/portal/files/19535198/MS_PACMAN_RL.pdf) (2013)

### Few-shot transfer learning

* [Zero-Shot Transfer Learning for Event Extraction](https://arxiv.org/pdf/1707.01066.pdf) (2017)
* [Learning a Deep Embedding Model for Zero-Shot Learning](https://www.eecs.qmul.ac.uk/~sgg/papers/ZhangEtAl_CVPR2017.pdf) (2017)

### Meta transfer learning

* [Transfer Learning via Learning to Transfer](http://proceedings.mlr.press/v80/wei18a/wei18a.pdf) (2018)

### Applications

#### Medical imaging: 

* [Deep Convolutional Neural Networks forComputer-Aided Detection: CNN Architectures, Dataset Characteristics and Transfer Learning](https://arxiv.org/pdf/1602.03409) (2016)
* [Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?](https://arxiv.org/pdf/1706.00712.pdf) (2017)
* [Comparison of deep transfer learning strategies for digital pathology](https://orbi.uliege.be/bitstreaom/2268/222511/1/mormont2018-comparison.pdf) (2018)


#### Robotics

* [A Deep Convolutional Neural Network for Location Recognition and Geometry Based Information](http://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CNN_LOCALIZATION_2018.pdf) (2018)

## Unsupervised Domain Adaptation

Transfer between a source and a target domain. In unsupervised domain adaptation, only the source domain can have labels.

### Theory

#### General

* [A theory of learning from different domains](http://www.alexkulesza.com/pubs/adapt_mlj10.pdf) (2010)

#### Multi-source

* [Domain Adaptation with Multiple Sources](https://papers.nips.cc/paper/3550-domain-adaptation-with-multiple-sources.pdf) (2008)
* [Algorithms and Theory for Multiple-Source Adaptation](https://arxiv.org/pdf/1805.08727.pdf) (2018)

### Adversarial methods

#### Learning a latent space

* **DANN**: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/pdf/1505.07818.pdf) (2015)
* **JAN**: [Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/pdf/1605.06636.pdf) (2016)
* **CoGAN**: [Coupled Generative Adversarial Networks](https://arxiv.org/pdf/1606.07536.pdf) (2016)
* **DRCN**: [Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1607.03516.pdf) (2016)
* **DSN**: [Domain Separation Networks](https://arxiv.org/pdf/1608.06019.pdf) (2016)
* **ADDA**: [Adaptative Discriminative Domain Adaptation](https://arxiv.org/pdf/1702.05464.pdf) (2017)
* **GenToAdapt**: [Generate To Adapt: Aligning Domains using Generative Adversarial Networks](https://arxiv.org/pdf/1704.01705.pdf) (2017)
* **WDGRL**: [Wasserstein Distance Guided Representation Learning for Domain Adaptation](https://arxiv.org/pdf/1707.01217.pdf) (2017)
* **CyCADA**: [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](http://proceedings.mlr.press/v80/hoffman18a/hoffman18a.pdf) (2017)
* **DIRT-T**: [A DIRT-T Approach to Unsupervised Domain Adaptation](https://arxiv.org/pdf/1802.08735.pdf) (2017)
* **DupGAN**: [Duplex Generative Adversarial Network for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Duplex_Generative_Adversarial_CVPR_2018_paper.pdf) (2018)
* **MSTN**: [Learning Semantic Representations for Unsupervised Domain Adaptation](http://proceedings.mlr.press/v80/xie18c/xie18c.pdf) (2018)

#### Image-to-Image translation

* **DIAT**: [Deep Identity-aware Transfer of Facial Attributes](https://arxiv.org/pdf/1610.05586.pdf) (2016)
* **Pix2pix**: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) (2016)
* **DTN**: [Unsupervised Cross-domain Image Generation](https://arxiv.org/pdf/1611.02200.pdf) (2016)
* **SimGAN**: [Learning from Simulated and Unsupervised Images through Adversarial Training (2016)](https://arxiv.org/pdf/1612.07828.pdf) (2016)
* **PixelDA**: [Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/pdf/1612.05424.pdf) (2016)
* **UNIT**: [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/pdf/1703.00848.pdf) (2017)
* **CycleGAN**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593) (2017)
* **DiscoGAN**: [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/pdf/1703.05192.pdf) (2017)
* **DualGAN**: [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/pdf/1704.02510.pdf) (2017)
* **SBADA-GAN**: [From source to target and back: symmetric bi-directional adaptive GAN](https://arxiv.org/pdf/1705.08824.pdf) (2017)
* **DistanceGAN**: [One-Sided Unsupervised Domain Mapping](https://arxiv.org/pdf/1706.00826.pdf) (2017)
* **pix2pixHD**: [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf) (2018)
* **I2I**: [Image to Image Translation for Domain Adaptation](https://arxiv.org/pdf/1712.00479.pdf) (2017)
* **MUNIT**: [Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732) (2018)

#### Multi-source adaptation
* **StarGAN**: [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/pdf/1711.09020.pdf) (2017)
* **XGAN**: [XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings](https://arxiv.org/pdf/1711.05139.pdf) (2017)
* **BicycleGAN** : [Toward Multimodal Image-to-Image Translation](https://arxiv.org/pdf/1711.11586.pdf) (2017)
* [Label Efficient Learning of Transferable Representations across Domains and Tasks](https://arxiv.org/pdf/1712.00123.pdf) (2017)
* **ComboGAN**: [ComboGAN: Unrestrained Scalability for Image Domain Translation](https://arxiv.org/pdf/1712.06909.pdf) (2017)
* **AugCGAN**: [Augmented CycleGAN: Learning Many-to-Many Mappings from Unpaired Data](https://arxiv.org/abs/1802.10151) (2018)
* **RadialGAN**: [RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks](https://arxiv.org/abs/1802.06403) (2018)
* **MADA**: [Multi-Adversarial Domain Adaptation](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17067/16644) (2018)
* **MDAN**: [Multiple Source Domain Adaptation with Adversarial Learning](https://arxiv.org/pdf/1705.09684.pdf) (2018)

#### Temporal models (videos)

* **Model F**: [Unsupervised Domain Adaptation for Face Recognition in Unlabeled Videos](https://arxiv.org/pdf/1708.02191.pdf) (2017)
* **RecycleGAN**: [Recycle-GAN: Unsupervised Video Retargeting](https://arxiv.org/pdf/1808.05174.pdf) (2018)
* **Vid2vid**: [Video-to-Video Synthesis](https://arxiv.org/pdf/1808.06601.pdf) (2018)
* **Temporal Smoothing (TS)**: [Everybody Dance Now](https://arxiv.org/pdf/1808.07371.pdf) (2018)

### Optimal Transport

* **OT**: [Optimal Transport for Domain Adaptation](https://arxiv.org/pdf/1507.00504.pdf) (2015)
* [Theoretical Analysis of Domain Adaptation with Optimal Transport](https://arxiv.org/pdf/1610.04420.pdf) (2016)
* **JDOT**: [Joint distribution optimal transportation for domain adaptation](https://arxiv.org/pdf/1705.08848.pdf) (2017)
* **Monge map learning**: [Large Scale Optimal Transport and Mapping Estimation](https://arxiv.org/pdf/1711.02283.pdf) (2017)
* **JCPOT**: [Optimal Transport for Multi-source Domain Adaptation under Target Shift](https://arxiv.org/pdf/1803.04899.pdf) (2018)
* **DeepJDOT**: [DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation](https://arxiv.org/pdf/1803.10081.pdf) (2018)

### Embedding methods

* [Unsupervised Domain Adaptation for Zero-Shot Learning](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kodirov_Unsupervised_Domain_Adaptation_ICCV_2015_paper.pdf) (2015)
* **DA<sub>assoc</sub>** : [Associative Domain Adaptation](https://arxiv.org/pdf/1708.00938.pdf) (2017)

### Kernel methods

* **SurK**: [Covariate Shift in Hilbert Space: A Solution via Surrogate Kernels](https://pdfs.semanticscholar.org/edb8/be020e228153163428e8b698aef1af4c5cad.pdf) (2015)
* **DAN**: [Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/pdf/1502.02791.pdf) (2015)
* **RTN**: [Unsupervised Domain Adaptation with Residual Transfer Networks](https://arxiv.org/pdf/1602.04433.pdf) (2016)
* **Easy DA**: [A Simple Approach for Unsupervised Domain Adaptation](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7899860) (2016)

### Autoencoder approach

* **MCAE**: [Learning Classifiers from Synthetic Data Using a Multichannel Autoencoder](https://arxiv.org/pdf/1503.03163.pdf) (2015)
* **SMCAE**: [Learning from Synthetic Data Using a Stacked Multichannel Autoencoder](https://arxiv.org/pdf/1509.05463.pdf) (2015)

### Subspace Learning

* **SGF**: [Domain Adaptation for Object Recognition: An Unsupervised Approach](https://pdfs.semanticscholar.org/d3ed/bfee56884d2b6d9aa51a6c525f9a05248802.pdf) (2011)
* **GFK**: [Geodesic Flow Kernel for Unsupervised Domain Adaptation](https://pdfs.semanticscholar.org/0a59/337568cbf74e7371fb543f7ca34bbc2153ac.pdf) (2012)
* **SA**: [Unsupervised Visual Domain Adaptation Using Subspace Alignment](https://pdfs.semanticscholar.org/51a4/d658c93c5169eef7568d3d1cf53e8e495087.pdf) (2015)
* **CORAL**: [Return of Frustratingly Easy Domain Adaptation](https://arxiv.org/pdf/1511.05547.pdf) (2015)
* **Deep CORAL**: [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf) (2016)
* **ILS**: [Learning an Invariant Hilbert Space for Domain Adaptation](https://arxiv.org/pdf/1611.08350.pdf) (2016)
* **Log D-CORAL**: [Correlation Alignment by Riemannian Metric for Domain Adaptation](https://arxiv.org/pdf/1705.08180.pdf) (2017)

### Self-Ensembling methods

* **MT**: [Self-ensembling for domain adaptation](https://arxiv.org/pdf/1706.05208.pdf) (2017)

### Other

* [Adapting Visual Category Models to New Domains](https://scalable.mpi-inf.mpg.de/files/2013/04/saenko_eccv_2010.pdf) (2010)
* **AdaBN**: [Revisiting Batch Normalization for Practical Domain Adaptation](https://arxiv.org/pdf/1603.04779.pdf) (2016)

## Semi-supervised Domain Adaptation

All the source points are labelled, but only few target points are.

### General methods

* **da+lap-sim** : [Semi-Supervised Domain Adaptation with Instance Constraints](http://jeffdonahue.com/papers/DAInstanceConstraintsCVPR2013.pdf) (2013)

### Subspace learning

* **EA++**: [Co-regularization Based Semi-supervised Domain Adaptation](https://papers.nips.cc/paper/4009-co-regularization-based-semi-supervised-domain-adaptation.pdf) (2010)
* **SDASL**: [Semi-supervised Domain Adaptation with Subspace Learning for Visual Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yao_Semi-Supervised_Domain_Adaptation_2015_CVPR_paper.pdf) (2015)

### Copulas methods

* **NPRV**: [Semi-Supervised Domain Adaptation with Non-Parametric Copulas](https://papers.nips.cc/paper/4802-semi-supervised-domain-adaptation-with-non-parametric-copulas.pdf) (2013)

## Few-shot Supervised Domain Adaptation

Only a few target examples are available, but they are labelled

### Adversarial methods

* **FADA**: [Few-Shot Adversarial Domain Adaptation](https://arxiv.org/pdf/1711.02536.pdf) (2017)
* **Augmented-Cyc**: [Augmented Cyclic Adversarial Learning for Domain Adaptation](https://arxiv.org/pdf/1807.00374.pdf) (2018)

### Embedding methods

* **CCSA**: [Unified Deep Supervised Domain Adaptation and Generalization](https://arxiv.org/pdf/1709.10190.pdf) (2017)

## Applied Domain Adaptation

Domain adaptation applied to other fields

### Physics

* [Learning to Pivot with Adversarial Networks](http://papers.nips.cc/paper/6699-learning-to-pivot-with-adversarial-networks.pdf) (2016)
* [Adversarial Domain Adaptation for Identifying Phase Transitions](https://arxiv.org/pdf/1710.08382.pdf) (2017)

* [Identifying Quantum Phase Transitions with Adversarial Neural Networks](https://arxiv.org/abs/1710.08382) (2017)

* [Automated discovery of characteristic features of phase transitions in many-body localization](https://arxiv.org/abs/1806.00419) (2017)

### Audio Processing
* [Autoencoder-based Unsupervised Domain Adaptation for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6817520) (2014)
* [Adversarial Teacher-Student Learning for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1804.00644.pdf) (2018)

# Datasets

## Image-to-image

* [MNIST](http://yann.lecun.com/exdb/mnist/) vs [MNIST-M](https://drive.google.com/file/d/0B9Z4d7lAwbnTNDdNeFlERWRGNVk/view) vs [SVHN](http://ufldl.stanford.edu/housenumbers/) vs [Synth](https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view) vs [USPS](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps): digit images
* [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) vs [Syn Signs](http://graphics.cs.msu.ru/en/node/1337) : traffic sign recognition datasets, transfer between real and synthetic signs.
* [NYU Depth Dataset V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html): labeled paired images taken with two different cameras (normal and depth)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): faces of celebrities, offering the possibility to perform gender or hair color translation for instance
* [Office-Caltech dataset](https://people.eecs.berkeley.edu/~jhoffman//domainadapt/): images of office objects from 10 common categories shared by the Office-31 and Caltech-256 datasets. There are in total four domains: Amazon, Webcam, DSLR and Caltech.
* [Cityscapes dataset](https://www.cityscapes-dataset.com/): street scene photos (source) and their annoted version (target)
* [UnityEyes](http://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) vs [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/): simulated vs real gaze images (eyes)
* [CycleGAN datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/): horse2zebra, apple2orange, cezanne2photo, monet2photo, ukiyoe2photo, vangogh2photo, summer2winter
* [pix2pix dataset](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/): edges2handbags, edges2shoes, facade, maps
* [RaFD](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main): facial images with 8 different emotions (anger, disgust, fear, happiness, sadness, surprise, contempt, and neutral). You can transfer a face from one emotion to another.
* [VisDA 2017 classification dataset](http://ai.bu.edu/visda-2017/#browse): 12 categories of object images in 2 domains: 3D-models and real images.
* [Office-Home dataset](http://hemanthdv.org/OfficeHome-Dataset/): images of objects in 4 domains: art, clipart, product and real-world.

## Text-to-text

* [Amazon review benchmark dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/): sentiment analysis for four kinds (domains) of reviews: books, DVDs, electronics, kitchen
* [ECML/PKDD Spam Filtering](http://www.ecmlpkdd2006.org/challenge.html#download): emails from 3 different inboxes, that can represent the 3 domains.
* [20 Newsgroup](http://qwone.com/~jason/20Newsgroups/): collection of newsgroup documents across 6 top categories and 20 subcategories. Subcategories can play the role of the domains, as describe in [this article](https://arxiv.org/pdf/1707.01217.pdf).

# Results

The results are indicated as the prediction accuracy (in %) in the target domain after adapting the source to the target. For the moment, they only correspond to the results given in the original papers, so the methodology may vary between each paper and these results must be taken with a grain of salt.

## Digits transfer (unsupervised)

| Source<br>Target    | MNIST<br>MNIST-M  | Synth<br>SVHN | MNIST<br>SVHN | SVHN<br>MNIST | MNIST<br>USPS | USPS<br>MNIST
| ---                 | ---               | ---           | ---           | ---           | ---           | ---     |
| SA                  | 56.90             | 86.44         | ?             | 59.32         | ?             | ?       |
| DANN                | 76.66             | 91.09         | ?             | 73.85         | ?             | ?       |
| CoGAN               | ?                 | ?             | ?             | ?             | 91.2          | 89.1    |
| DRCN                | ?                 | ?             | 40.05         | 81.97         | 91.80         | 73.67   |
| DSN                 | 83.2              | 91.2          | ?             | 82.7          | ?             | ?       |
| DTN                 | ?                 | ?             | 90.66         | 79.72         | ?             | ?       |
| PixelDA             | 98.2              | ?             | ?             | ?             | 95.9          | ?       |
| ADDA                | ?                 | ?             | ?             | 76.0          | 89.4          | 90.1    |
| UNIT                | ?                 | ?             | ?             | 90.53         | 95.97         | 93.58   |
| GenToAdapt          | ?                 | ?             | ?             | 92.4          | 95.3          | 90.8    |
| SBADA-GAN           | 99.4              | ?             | 61.1          | 76.1          | 97.6          | 95.0    |
| DA<sub>assoc</sub>  | 89.47             | 91.86         | ?             | 97.60         | ?             | ?       |
| CyCADA              | ?                 | ?             | ?             | 90.4          | 95.6          | 96.5    |
| I2I                 | ?                 | ?             | ?             | 92.1          | 95.1          | 92.2    |
| DIRT-T              | 98.7              | ?             | 76.5          | 99.4          | ?             | ?       |
| DeepJDOT            | 92.4              | ?             | ?             | 96.7          | 95.7          | 96.4    |

# Challenges

* [Visual Domain Adaptation Challenge (VisDA)](http://ai.bu.edu/visda-2017/)
* [Open AI Retro Contest](https://blog.openai.com/retro-contest/)

# Libraries

No good library for the moment (as far as I know). If you're interested in a project of creating a generic transfer learning/domain adaptation library, please let me know.
