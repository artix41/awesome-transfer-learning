# Awesome Transfer Learning
A list of awesome papers and other cool resources on transfer learning and domain adaptation! Don't hesitate to suggest some resources I could have forgotten.

# Tutorials and Blogs

* Transfer Learning − Machine Learning's Next Frontier [\[Blog\]](http://ruder.io/transfer-learning/index.html)

# Papers

Papers are ordered by theme and inside each theme by publication date (submission date for arXiv papers).

## Surveys

* A Survey on Transfer Learning (2009) [\[Paper\]](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)
* A Survey of transfer learning (2016) [\[Paper\]](https://link.springer.com/article/10.1186/s40537-016-0043-6)
* Domain Adaptation for Visual Applications: A Comprehensive Survey (2017) [\[arXiv\]](https://arxiv.org/pdf/1702.05374.pdf)

## General

* Adapting Visual Category Models to New Domains (2010) [\[Paper\]](https://scalable.mpi-inf.mpg.de/files/2013/04/saenko_eccv_2010.pdf)

## Adversarial Domain Adaptation

* Domain-Adversarial Training of Neural Networks (2015) [\[arXiv\]](https://arxiv.org/pdf/1505.07818.pdf)
* Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (2016) [\[arXiv\]](https://arxiv.org/pdf/1607.03516.pdf)
* Domain Separation Networks (2016) [\[arXiv\]](https://arxiv.org/pdf/1608.06019.pdf)
* Unsupervised Cross-domain Image Generation (2016) [\[arXiv\]](https://arxiv.org/pdf/1611.02200.pdf)
* Learning to Pivot with Adversarial Networks (2016) [\[arXiv\]](https://arxiv.org/pdf/1611.01046.pdf)
* Image-to-Image Translation with Conditional Adversarial Networks (2016) [\[arXiv\]](https://arxiv.org/pdf/1611.07004.pdf)
* Learning from Simulated and Unsupervised Images through Adversarial Training (2016) [\[arXiv\]](https://arxiv.org/pdf/1612.07828.pdf)
* Adaptative Discriminative Domain Adaptation (2017) [\[arXiv\]](https://arxiv.org/pdf/1702.05464.pdf)
* Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (2017) [\[arXiv\]](https://arxiv.org/pdf/1703.10593)
* Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (2017) [\[arXiv\]](https://arxiv.org/pdf/1703.05192.pdf)
* DualGAN: Unsupervised Dual Learning for Image-to-Image Translation (2017) [\[arXiv\]](https://arxiv.org/pdf/1704.02510.pdf)
* Generate To Adapt: Aligning Domains using Generative Adversarial Networks (2017) [\[arXiv\]](https://arxiv.org/pdf/1704.01705.pdf)
* From source to target and back: symmetric bi-directional adaptive GAN (2017) [\[arXiv\]](https://arxiv.org/pdf/1705.08824.pdf)
* Adversarial Representation Learning for Domain Adaptation (2017) [\[arXiv\]](https://arxiv.org/pdf/1707.01217.pdf)
* CyCADA: Cycle-Consistent Adversarial Domain Adaptation (2017) [\[arXiv\]](https://arxiv.org/pdf/1707.01217.pdf)

## Optimal Transport

* Theoretical Analysis of Domain Adaptation with Optimal Transport (2016) [\[arXiv\]](https://arxiv.org/pdf/1610.04420.pdf)
* Joint distribution optimal transportation for domain adaptation (2017) [\[arXiv\]](https://arxiv.org/pdf/1705.08848.pdf)

## Kernel Embdedding

* Learning Transferable Features with Deep Adaptation Networks (2015) [\[arXiv\]](https://arxiv.org/pdf/1502.02791.pdf)
* Unsupervised Domain Adaptation with Residual Transfer Networks (2016) [\[arXiv\]](https://arxiv.org/pdf/1602.04433.pdf)

## Autoencoder approach

* Learning Classifiers from Synthetic Data Using a Multichannel Autoencoder (2015) [\[arXiv\]](https://arxiv.org/pdf/1503.03163.pdf)
* Learning from Synthetic Data Using a Stacked Multichannel Autoencoder (2015) [\[arXiv\]](https://arxiv.org/pdf/1509.05463.pdf)


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
