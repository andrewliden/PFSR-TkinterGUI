# Progressive Face Super Resolution
Deokyun Kim, Minseon Kim, Gihyun Kwon, and Dae-shik Kim, [Progressive Face Super-Resolution via Attention to Facial Landmark](https://arxiv.org/abs/1908.08239), The British Machine Vision Conference 2019 (BMVC 2019)

#Modified for simple uses by Andrew B. Liden

### Prerequisites
* Python 3.6
* Pytorch 1.0.0
* CUDA 9.0 or higher

### Data Preparation (only needed for original implementation test)

* [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

create a folder:

```bash
 mkdir dataset

```
and then, download dataset. Anno & Img.


## Original implementation Test

```bash
$ python eval.py --data-path './dataset' --checkpoint-path 'CHECKPOINT_PATH/****.ckpt'
```
<br/>

Simple, single image test.
```bash
$ python simplefilter.py --img-path 'your_image_name_here.jpg'
'''
<br/>
Other available arguments:
--result-path 'your_result_image_name_here.jpg'
--workers N
where N is some integer greater than 0.

TK interface image filtering program:
```bash
$ python filterTK.py



Simplest install procedure:
	Install Anaconda:
		https://www.anaconda.com/distribution/

	Install Torch - Procedure below
		Visit https://pytorch.org/
		Open the anaconda prompt,
		then run the command specified for conda.  
		As of the time of writing, it is:
		conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
