# iMet2019-FGVC6



| model         | LB    |
| ------------- |:--------------------------:|
| Vgg16 no augmentation | 0.507 |
| Resnet50 + aug       | 0.565 |
| Se-resnext50 + aug + 288*288 + (bs==32)  | 0.593  |
| Se-resnext50 + aug + 288*288 + (bs==64) | 0.604 |
| Se-resnext50 + aug + 288*288 + (bs==64) + TTA  | 0.615 |
| Se-resnext50 + aug + 288*288 + 64 + TTA + 5fold | 0.635 |


What worked for me:  
* a good combination of learning rate and batch size.  
* larger image size  
* average prediction from multiple folds


What didn't work for me:
* Se-resnext101 + 288 + (bs==40): kind of interesting, my guess is that the batch size is too low.
* focal loss: somehow my validation f2 score became unstable and could not converge better than that of BCE loss.



Some thoughts about fastai:

I'm new to deep learning, especially Pytorch and Fastai.

Originally I was planning to try this competition through Pytorch and was intimidated a little bit that
Fastai might be too high level and thus not flexible enough.

However, at some point I found some techniques easily available in Fastai but not in valila Pytorch.

1. LR policy like AdamWR, SGDR and one cycle policy. This is what made me decided that I needed to dig deeper into fastai.  
2. For me, Pytorch seems to take longer in preparing data(loading + preprocessing) than Fastai
3. Differential Learning rates(not 100% sure)

Also, what really made me comfortable with using Fastai is the fact that it's much more flexible than I thought. Because it is built upon Pytorch, Fastai has no problem dealing with

1. Pytorch dataloader
2. Pytorch model
3. Custom loss function


Some challenges:

1. Not sure how to levrage augmentation techniques from Fastai and Pytorch at the same time.
2. cnn_learner doesn't accept Pytorch dataloader somehow.(I found a workaround though)

All in all, I appreciate a lot that Fastai team built such a good package that can be used so easily.

Happy kaggling!
