# iMet2019-FGVC6

**To run this solution**:

0. put train/test files in the same file structure as required in kaggle kernels
1. change variable MODEL_DIR to be the directory where you put se-resnext50 weights file
2. change variable MODEL_NAME to be the name of se-resnext50 weights file 
3. run :)

**LB journey**:

| model         | LB    |
| ------------- |:--------------------------:|
| Vgg16 with no augmentation | 0.507 |
| Resnet50 + aug       | 0.565 |
| Se-resnext50 + aug + 288*288 + (bs==32)  | 0.593  |
| Se-resnext50 + aug + 288*288 + (bs==64) | 0.604 |
| Se-resnext50 + aug + 288*288 + (bs==64) + TTA  | 0.615 |
| Se-resnext50 + aug + 288*288 + (bs==64) + TTA + 5fold | 0.635 |


What worked for me:  
* a good combination of learning rate and batch size.  
* larger image size  
* average prediction from multiple folds
* test time augmentation


What didn't work for me:
* Se-resnext101 + 288 + (bs==40): kind of interesting, my guess is that the batch size is too low.
* focal loss: somehow my validation f2 score became unstable and could not converge better than that of BCE loss.
* default Fastai Se-resnext50 structure.
