[toc]

## Number of experiments

- 5 baseline
- 6 siamese
- 12 MIL
- 15 Single Axial 

Total = 5+6+12+15 = 38

5 folds => 190 trials





## Experiments Hyperparameters

Stratified 5-fold cross validation:

```
Dataset size 465
 - Number of class 1 patients 137 29.46%
 - Number of class 0 patients 328 70.54%
Test: 60
 - Number of class 1 patients 17
 - Number of class 0 patients 43
Test set relative size 12.9%
Validation set: 405
 - Number of class 1 patients 120
 - Number of class 0 patients 285
Validation set relative size 87.1%
Fold size: 81.0
 - Number of class 1 patients 24.0
 - Number of class 0 patients 57.0
Train size (4 folds): 324.0
```

In each fold, the final time metrics are computed based on the checkpoint with the highest test fold f1-score 

300 epochs

Adam Optimizer

Learning rate = 0.0005

Weight decay = 0.0001





## Recorded information, for each fold

**Before training** (model summary)

- Type (baseline, siamese, MIL, axial)
- Model name (String)
- Experiment ID (String)
- Encoder
    - Encoder name (String)
    - Pretrained (Boolean)
    - Frozen (Boolean)
    - Output dim (Integer)

**During training** (with tensor board)

- Train, val and test accuracy
- Train, val and test loss
- Train, val and test f1-score
- Log

**After training**

- Best model's weight

- Best model's epoch
- Best model's train, val and test
    - Test accuracy
    - Precision
    - Recall
    - F1-score
    - AUC score
- Best model's prediction (binary and probability) for the test set





## Experiments details

### 0. Baseline experiments

**0.1. 3D CNN followed by logistic regression**

- `0.1.1.baseline-3DCNN-CustomCNN_3D` - CustomCNN 3D
- `0.1.2.baseline-3DCNN-DeepSymNet` - DeepSymNet (without the L-1 norm layer)
- `0.1.3.baseline-3DCNN-ResNet18_3D` - ResNet18 3D
- `0.1.4.baseline-3DCNN-ResNet34_3D` - ResNet34 3D

**0.2. Mirror brain subtracted to itself**

Escolher o melhor encoder

- One of { CustomCNN 3D, DeepSymNet, ResNet18 3D, ResNet34 3D }



### 1. Siamese Net experiments

**1.1. Encodings comparison before `GlobalAveragePooling`**

- `1.1.1.siamese-before-CustomCNN-3D` - CustomCNN 3D
- `1.1.2.siamese-before-DeepSymNet` - DeepSymNet
- `1.1.3.siamese-before-ResNet18-3D` - ResNet18 3D
- `1.1.4.siamese-before-ResNet34-3D` - ResNet34 3D

**1.2. Encodings comparison after `GlobalAveragePooling`**

Escolher o melhor encoder

- One of { CustomCNN 3D, DeepSymNet, ResNet18 3D, ResNet34 3D }

**1.3. Concatenated tangled encodings comparison**

Escolher o melhor encoder

- One of { CustomCNN 3D, DeepSymNet, ResNet18 3D, ResNet34 3D }



### 2. MIL experiments

**2.1. MIL axial slices max**

- `2.1.1.MIL-axial-max-CustomCNN` - CustomCNN
- `2.1.2.MIL-axial-max-ResNet18` - ResNet18
- `2.1.3.MIL-axial-max-ResNet34` - ResNet34
- `2.1.4.MIL-axial-max-EfficientNetB0` - EfficientNet B0
- `2.1.5.MIL-axial-max-EfficientNetB1` - EfficientNet B1

**2.2. MIL axial slices max, ImageNet pretrained**

Escolher o melhor encoder and then

- Frozen weights (only fine-tune)
- Init weights with ImageNet features

**2.3. MIL axial slices mean**

Escolher o melhor encoder e tipo de inicialização

- One of { CustomCNN, ResNet18, ResNet34, EfficientNet B0, EfficientNet B1 }

**2.4. MIL axial slices Ilse Attention**

Escolher o melhor encoder e tipo de inicialização

- One of { CustomCNN, ResNet18, ResNet34, EfficientNet B0, EfficientNet B1 }

**2.4. MIL cubes mean**

- `2.4.1.MIL-cubes-mean-CustomCNN-3D` - CustomCNN

**2.5. MIL cubes max**

- `2.5.1.MIL-cubes-max-CustomCNN-3D` - CustomCNN

**2.6. MIL cubes Ilse Attention**

- `2.6.1.MIL-cubes-attention-CustomCNN-3D` - CustomCNN



### 3. Single Axial Slices experiments

**3.1. ResNet18, ASPECTS height A**

- `3.1.1.Axial-1-A-ResNet18` - 1 axial slice
- `3.1.2.Axial-3-A-ResNet18` - 3 axial slices
- `3.1.3.Axial-5-A-ResNet34` - 5 axial slices
- `3.1.4.Axial-7-A-ResNet18` - 7 axial slices
- `3.1.5.Axial-9-A-ResNet18` - 9 axial slices

**3.2. Average N? slices, ASPECTS height A**

Escolher o melhor número de slices N

- `3.2.1.Axial-5-A-CustomCNN` - CustomCNN
- `3.2.2.Axial-5-A-ResNet18` - ResNet18
- `3.2.3.Axial-5-A-ResNet34` - ResNet34
- `3.2.4.Axial-5-A-EfficientNetB0` - EfficientNet B0
- `3.2.5.Axial-5-A-EfficientNetB1` - EfficientNet B1

**3.4. Average N? slices, ASPECTS height B**

Escolher o melhor encoder e o melhor número de slices N

- One of { CustomCNN, ResNet18, ResNet34, EfficientNet B0, EfficientNet B1 }

**3.5. Average N? slices, cerebelo?**

Escolher o melhor encoder e o melhor número de slices

- One of { CustomCNN, ResNet18, ResNet34, EfficientNet B0, EfficientNet B1 }

**3.6. Ensemble**

- `3.6.1.Axial-ensemble-majority` - Majority vote
- `3.6.2.Axial-ensemble-average` - Average predictions
- `3.6.3.Axial-ensemble-majority` - Weighted average predictions





## Ignored experiments

- Don't use augmentations
- Use other augmentations (combinations of the proposed augmentations or more powerful augmentations like CutMix or MixUp)
- *Visible* target variable
- Don't preprocess data
- Outras alturas dos axial slices
- Aproximar o ensemble method ao MIL, aumentando progressivamente o número de slices
- Constrastive learning
- Comparar resultado do sistema com o de medico
- Boost dos dados tabelares
- Fazer uma coisa semelhante ao feito em *Interpretability-guided Content-based Medical Image Retrieval*
- ViT encoders
