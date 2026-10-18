# 20-Class Image Classification with Deep Learning

A deep learning project developed for FIT3181 at Monash University Malaysia, exploring different convolutional neural networks and pretrained vision transformers for multi-class image classification.

The project compares several architectures and fine-tuning strategies, including ResNet18, EfficientNet-B3, CLIP, SigLIP2, and Perception Encoder (PE).

**Official final Kaggle score: 0.95110 (95.11%)**

**Best additional experimental score: 0.96927 (96.927%)**

## 1. Project Overview

The objective was to classify images into 20 categories, including animals and everyday objects.

The project involved training and evaluating multiple models, experimenting with data augmentation and transfer learning, and generating predictions for an unseen Kaggle test set.

A stratified training-validation split was used, with 15% of the labelled training data reserved for validation.

## 2. Models and Experiments

Several architectures were explored to compare conventional convolutional networks with larger pretrained vision transformers.

| Model | Approach | Purpose |
|---|---|---|
| ResNet18 | Transfer learning and fine-tuning | Establish a CNN baseline |
| EfficientNet-B3 | Transfer learning and fine-tuning | Explore an alternative CNN architecture |
| DFN5B CLIP ViT-H/14-378 | Two-stage fine-tuning | Official final submission |
| SigLIP2 Giant | Fine-tuning the final 16 transformer blocks | Explore a larger vision transformer |
| PE-Core-G14-448 | Additional fine-tuning with style augmentation | Investigate improved generalisation |

## 3. Official Final Model — DFN5B CLIP

The official final submission used the pretrained DFN5B CLIP ViT-H/14-378 vision encoder with a custom classification head.

### Architecture

The model consists of:

1. **Pretrained CLIP visual encoder** to extract image features.
2. **LayerNorm** to normalise the extracted features.
3. **Dropout (0.20)** to reduce overfitting.
4. **Linear classification layer** mapping 1,024 features to 20 output classes.

### Training Strategy

Training was divided into two stages.

**Stage 1 — Classification Head Training**

The pretrained encoder was frozen while the new classification head was trained for up to 10 epochs.

**Stage 2 — Partial Encoder Fine-Tuning**

The last eight transformer blocks, final encoder normalisation layer, and classification head were made trainable. Different learning rates were applied to the encoder and classifier.

AdamW, label smoothing, and cosine annealing learning-rate scheduling were used during training.

The best checkpoint was selected using validation performance.

### Data Augmentation

The final model used random resized cropping, horizontal flipping, colour jittering, and CLIP-specific image normalisation.

### Result

The final model achieved a **Kaggle leaderboard score of 0.95110 (95.11%)** and was selected for the official assignment submission.

## 4. Additional Experiment — Perception Encoder

An additional experiment explored the PE-Core-G14-448 vision transformer with style-aware data augmentation.

Unlike the official CLIP model, this experiment started from a previously fine-tuned PE checkpoint and performed another round of training.

### Style-Aware Augmentation

A custom `RandomStyleVariation` transformation was implemented to expose the model to different image appearances.

The augmentation randomly applied:

- Original image appearance (60%)
- Grayscale conversion (15%)
- Posterization (10%)
- Automatic contrast enhancement (5%)
- Sketch-like transformation (10%)

These transformations were combined with random resized cropping, horizontal flipping, and colour jittering.

The objective was to investigate whether greater visual diversity during training could improve generalisation.

### Additional Fine-Tuning

The experiment fine-tuned the final eight transformer blocks and classification head for five additional epochs, using separate learning rates for the pretrained encoder and classifier.

Mixed-precision training, gradient accumulation, and gradient checkpointing were used to support training the large model.

### Result

The experiment achieved a **Kaggle score of 0.96927 (96.927%)**, higher than the official final submission.

However, this model was explored separately and was **not used in the official assignment submission or report**.

Because the model architecture and training approach differed from the official submission, this result should not be interpreted as the effect of augmentation alone.

## 5. Other Experiments

**ResNet18:** Used pretrained convolutional features with classifier training followed by fine-tuning of the final residual block.

**EfficientNet-B3:** Explored a more parameter-efficient convolutional architecture, including classifier training and feature extractor fine-tuning.

**SigLIP2 Giant:** Investigated a larger pretrained vision transformer by training a classification head and subsequently fine-tuning the final 16 transformer blocks.

These experiments helped compare different pretrained architectures and approaches to adapting them to the classification task.

## 6. Technologies

- Python
- PyTorch and Torchvision
- OpenCLIP
- Hugging Face Transformers and Hub
- Facebook Perception Encoder
- NumPy and pandas
- Scikit-learn
- Matplotlib
- Google Colab
- Kaggle

## 7. Repository Contents

The repository includes the official final Kaggle notebook and selected experimental notebooks for the different architectures.

The original dataset and trained model checkpoints are not included.

Some notebooks depend on pretrained weights, Google Colab paths, or previously saved checkpoints. In particular, the PE experiment requires an earlier fine-tuned PE checkpoint to reproduce its additional training stage.

## 8. Key Takeaways

This project provided practical experience in transfer learning, model fine-tuning, computer vision, data augmentation, validation-based model selection, and GPU training.

Comparing different architectures demonstrated how pretrained vision models can be adapted to a specialised image classification task. The additional PE experiment also explored the potential of style-aware augmentation for improving generalisation.

**Note:** The project uses third-party pretrained model architectures and weights. The training, classification, and experimental workflows are documented in the accompanying notebooks.
