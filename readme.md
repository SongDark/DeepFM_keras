# DeepFM in Keras

# Introduction

A simple DeepFM.
See details at [here](https://blog.csdn.net/songbinxu/article/details/80151814).

## Environments

- Keras 2.0.8
- TensorFlow 1.7

## Usage

```python
    python DeepFM.py
``` 

## Reference

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
[paper address](https://arxiv.org/abs/1703.04247)

## Notes

If you meet errors about Embedding Layer, try fix the compute_mask function.

```python
    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        else:
            # return K.not_equal(inputs, 0)
            mask = K.repeat(K.not_equal(inputs, 0), self.output_dim)
            mask = tf.transpose(mask, [0,2,1])
            return mask
```