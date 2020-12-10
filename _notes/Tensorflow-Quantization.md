---
layout: note
name: Tensorflow Quantization
type: misc
date: September 15, 2020
---

**Quantization Aware Training**

Fundamentals of quantization aware training.

Why do we need this? To optimize ML models, make them smaller and faster. Make models more efficient to execute - faster compute, lower memory/disk/battery.

ML model is a dataflow graph of different tensor computations. Has static and dynamic parameters. Quantization reduces the static parameters to lower resolution e.g. floating 32bit to int8. The interactions between static and dynamic parameters are also executed in lower precision. 

Maps from `-3e38<x<3e38` to `-127<x<127` uniformly.

Scale = max - min / 2^bits.

Quantized_value = float_value / scale

Float_value = quantized_value * scale

But quantization is lossy. Inference losses happen when fussing the ReLUs.

QAT - training time technique to improve acc of quantized models. It introduces **inference time quantization** errors during training, so model learns robust parameters. Make training path as similar as inference path. So as to mimic errors that will occur during inference, in training itself. 

To mimic errors, we can use FakeQuant. In forward pass of training, we quantize the input tensors to lower precision, the convert them back to float, which introduces the quantization loss. It also aligns the floating point numbers with int8 buckets. 

We also model inference path. Inference optimization may fuse ReLu activations, or fold batchnorm to conv layers. QAT applies the same transforms to training. 

To quantize an entire model - 
```
import tensorflow_model_optimization as tfmot
quantized_model = tfmot.quantization.keras.quantize_model(model)
```

To quantize specific layers -
```
quantize_annotate_layer = tumor.quantization.keras.quantize_annotate_layer
quantized_model = tfmot.quantization.keras.quantize_apply(model)

quantize_annotate_layer(conv2d(), quantize_config=QConfig())
```

***Model Transforms*** - to make training path mimic inference path.

Summary - flexible keras API allows easy experimentation, simulating quantization loss on diff backends/algorithms. 

***Keras Layer Lifecycle***

Layer: represents a NN layer

Model - defines a network of layers

Wrapper - a wrapper that encapsulates a layer and allows us to apply modifications to layers i.e. inject ops before or after construction of layer.

__init__ constructs Layer ‘objects.
`build(input_shape)` - lazily constructs necessary TF vars
`call(inputs)` - constructs TF graph

Init called when models is defined. Build called when model is initialized. Call called when model is fit e.g. 

```
class ClipWrapper(Wrapper):
    def build(self, input_shape):
        self.layer.build(input_shape)
        self.min = self.add_weight(‘min’, initializer=-6)
        self.max = self.add_weight(‘max’, initializer=6)

    def call(self, inputs):
        x = tf.keras.backend.clip(inputs, self.min, self.max)
        return self.layer.call(x)
```

