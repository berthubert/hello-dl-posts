---
title: "Hello Deep Learning"
date: 2023-03-17T13:58:02+01:00
draft: true
---
<center>
<video width="100%" autoplay loop muted>
    <source src="learning.mp4"
            type="video/mp4">
    Sorry, your browser doesn't support embedded videos.
</video>
</center>

Very much a work in progress. Mission statement:

A from scratch GPU-free introduction to modern machine learning. Many tutorials exist already of course, but this one aims to really explain what is going on. Also, we'll develop the demo until it is actually useful on **real life** data which you can supply yourself.

Other documents start out from the (very impressive) PyTorch environment, or they attempt to math it up from first principles.

Trying to understand deep learning via PyTorch is like trying to learn aerodynamics from flying an Airbus A380.

Meanwhile the pure maths approach ("see it is easy, it is just a Jacobian matrix") is probably only suited for seasoned mathematicians.

The goal of this tutorial is to develop modern neural networks entirely from scratch, but where we still end up with really impressive results.

[Code is here](https://github.com/berthubert/hello-dl). 

Chapters:

 * [Introduction](https://github.com/berthubert/hello-dl/blob/main/README.md#hello-dl)
 * [Chapter 1: Linear combinations](https://berthub.eu/tmp/hello-dl/hello-deep-learning-chapter1)
 * [Chapter 2: Some actual learning, backward propagation](https://berthub.eu/tmp/hello-dl/first-learning)
 * [Chapter 3: Automatic differentiation](https://berthub.eu/tmp/hello-dl/autograd)
 * [Chapter 4: Recognizing handwritten digits using a multi-layer network: batch learning SGD](https://berthub.eu/tmp/hello-dl/handwritten-digits-sgd-batches)
 * [Chapter 5: Neural disappointments, convolutional networks, recognizing handwritten **letters**](https://berthub.eu/tmp/hello-dl/dl-convolutional/)
 * [Chapter 6: Inspecting and plotting what is going on, hyperparameters, momentum, ADAM](https://berthub.eu/tmp/hello-dl/hyperparameters-inspection-adam)
 * [Chapter 7: Dropout, data augmentation and weight decay, quantisation](https://berthub.eu/tmp/hello-dl/dropout-data-augmentation-weight-decay)
 * [Chapter 8: An actual 1300 line from scratch handwritten letter OCR program](https://berthub.eu/tmp/hello-dl/dl-ocr-demo) (WIP)
 * [Chapter 9: Gated Recurring Unit / LSTM: Some language processing, DNA scanning](https://berthub.eu/tmp/hello-dl/dl-gru-lstm-dna) (WIP)
 * [Chapter 10: What does it all mean?](https://berthub.eu/tmp/hello-dl/dl-what-does-it-all-mean) (WIP)
