---
title: "Hello Deep Learning"
date: 2023-03-29T11:59:00+02:00
draft: false
---
<center>
<video width="100%" autoplay loop muted playsinline>
    <source src="learning.mp4"
            type="video/mp4">
    Sorry, your browser doesn't support embedded videos.
</video>
</center>

A from scratch GPU-free introduction to modern machine learning. Many tutorials exist already of course, but this one aims to really explain what is going on, from the ground up. Also, we'll develop the demo until it is actually useful on **real life** data which you can supply yourself.

Other documents start out from the (very impressive) PyTorch environment, or they attempt to math it up from first principles.
Trying to understand deep learning via PyTorch is like trying to learn aerodynamics from flying an Airbus A380.

Meanwhile the pure maths approach ("see it is easy, it is just a Jacobian matrix") is probably only suited for seasoned mathematicians.

The goal of this tutorial is to develop modern neural networks entirely from scratch, but where we still end up with really impressive results.

[Code is here](https://github.com/berthubert/hello-dl). Markdown for blogposts can [also be found on GitHub](https://github.com/berthubert/hello-dl-posts) so you can turn typos into pull requests (thanks!).

Chapters:

 * [Introduction](../hello-deep-learning-intro) (which you can skip if you want)
 * [Chapter 1: Linear combinations](../hello-deep-learning-chapter1)
 * [Chapter 2: Some actual learning, backward propagation](../first-learning)
 * [Chapter 3: Automatic differentiation](../autograd)
 * [Chapter 4: Recognizing handwritten digits using a multi-layer network: batch learning SGD](../handwritten-digits-sgd-batches)
 * [Chapter 5: Neural disappointments, convolutional networks, recognizing handwritten **letters**](../dl-convolutional/)
 * [Chapter 6: Inspecting and plotting what is going on, hyperparameters, momentum, ADAM](../hyperparameters-inspection-adam)
 * [Chapter 7: Dropout, data augmentation and weight decay, quantisation](../dropout-data-augmentation-weight-decay)
 * [Chapter 8: An actual 1700 line from scratch handwritten letter OCR program](../dl-ocr-demo) 
 * Chapter 9: Gated Recurring Unit / LSTM: Some language processing, DNA scanning
 * Chapter 10: Attention, transformers, how does this compare to ChatGPT?
 * [Chapter 11: Further reading & worthwhile projects](../dl-and-now-what)
 * Chapter 12: What does it all mean?


<!--  
 * [Chapter 9: Gated Recurring Unit / LSTM: Some language processing, DNA scanning](../dl-gru-lstm-dna) (WIP)
 * [Chapter 10: Attention, transformers, how does this compare to ChatGPT?](../dl-attention-transformers-chatgpt) (nothing yet) -->
 * [Chapter 12: What does it all mean?](../dl-what-does-it-all-mean) (WIP)
