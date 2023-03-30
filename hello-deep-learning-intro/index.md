---
title: "Hello Deep Learning: Intro"
date: 2023-03-30T12:00:00+02:00
draft: false
---
> This page is part of the [Hello Deep Learning](../hello-deep-learning) series of blog posts. Also, feel free to skip this intro and [head straight for chapter 1](../hello-deep-learning-chapter1) where the machine learning begins! 

Deep learning and 'generative AI' have now truly arrived. If this is a good thing very much remains to be seen. What is certain however is that these technologies will have a huge impact.

Up to late 2022, I had unwisely derided the advances of deep learning as overhyped nonsense from people doing fake demos. Turned out this was only half false - many of the demos were indeed fake.

But meanwhile, truly staggering things were happening, and I had ignored all of that. In hindsight, I wish I had read and believed Andrej Karpathy's incredibly important 2015 post [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). The examples in there are self-contained proof something very remarkable had been discovered.

For me this meant I had to catch up and figure out what was going on. What is this magical stuff really? Soon I found myself in a maze of confusing YouTube videos and Jupyter notebooks that showed me awesome things, but that did not address how all this magic worked. Also, quite often when trying to reproduce what I had seen, the magic did not actually work.

To make up for my somewhat idiotic ignorance, I went back to first principles to emulate a bit of what Andrej Karpathy had achieved: I set out to build a a self-contained, simple, but still impressive demo of the technologies involved, one that would really showcase this awesome new technology, including its pitfalls.

The goal is to really start from the ground up. Many other projects will tell you how to use the impressive deep learning tooling that is now available. This project hopes to show you what this tooling is actually doing for you to make the magic happen. And not only show: we're going to start truly from scratch - this is not built on top of PyTorch or TensorFlow. It it built on top of plain C++. 

In the chapters of this 'Hello Deep Learning' project, we'll build several solutions that do actually impressive things. The first solution is a relatively small from scratch program that will learn how to recognize handwritten letters, and also perform this feat on actual real life data -- something many projects conveniently skip.

Along the way we'll cover many of the latest deep learning techniques, and employ them in our little programs.

In this project, the 'from scratch' part means that we'll only be depending on system libraries, [a logging library](https://berthub.eu/articles/posts/big-data-storage/), [a matrix library](https://en.wikipedia.org/wiki/Eigen_(C%2B%2B_library)) and [an image processing library](https://github.com/nothings/stb). It serves no educational purpose to develop any of these things as part of this series. Yet, we will spend time on what the matrix library is doing for us, and why you should not ever roll your own.

I hope you'll enjoy this trip through the fascinating world of deep learning. It has been my personal way of making up for years of ignorance, and with some luck, this project will not only have been useful for me.

Finally, all pages are [hosted on github](https://github.com/berthubert/hello-dl-posts) and I very much look forward to receiving your pull requests to fix my inevitable mistakes or fumbled explanations!

Now, do head on to [Chapter 1: Linear combinations](../hello-deep-learning-chapter1).
