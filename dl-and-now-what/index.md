---
title: "Hello Deep Learning: Further reading & worthwhile projects"
date: 2023-03-30T12:00:09+02:00
draft: false
---
> This page is part of the [Hello Deep Learning](../hello-deep-learning) series of blog posts. You are very welcome to improve this page [via GitHub](https://github.com/berthubert/hello-dl-posts/blob/main/dl-and-now-what/index.md)!

After having completed this series of blogposts (well done!) you should have a good grounding in what deep learning is actually doing. However, this was of course only a small 20k word introduction, so there is a lot left to learn.

Unfortunately, there is a lot of nonsense online. Either the explanations are sloppy or they are just plain wrong. 

Here is an as yet pretty short list of things I've found to be useful. I very much hope to hear from readers about their favorite books and sites. You can send [pull requests directly](https://github.com/berthubert/hello-dl-posts/blob/main/dl-and-now-what/index.md) or email me on bert@hubertnet.nl

Sites:
 * The [PyTorch documentation](https://pytorch.org/docs/stable/index.html) is very useful, even if you are not using PyTorch. It describes pretty well how many layers work exactly.
 * [Andrej Karpathy](https://twitter.com/karpathy)'s [micrograd](https://github.com/karpathy/micrograd) Python autogradient implementation is a tiny work of art
 * Andrej Karpathy's post [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), and also [this post](https://karpathy.github.io/2019/04/25/recipe/)
 * [FastAI](https://fast.ai)'s Jupyter notebooks.

Projects:
 * [Whisper.cpp](https://github.com/ggerganov/whisper.cpp), by hero worker [Georgi Gerganov](https://ggerganov.com/). An open source self-contained C++ version of OpenAI's whisper speech recognition model. You can run this locally on very modest hardware and it is incredibly impressive. Because the source code is so small it is a great learning opportunity.
 * [Llama.cpp](https://github.com/ggerganov/llama.cpp), again by Georgi, a C++ version of Meta's Llama "small" large language model that can run on reasonable hardware. Uses quantisation to fit in normal amounts of memory. If prompted well, the Llama model shows ChatGPT-like capabilities.
 
