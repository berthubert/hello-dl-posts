---
title: "Deep Learning: Doing some actual OCR on handwritten characters"
date: 2023-03-25T15:49:25+01:00
draft: true
---
The previous chapters have often mentioned the chasm between "my deep learning neural networks works on my data" and "it actually works in the real world". It is perhaps for this reason that almost all demos and youtube tutorials you find online never do any real world testing.

Here, we are going to do it, and this will allow us to experience first hand how hard this is. We're going to build a computer program that reads handwritten letters from a provided photo.

Here is our input image, which already has a story behind it:

<center>

![](cleaned.jpeg)

<p></p>
</center>

When I first got the OCR program working, results were very depressing. The network struggled mightily on some letters, often just not getting them right. Whatever I did, the 'h' would not work for example. First I blamed my own sloppy handwriting, but then I studied what the network was trained on:

<center>

![](h-poster.png)

<p></p>
</center>

Compare this to how I (& many other Europeans) write an h:
<center>

![](h.png)

<p></p>
</center>

No amount of training on the MNIST set is going to teach a neural network to consistently recognize this as an h - it is simply not in the training set.

So that was the first lesson - really be aware of what is in your training set. If it is different from what you thought, results might very well disappoint. To move on, I changed the test image to something that uses handwriting like what is actually in the MNIST data.

# Practicalities
What we have is a network that does pretty well on 28 by 28 pixel representations of letters, where the background pixel value is 0. By contrast, the input image is millions of pixels, in full colour even. 

The first thing to do is to turn the image into a black and white gray scale version, where we also adjust the white balance so black is actually black and where the gray that passes for white is actually white.

From OCR theory I learned that the first step in character segmentation is to recognize lines of text. This is done by making a graph of the total intensity per horizontal line of the image. From this graph, you then try to select intervals of high intensity that look like they might represent a line of text.

For each line, you then travel from left to right to try to box in characters.

TBC


