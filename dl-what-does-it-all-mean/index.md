---
title: "Deep Learning: What does it all mean?"
date: 2023-03-11T20:43:44+01:00
draft: true
---
XXX draft not done XXX

In writing this series, I've first hand experienced the 'wow factor' of having your new neural network do impressive things. As part of my alarming lack of focus, I am also a very amateur biologist. I study DNA and evolution, as exhibited for example in [my Nature Scientific Data paper](https://www.nature.com/articles/s41597-022-01179-8).

Microbial life can be completely recreated from its pretty small genome, which is typically a few million DNA letters long. Each DNA letter (A, C, G or T) carries 2 bits of information. A whole bacterium therefore can be regarded as having around a megabyte of parameters. Incidentally, this is of similar size to many interesting neural networks.

Both bacteria and neural networks can evolve new functionality by changing random parameters. For bacteria, we can see this process in action at day long timescales. For example, under lab conditions, a bacterial strain can evolve resistance to an antibiotic within a week. Other more fundamental things take a lot longer, but still happen. For example, in the [E. coli long-term evolution experiment](https://en.wikipedia.org/wiki/E._coli_long-term_evolution_experiment), bacteria took around 33000 generations to evolve a way to live off citrate under aerobic (with oxygen) conditions.

The similarity here is that both networks and life have millions (or billions) of parameters, and that through changes of these, there is a path towards great improvements. 

This is in stark contrast to traditional computer programs, where if you make a change, either nothing happens or your program crashes. There is no random walk imaginable that suddendly adds new features or higher performance to your work.

Now, it is not evident that the gradient descent techniques from neural networks are guaranteed to find interesting minima. But from observation, they very often too. Similarly, life has clearly been extremely successful achieving interesting goals by tweaking millions or billions of parameters. 

Traditional optimizers of simpler functions often get stuck at local minima. But it appears that if you create a solution where you can tweak not just a few parameters but millions of them, it is possible to have a fitness landscape where it is extremely hard to get stuck in a local minimum. Or in other words, even without heroics, your network can wind its way down to a very good optimum.

The outrageous success of both life and neural networks appears to argue for this hypothesis. 

# Generative AI
It has been fascinating to see the discussion around what ChatGPT and similar systems do. Are they intelligent? What does that question even mean? ChatGPT sounds unreasonably sure of itself at times, even when it is generating text that is dead wrong. To the people that use this to disparage AI, I ask, have you ever met any people?

