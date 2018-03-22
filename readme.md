# This project has been closed
After struggling for a while to implement a Boundary Equilibrium Generative Adversarial Network
using keras on tensorflow, I have decided to reimplement everything using pure tensorflow. The old code
is in the 'build' branch, which will not be merged with master.

I have realised that keras is not the best for implementing slightly more advanced models,
such as those with custom loss functions with external variables. To accomplish this using
keras, I would have to use an ugly and hacky approach, such as passing the external variables through
the model.fit() function together with the input data. It is also troublesome to display the
values of tensors while training. I had to define a custom Callback object which would receive
the values of the tensors while training. The data could then be printed from this object.

##### Some random things that I learnt
1) It is really important to understand how the tensorflow graph works, especially if I want to
do slightly more advanced things like custom loss functions. The old 'slap some layers together
and hope it works' strategy cannot be used anymore.
2) This is not a good idea:
`model.add(layers.Dense(np.prod(params.IMG_SIZE)))`.
If `params.IMG_SIZE == (200,200,3)`, then I will be adding 200x200x3=**120,000 nodes**
3) How to estimate memory requirements for a neural net. See https://datascience.stackexchange.com/questions/17286/cnn-memory-consumption
4) How to use AWS for machine learning.
5) It is really important for debugging to be able to see the values of all parameters as training
progresses.

I will keep this repository to remind me of mistakes I made and the things that I have learnt
through this project.

## generate-my-face
This is an attempt to build a Generative Adversarial Network to generate my own face.
Here are some images that I managed to generate so far. They aren't very good.

Overfit

![Alt text](cool/overfit.png?raw=true "Overfit")

---

Blurry

![Alt text](cool/blurry.png?raw=true "Overfit")

---

Abstract Art

![Alt text](cool/abstract_art.png?raw=true "Overfit")


## Next steps
This looks promising.
https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/

## How to use
I won't be including the training set I used.

Put all your training images in the images/ directory.
Then open 'run.py' and:
1) Edit the `train()` function to correctly build training data from your images
(all you have to do is make sure that `get_training_data` finds your images
with the correct image type)
2) Edit the `main()` function to set how many training epochs you want and how often
you want a generated image to be saved. (The most number of epochs I've run it
for is 2000, and there generally was no improvement after about 500 epochs or so.)

Then open `models/parameters.py` and set the parameters accordingly.
