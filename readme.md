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
For personal reasons, I won't be including the training set I used.
(Would you like a bunch of strangers to have access to your pictures?
...Oh wait, that sounds like what everyone does on Facebook)

Put all your training images in the images/ directory.
Then open 'run.py' and:
1) Edit the `train()` function to correctly build training data from your images
(all you have to do is make sure that `get_training_data` finds your images
with the correct image type)
2) Edit the `main()` function to set how many training epochs you want and how often
you want a generated image to be saved. (The most number of epochs I've run it
for is 2000, and there generally was no improvement after about 500 epochs or so.
Yeah the current GAN that I've got here is sh*t...Or to be diplomatic, in the
 style of what you say during an interview, "Training a GAN
is a very challenging and rewarding task")

Then open `models/parameters.py` and set the parameters accordingly.
