# OpenAI_atari

## Summary
My attempt to use Tensorflow and Keras to train a Q-learning agent to play Pitfall on OpenAI gym. The wrapper was developed to facilitate an interaction with OpenAI gym and allow me to easily change the neural network parameters and log how the changes effected the models ability to progress in the game. The rewards in Pitfall are a long ways off so I had to use a large replay buffer. Which meant my current hardware was not powerful enough to train multiple models in the time I had availiable. Fortunatly I was able to use my 300 Google Cloud credit and spin up a VM with a powerfull GPU. 
