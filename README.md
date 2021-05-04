# OpenAI_atari

## Summary
My attempt to use Tensorflow and Keras to train a Q-learning agent to play Pitfall on OpenAI gym. The wrapper was developed to facilitate an interaction with OpenAI gym and allow me to easily change the neural network parameters and log how the changes effected the models ability to progress in the game. The rewards in Pitfall are a long ways off so I had to use a large replay buffer. Which meant my current hardware was not powerful enough to train multiple models in the time I had availiable. Fortunatly I was able to use my 300 Google Cloud credit and spin up a VM with a powerfull GPU. 

Sample runs using a larger and a smaller CNN, each with a 1 million move replay buffer. Subjectively the smaller CNN was more careful but the larger CNN made it further into the game.

![largeNN1mil](https://github.com/aurvandel/OpenAI_atari/blob/30137c48fa046a5bb2f47b46c9beb0a9455e0d23/atari/largeNN1mil.gif) 

![smallNN1mil](https://github.com/aurvandel/OpenAI_atari/blob/30137c48fa046a5bb2f47b46c9beb0a9455e0d23/atari/smallNN1mil.gif)

## Sample plots comparing the results of runs with different settings

### Large CNN with 2k move replay buffer

#### Training

![Large CNN 2k buffer train](https://github.com/aurvandel/OpenAI_atari/blob/30137c48fa046a5bb2f47b46c9beb0a9455e0d23/atari/output/largerNN_2k_buffer/run20200416-071839_Pitfall-v4/training_info.png)

#### Evaluation

![Large CNN 2k buffer eval](https://github.com/aurvandel/OpenAI_atari/blob/30137c48fa046a5bb2f47b46c9beb0a9455e0d23/atari/output/largerNN_2k_buffer/run20200416-071839_Pitfall-v4/evaluation.png)

| avg_score           | avg_Q              | 
|---------------------|--------------------| 
| -317.63224937571897 | 4.001707762392947  | 
| -46.76040052748487  | 3.5257336499018925 | 
| -38.541254108850026 | 5.154112006786231  | 
| -125.14807483173884 | 5.221036148929534  | 

### Small CNN with 1 million move replay buffer

#### Training

![Small CNN 1 mil buffer train](https://github.com/aurvandel/OpenAI_atari/blob/30137c48fa046a5bb2f47b46c9beb0a9455e0d23/atari/output/smallerNN/run20200411-053816_Pitfall-v4/training_info.png)

#### Evaluation

![Small CNN 1 mil buffer eval](https://github.com/aurvandel/OpenAI_atari/blob/30137c48fa046a5bb2f47b46c9beb0a9455e0d23/atari/output/smallerNN/run20200411-053816_Pitfall-v4/evaluation.png)

| avg_score          | avg_Q              | 
|--------------------|--------------------| 
| -390.7353559290824 | 3.7946990955465365 | 
| -328.9562328111802 | 4.253121242812553  | 
| -412.5182987616657 | 5.083157772519628  | 