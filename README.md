# What's the optimal Blackjack strategy?
I tried to apply a rudimentary reinforcement learning algorithm to figure it out.

The first step, of course, is to create an actual blackjack game. Then, I made a model play it a million times.

The result is a little odd. The model hits every time the dealer has an ace, even when the player has a hand of 21. I'll have to look into this more in the future, because that is definitely not a smart move.

## Resulting Decision Matrix
![Figure_1](https://github.com/user-attachments/assets/622e9d97-7f07-4271-b4bb-f781ba9bb8c6)

## Model Confidence Heatmap
![image](https://github.com/user-attachments/assets/99f93556-988c-49c3-89e2-65c2425f46eb)
