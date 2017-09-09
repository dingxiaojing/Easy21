# Easy21
Reinforcement Learning algorithms for a card game Easy21
A solution of the assignment in [David Silver's course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

- `Easy21.py` A enviroment implemented the card game [Easy21](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf), a modified *blackjack* based on [OpenAI Gym](https://gym.openai.com/docs)
- `MonteCarloControl.py` Implementation of Monte-Carlo Control to Easy21 and plot the optimal state value function
- `Sarsa.py` Implementation of Sarsa(lambda) control to Easy21 and plot the MSE against lambda and episode number
- `LinearFA.py` Apply a linear function approximator to value function and repeat Sarsa(lambda)
