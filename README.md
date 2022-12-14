# GLM_Network
## The GLM_Network is a combination of the Guassian linear model and a neural network in order to reconstruct an unknown function.
### The GLM try to model data with 
$$f(x)=\sum\theta_iX^i(x)$$
where $X^i(x)$ are arbitrary base functions and $\theta_i$ are free parameters

### Using a simple neural network, the base functions are made and then feed to the GLM.

### For a simple toy data see example.ipynb

### For applying the method on the recent Hubble data, see hub.ipynb
