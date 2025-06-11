# How verification works

Vericifation happens in 5 steps:

- Step 1: Parse the configuration files for the pre trained model
- Step 2: Try constructing each model according to the configuration
- Step 3: Run the sb3 model => collect the rewards received per episode
- Step 4: Run the r2l model => collect the rewards received per episode
- Step 5: Compare the rewards according to some metric

## Step 1

TODO: writeup

## Step 2

