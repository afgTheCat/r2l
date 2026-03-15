# Sampler


Couple of caveats:

  - A sampler is concerned with running rollouts
  - The sampler does not reset the environment between two rollouts
  - The action to be taken is decided based on what the last state is. That is:
    -- If the buffer already has a last state, we can use that (back on the buffer)
    -- If we just reset without resetting, we should have this backed up somewhere
    -- If there is no last state in the buffer + no backed up thing, that means we have 0 rollouts collected -> reset the state
  - Then we might take a step using the action. If we reach an ending state, the next state will be a reset state.
  - We push stuff later!
  - What is notable here, is that the rollout buffer we used were one longer than the actions, meaning that we save the last state which is reached

So what we can do here are two things:

  - we can just create a RingBuffer regardless? Probably no, but it does not matter.
  - what is bad here, is that with a RingBuffer, there are no guarantees that the data will be in order
  - AsRef<trait Buffer>>? where the Buffer is an iterator thingy. What we can do is the following:

  - The problem is the following:
    - if U and T are the same -> we can just return the Buffer
    - if U and T are not the same -> we should just create a new Buffer

    - the only problem this introduces is that 
