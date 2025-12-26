# Self‑Driving Lane Car — Reinforcement Learning Simulation

**Author:** Pedro Yanez Melendez

## Purpose
- Design and implement a self‑driving lane car simulation
- Apply reinforcement learning to learn driving behavior
- Combine AI training with an interactive web experience
- Deploy the system as a mobile‑responsive web page

## What this project does
- Simulate a lane‑based driving environment
- Allow manual driving and automatic (AI) driving modes
- Train an AI agent using reinforcement learning in Google Colab
- Export a trained policy and run it directly in the browser
- Work on desktop and mobile devices without backend services

## Technologies
- Python for reinforcement learning training
- Deep Q‑Learning (DQN) for policy optimization
- Google Colab for experimentation and training
- HTML, CSS, and JavaScript for the web interface
- GitHub Pages for deployment

## Reinforcement learning approach
- Define environment state from lanes, obstacles, and speed
- Define discrete actions: move left, stay, move right
- Define rewards for survival and progress
- Penalize collisions and unnecessary lane changes
- Train progressively with increasing difficulty
- Export learned weights as a reusable policy

## Web integration
- Load trained policy directly in the browser
- Execute inference fully client‑side
- Switch between manual and AI control at runtime
- Render using HTML canvas
- Support keyboard and touch input

## Mobile support
- Adapt layout to screen size automatically
- Support touch controls for lane changes
- Maintain identical AI behavior across devices

## Key results
- Build an end‑to‑end AI system from training to deployment
- Demonstrate real reinforcement learning, not scripted behavior
- Achieve fast iteration and experimentation speed
- Create a reusable pattern for AI‑powered web simulations

## How to use
- Open the web page
- Press M to drive manually
- Press T to activate AI driving
- Refresh the page to reload the policy

## Notes
- Retrain the AI by running the Colab notebook again
- Replace the policy file to update behavior
- No server or cloud runtime required after deployment
