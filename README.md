# gym-blackjack-sample

### INSTALLATION
1. Review [install guide](http://docs.bons.ai/guides/getting-started.html#install-prerequisites) for installing the Bonsai CLI.
2. Install the simulator's requirements:
       `pip install -r requirements.txt`

### HOW TO TRAIN YOUR BRAIN
1. If you haven't already created a BRAIN at the website, create one now:
       `bonsai create <your_brain>`
2. Load your Inkling file into your brain. Review our [Inkling Guide](http://docs.bons.ai/guides/inkling-guide.html) for help with Inkling.
       `bonsai push`
3. Enable training mode for your brain. Please note that training may take many hours.
       `bonsai train start`
4. Connect a simulator for training. For inspiration, check out our [Mountain Car demo](https://github.com/BonsaiAI/gym-mountaincar-sample).
       `python blackjack_simulator.py --train-brain=<your_brain> --headless`
5. When training has hit a sufficient accuracy, disable training mode.
       `bonsai train stop`

### USE YOUR BRAIN

1. Run the simulator using predictions from your brain.
       `python blackjack_simulator.py --predict-brain=<your_brain> --predict-version=latest`
