# Hands-on Lab: Neural Network Training Experiments

## Context
This week’s lab is focused on understanding basic neural network training behavior, then validating how optimizer choice affects performance.

## Notebooks to run
1. **Baseline notebook (start here):**
   - https://github.com/samx18/nn-training-demo/blob/main/minimal_pytorch_nn_training.ipynb
2. **Teaching notebook with session experiments:**
   - https://github.com/samx18/nn-training-demo/blob/main/minimal_pytorch_nn_training_variations.ipynb
3. **Real-life-like dataset variant (house prices):**
   - https://github.com/samx18/nn-training-demo/blob/main/minimal_pytorch_nn_training_variations_house_prices.ipynb

## Lab objectives
- Run and understand the baseline notebook end-to-end.
- Reproduce the experiments from the two variation notebooks.
- Perform additional experiments of your own and document observations.
- We obtained strong results with **ReLU + Adam**; try to achieve comparable performance using **SGD**.

## Suggested workflow
1. Run baseline notebook exactly as-is.
2. Note baseline metrics (loss curves, train/validation behavior, final error).
3. Run notebook variations and compare outcomes.
4. Replace Adam with SGD in at least one setup and tune:
   - learning rate
   - momentum
   - batch size
   - number of epochs
5. Record what changed and why you think performance moved up/down.

## Recommended experiment matrix
| Experiment | Activation | Optimizer | LR | Momentum | Notes |
|---|---|---|---:|---:|---|
| E1 (baseline) | ReLU | Adam | 0.001 | N/A | Reference run |
| E2 | ReLU | SGD | 0.1 | 0.0 | High LR sanity check |
| E3 | ReLU | SGD | 0.01 | 0.9 | Common SGD setup |
| E4 | ReLU | SGD | 0.001 | 0.9 | Lower LR stability check |
| E5 | ReLU | SGD | tuned | tuned | Best SGD attempt |

## What to document
For each run, capture:
- Hyperparameters used.
- Final training and validation metrics.
- Whether training was stable/unstable.
- Any signs of overfitting or underfitting.
- 2–3 bullet conclusions from the run.

## Submission checklist
- [ ] Baseline notebook run successfully.
- [ ] Both variation notebooks run successfully.
- [ ] At least 3 SGD experiments performed.
- [ ] Best SGD configuration identified.
- [ ] Observations documented clearly.

## Support
If you get blocked at any point (environment issues, convergence issues, unclear output), post your error/plots and questions in the group so we can help quickly.
