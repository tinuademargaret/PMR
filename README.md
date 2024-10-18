# Generative Verifiers for Implicit Bias Detection

This project replicates and extends the work presented in the DeepMind paper "Generative Verifiers: Reward Modeling as Next-Token Prediction" [1], applying it to improve the judgment of LLaMA 3.1 model on implicit bias.

## Project Overview

The main goal of this project is to implement a Generative Reward Modeling (GenRM) approach, specifically GenRM-CoT (Chain-of-Thought), to enhance the ability of language models to detect implicit biases. We use a dataset adapted from the Implicit Association Test, as described in "Measuring Implicit Bias in Explicitly Unbiased Large Language Models" [2].

## Methodology

Our approach follows the GenRM-CoT methodology outlined in the following steps:

1. **Model Fine-tuning:** We fine-tune a LLaMA 3 model to produce verification chain-of-thought (CoT) rationales before yielding a final Yes/No token.

2. **Dataset Preparation:** We use a dataset based on the Implicit Association Test, which measures automatic associations between concepts. For each example in the dataset:
   - A biased response is provided.
   - Claude (another AI model) is prompted to generate a preferred, unbiased response.

3. **Training:** The model is trained on both the biased and preferred responses, learning to distinguish between them and generate appropriate CoT rationales.

4. **Inference:** At test time, we sample multiple CoT rationales and use majority voting to compute the average probability of 'Yes', allowing the model to leverage additional inference-compute for better verification.

## Key Components

- **Unified Training:** The project integrates reward modeling (distinguishing correct and incorrect solutions) with supervised fine-tuning for generating correct solutions.

- **Chain-of-Thought Reasoning:** The model generates intermediate reasoning steps before making a decision about solution correctness, potentially identifying subtle reasoning errors.

- **Majority Voting:** During inference, multiple verification CoT rationales are generated and their scores are averaged to produce a final prediction.

## Implementation Details

- **Model Architecture:** The project uses the LLaMA 3 model as the base architecture.
- **Training:** Training involves minimizing the supervised fine-tuning loss on a dataset containing problem-solution pairs, verification rationales, and correctness labels.
- **Inference:** Inference uses majority voting across multiple generated rationales to improve accuracy.

<!-- ## Results

(Include a brief summary of your results here, comparing the performance of your GenRM-CoT approach to baseline models in detecting and mitigating implicit bias.)

## Future Work

- Explore the use of synthetic verification rationales to further improve model performance.
- Investigate the model's performance on other types of bias beyond those captured in the current dataset.
- Analyze the generated CoT rationales to gain insights into the model's reasoning process about bias. -->

## References

1. Generative Verifiers: Reward Modeling as Next-Token Prediction. (2023). arXiv:2408.15240
2. Measuring Implicit Bias in Explicitly Unbiased Large Language Models. (2024). arXiv:2402.04105

## Acknowledgements

This project builds upon the work of researchers at DeepMind and the authors of the implicit bias measurement paper.