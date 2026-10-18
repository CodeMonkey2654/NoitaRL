# Paper Index

## MaestroMotif: Skill Design from Artificial Intelligence Feedback

Klissarov, Henaff, Raileanu, Sodhani, Vincent, Zhang, Bacon, Precup, Machado, D'Oro (2024). Added 2026-10-18.
Link: [link to page](https://arxiv.org/abs/2412.08542d). PDF: [2026-10-18_MaestroMotif.pdf](2026-10-18_MaestroMotif.pdf)

**Summary:** Skills are described in plain English. An LLM compares pairs of gameplay moments to build a reward for each skill, and an LLM-written controller sequences the skills. All skills are trained with PPO inside one network. On NetHack it reaches the Gnomish Mines 46% of the time vs 3% for PPO, and Delphi 29% vs 0%.

**Why it matters for Noita:** It follows the same path as the plan: a PPO baseline, then hierarchical skills such as descending, fighting, digging, and editing wands. LLM-feedback rewards give dense signal where the game's own rewards are sparse. One shared network beat separate heads per skill.

## Proximal Policy Optimization Algorithms
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford et al. (2017).
Link: https://arxiv.org/abs/1707.06347. PDF: [2017_PPO.pdf](2017_PPO.pdf)

**Summary:** Introduces PPO, a policy-gradient method that alternates between collecting experience and optimizing a clipped surrogate objective over several minibatch epochs. The clipping keeps each update close to the previous policy, giving much of the stability of trust-region methods with far simpler code.

**Intuition for Noita:** PPO is the workhorse optimizer for our skill-conditioned policy: cheap, stable, and easy to scale across many Noita instances. Its clipped updates help keep learning steady when rewards come from noisy milestone and curiosity signals.

## IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan et al. (2018).
Link: https://arxiv.org/abs/1802.01561. PDF: [2018_IMPALA.pdf](2018_IMPALA.pdf)

**Summary:** IMPALA decouples many actors that generate experience from a central learner that trains on it. Because actors act with slightly stale policies, it uses an off-policy correction called V-trace to keep learning accurate at high throughput.

**Intuition for Noita:** This is the blueprint for our distributed actor-learner on HPC: many Noita game processes feeding GPU learners. V-trace-style corrections matter when actor policies lag behind the learner across the cluster.

## Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning
Aleksei Petrenko, Zhehui Huang, Tushar Kumar, Gaurav Sukhatme et al. (2020).
Link: https://arxiv.org/abs/2006.11751. PDF: [2020_SampleFactory.pdf](2020_SampleFactory.pdf)

**Summary:** Sample Factory is a highly optimized asynchronous RL system for a single machine, designed to maximize simulation and training throughput from pixels. It carefully pipelines environment stepping, inference, and learning to avoid idle hardware.

**Intuition for Noita:** Noita is slow to simulate, so throughput engineering decides how much experience we can collect per HPC node. Its pipelining and batching tricks are directly applicable to our actor-learner and PPO setup.

## Exploration by Random Network Distillation
Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov (2018).
Link: https://arxiv.org/abs/1810.12894. PDF: [2018_RND.pdf](2018_RND.pdf)

**Summary:** RND gives an exploration bonus equal to how badly a trained network predicts the output of a fixed, randomly initialized network on the current observation. Novel states are poorly predicted and thus rewarded, which helps on hard-exploration games like Montezuma's Revenge.

**Intuition for Noita:** RND is a simple, robust curiosity signal for pushing the agent into new biomes and interactions in Noita. Its novelty scores can also feed curiosity-prioritized replay alongside world-model error.

## Benchmarking the Spectrum of Agent Capabilities
Danijar Hafner (2021).
Link: https://arxiv.org/abs/2109.06780. PDF: [2021_Crafter.pdf](2021_Crafter.pdf)

**Summary:** Introduces Crafter, a 2D open-world survival game where agents are scored on unlocking a set of achievements that require exploration, generalization, and long-horizon reasoning. It is designed to be fast to run and to evaluate many capabilities from a single environment.

**Intuition for Noita:** Crafter is the closest cheap proxy for Noita: procedural worlds, survival, and a tree of milestones. We can prototype our milestone auxiliary losses and evaluation methodology here before paying Noita's simulation cost.

## Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos
Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga et al. (2022).
Link: https://arxiv.org/abs/2206.11795. PDF: [2022_VPT.pdf](2022_VPT.pdf)

**Summary:** VPT trains an inverse dynamics model on a small labeled dataset to infer the actions in unlabeled Minecraft videos, then uses those pseudo-labels to pretrain a policy by behavioral cloning. The pretrained policy is then fine-tuned with imitation and reinforcement learning to reach hard tasks.

**Intuition for Noita:** There is lots of Noita gameplay video online; VPT shows how to turn it into a warm-start policy. Starting PPO from such a prior should drastically cut the exploration needed for basic movement, wand use, and descent.

## Deep Hierarchical Planning from Pixels

Danijar Hafner, Kuang-Huei Lee, Ian Fischer, Pieter Abbeel (2022).
Link: https://arxiv.org/abs/2206.04114. PDF: [2022_Director.pdf](2022_Director.pdf)

**Summary:** Director learns a manager that picks goals in the latent space of a learned world model and a worker that learns to reach them, all trained from pixels inside imagination. The goals can be decoded into images, making the agent's high-level decisions interpretable.

**Intuition for Noita:** This is the template for our manager over skills: high-level choices every few steps, low-level control by the worker. Training both inside the Dreamer-style world model keeps hierarchy learning cheap relative to real Noita steps.

## Mastering Diverse Domains through World Models
Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap (2023).
Link: https://arxiv.org/abs/2301.04104. PDF: [2023_DreamerV3.pdf](2023_DreamerV3.pdf)

**Summary:** DreamerV3 learns a world model of the environment and trains an actor-critic purely on imagined rollouts, using normalization and robustness tricks so one set of hyperparameters works across many domains. It was reported as the first algorithm to collect diamonds in Minecraft from scratch without human data.

**Intuition for Noita:** DreamerV3 is the core of our agent's world model, on top of the ResNet encoder, with milestone prediction added as auxiliary losses. Its fixed-hyperparameter robustness matters because tuning per experiment on Noita is expensive.

## SPRING: Studying the Paper and Reasoning to Play Games

Yue Wu, Shrimai Prabhumoye, So Yeon Min, Yonatan Bisk et al. (2023).
Link: https://arxiv.org/abs/2305.15486. PDF: [2023_SPRING.pdf](2023_SPRING.pdf)

**Summary:** SPRING has an LLM read the Crafter paper and then reason over a question graph about the current game state to choose actions, with no training. It shows that game knowledge from text can drive competent play in an open-world game.

**Intuition for Noita:** This motivates our optional LLM planner: use Noita wiki knowledge to suggest which skill or milestone to pursue next. Used sparingly at the manager level, it can inject domain knowledge that RL alone would take ages to discover.

## Curious Replay for Model-based Adaptation
Isaac Kauvar, Chris Doyle, Linqi Zhou, Nick Haber (2023).
Link: https://arxiv.org/abs/2306.15934. PDF: [2023_CuriousReplay.pdf](2023_CuriousReplay.pdf)

**Summary:** Curious Replay prioritizes replay of experiences the world model finds surprising or has rarely trained on, instead of sampling uniformly. This helps model-based agents like Dreamer adapt faster to changes and improves performance on Crafter.

**Intuition for Noita:** This is exactly our curiosity-prioritized replay: train the world model more on rare Noita events (new biomes, explosions, deaths). It keeps rare but important dynamics from being drowned out by common ones.

## Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning
Seungyong Moon, Junyoung Yeom, Bumsoo Park, Hyun Oh Song (2023).
Link: https://arxiv.org/abs/2307.03486. PDF: [2023_AchievementDistillation.pdf](2023_AchievementDistillation.pdf)

**Summary:** The paper shows a well-tuned PPO agent with a ResNet encoder is strong on Crafter, then adds achievement distillation, a contrastive objective that makes representations predict the next achievement. This improves achievement unlocking without an explicit world model or hierarchy.

**Intuition for Noita:** This justifies our ResNet encoder and the milestone auxiliary losses: teaching representations to anticipate the next Noita milestone. It is also a strong model-free baseline to beat before adding complexity.

## Motif: Intrinsic Motivation from Artificial Intelligence Feedback
Martin Klissarov, Pierluca D'Oro, Shagun Sodhani, Roberta Raileanu et al. (2023).
Link: https://arxiv.org/abs/2310.00166. PDF: [2023_Motif.pdf](2023_Motif.pdf)

**Summary:** Motif asks an LLM to compare pairs of game event captions and distills its preferences into an intrinsic reward model, which then trains an RL agent in NetHack. The resulting agent makes progress even without game-score rewards and behaves in more human-aligned ways.

**Intuition for Noita:** Noita has rich event/message text; Motif shows how an LLM can turn it into dense reward shaping offline. This complements RND curiosity and is a cheaper use of an LLM than calling a planner every step.

