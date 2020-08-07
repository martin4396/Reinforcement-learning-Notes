# 第四讲 Model Free Prediction

## Monte Carlo Reinforcement Learning

-  定义：再不清楚MDP的dynamic和reward的情况下，直接从**经历完整的episode**来学习状态的价值，通常情况下某状态的价值等于在多个episode中以该状态算得到的所有收获的平均
-  episode：agent进入某个状态之后，一直和环境交互直到**到达终止状态**。
-  Monte Carlo Policy Evaluation：再给定策略下，从一系列的完整episode经历中学习得到该策略下的状态价值函数。再解决问题的过程种主要使用的信息是**一系列完整episode**：包含状态的转移、行为的序列、中间状态的即时奖励以及最终状态的即时奖励。其特点是使用**有限的、完整的**episode产生的信息经验性地推导出每个状态的平均收获，以此来替代收获的期望，即状态价值。
   -  first visit Monte Carlo：再给定一个策略，使用一系列完整episode评估某一个状态s时，对于每一个episode，仅当该状态**第一次**出现时列入计算
   -  every visit Monte Carlo：再给定一个策略，使用一系列完整episode评估某一个状态s时，对于每一个episode，状态s**每次**出现在状态链中时，都要列入计算

-  Incremental Mean累进更新平均值：$\mu_k=\mu_{k-1}+1/k*(x_k-\mu_{k-1})$

-  蒙特卡洛累进更新：对episode的每一个状态S，有一个收获G，每碰到一次S，使用下式计算状态的平均价值：$V(S_t)\leftarrow V(S_t)+\alpha(G_t-V(S_t))$。这里的α参数是用来更新状态价值的。使用该方法可以扔掉已经算过的episode的信息。

   

## Temporal Difference Learning 时序差分学习 ($TD(0)$)

-  TD learning：通过学习不完整的episode，并且通过自身引导（bootstraping），猜测episode的结果，并且同时持续更新这个猜测。**在MC中，使用G来更新状态价值。而在TD中，使用离开该状态的即时奖励$R_{t+1}$与下一状态$S_{t+1}$的预估状态价值乘以γ组成，即Bellman方程的形式**：$V(S_t)\leftarrow V(S_t)+\alpha(R_{t+1}+\gamma V(S_{t+1})-V(S_t))$。其中$R_{t+1}+\gamma V(S_{t+1})$称为TD目标值，$\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$称为TD误差。Bootstrapping指的是用TD目标值代题收获G的过程。

   

## $TD(\lambda)$

-  n-step prediction:在当前状态**向前行动n步**，计算n步的return。同样TD target由两部分组成，已走的用确定的reward，剩下的用估计的状态价值代题。定义n步收获$G_t^{(n)}=R_{t+1}+\gamma R_{t+1}+...+\gamma^{n-1}R_{t+n}+\gamma^nV(S_{t+n})$，那么n步TD学习状态价值函数的更新公式为$V(S_t)\leftarrow V(S_t)+\alpha(G_t^{(n)}-V(S_t))$
-  如何确定n选多少：引入新参数$\lambda$，做到在不增加计算复杂度的情况下综合考虑所有部署的预测
   -  $\lambda$收获：定义$G_t^\lambda=(1-\lambda)\sum_{n=1}^\inf\lambda^{n-1}G_t^{(n)}$为λ收获。对应的λ预测携程TD（λ）：$V(S_t)\leftarrow V(S_t)+\alpha(G_T^\lambda-V(S_t))$。也就是说，给不同长度的预测分配了不同的权重，第一步为（1-λ），第二步为λ（1-λ），直至无穷。这样的好处是到无穷时权重为1。

-  效用追踪：结合频率启发和就近启发，给每一个状态引入一个数值：效用追踪，定义如下：$E_0(s)=0,E_t(s)=\gamma\lambda E_{t-1}(s)+1(S_t=s)$。这个E能够体现这个状态的效用，并且是随时间变化的。体现在公式里更新状态价值是这样的：$\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t),V(s)\leftarrow V(s)+\alpha\delta_tE_t(s)$。

   