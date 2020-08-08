# 第五讲 Model Free Control

## Introduction

-  On policy learning ：agent已有一个策略，并且**遵循这个策略采样**（采取一系列该策略下的行为），根据这一系列行为得到的奖励，更新状态函数，最后根据该更新的价值函数来优化策略得到较优的策略。由于要优化的策略是当前遵循的策略，所以是on policy
-  off policy learning ：agent虽有一个策略，但是不针对这个策略进行采样，而是**基于另一个策略进行采样**。这一策略可以是先前学习的一个策略，也可以是人类的策略等一些较为成熟的策略，通过对这类策略进行采样，得到这类策略下的行为，继而得到一系列奖励，然后更新价值函数。**即在自己的策略形成的价值函数的基础上观察别的策略产生的行为**，以此达到学习的目的。这种学习方式类似于“站在巨人的肩膀上可以看的更远”。由于这些策略是已有的策略，所以成为off policy learning

## On Policy Learning

### On-Policy Monte-Carlo Control

-  Model Free的两个条件：
   -  **用状态行为的价值Q来代替状态价值**：在模型位置的条件下无法知道当前状态的所有后续状态，进而无法确定当前状态下采取怎样的行为更合适。因此，使用状态行为的价值Q（s，a）来代替状态价值$\pi'(s)=argmax_{a\in A}Q(s,a)$。具体做法：从初始的Q和策略$\pi$开始，现根据这个策略对Q进行更新，然后基于更新的Q确定改善的贪婪算法。
   -  **ε-贪婪探索**：目标是使得某一状态下所有可能的行为都有一定几率被选中，保证了持续的探索。其中，1-ε的概率下选中最好的行为，ε概率下在所有可能的行为中选择（包括最好的行为）。

-  蒙特卡洛控制：使用Q函数进行策略评估，使用ε-贪婪探索来改善策略，最终可以收敛至最优策略。
-  GLIE（Greedy in the Limit with Infinite Exploration）
   -  在有限的时间内进行无限可能的探索。具体为：**所有已经经历的状态行为对会被无限次探索；另外随着探索的无限延申，贪婪算法中的ε趋向0**。例如取ε=1/k，则该ε-贪婪算法就有GLIE特性。
   -  控制流程
      -  对于给定策略，采样第k个episode：$\{S_1,A_1,R_2,...,S_T\}$~ $\pi$
      -  对于该Episode中出现的**每一个状态行为对**，更新计数和Q函数：$N(S_t,A_t)\leftarrow N(S_t,A_t)+1,Q(S_t,A_t)\leftarrow Q(S_t,A_t)+1/N(S_t,A_t)*(G_t-Q(S_t,A_t))$
      -  基于更新的Q函数按照如下方式改善策略：$\epsilon\leftarrow 1/k,\pi\leftarrow greedy(Q)$。一般来说蒙特卡洛要求完整的episode，所以改善策略是在整个episode的状态行为对都更新完之后再改善的。
   -  定理：GLIE蒙特卡洛控制能收敛至最优状态行为价值函数。

### On-Policy Temporal-Difference Control

-  **SARSA**：**针对一个状态S，以及一个特定的行为A，进而产生一个状态行为对（SA），与环境交互，环境收到个体的行为后会反馈R以及后续进入的状态S’。接下来个体遵循现有策略产生一个行为A‘，根据当前的状态行为价值函数得到后一个状态行为对（S’A‘）的价值Q，利用这个Q来更新前一个状态行为对（SA）的价值**
   -  与蒙特卡洛不同的是：每一个时间步，也就是再单个episode内每一次个体在状态St采取一个行为后都要更新Q值，同样使用ε-贪婪探索的形式来改善策略$Q(S,A)\leftarrow Q(S,A)+\alpha(R+\gamma Q(S',A')-Q(S,A))$
   -  n步SARSA：定义n步Q：$q_t^{(n)}=R_{t+1}+\gamma R_{t+2}+...+\gamma ^{n-1}R_{t+n}+\gamma^nQ(S_{t+n})$这里的n步Q中的$Q(S_{t+n})$表示的是在$S_{t+n}$状态下，根据策略选择$A_{t+n}$后得到的reward。则n步SARSA可以表示为：$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha(q_t^{(n)}-Q(S_t,A_t))$。如果给n步收获一个权重，能够利用到所有步的收获，则可以类似上一讲一样定义：$q_t^\lambda=(1-\lambda)\sum_{n=1}^\infin\lambda^{n-1}q_t^{(n)}$
   -  引入效用追踪（Eligibility Trace）：针对每一个状态行为对，有一个效用E0，体现的是一个结果与某一个状态行为对的因果关系（类似上一讲），发生次数越多或者发生时间越近效用越大。引入该概念后的SARSA（λ）的Q值更新为：$\delta_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t),Q(s,a)\leftarrow Q(S,a)+\alpha\delta_tE_t(s,a)$

## Off Policy Learning

### 离线TD学习

-  使用TD算法在遵循一个策略$\mu(a|s)$的同时评估另一个策略$\pi(a|s)$，具体数学表示为：$V(S_t)\leftarrow V(S_t)+\alpha(\frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}(R_{t+1}+\gamma V(S_{t+1}))-V(S_t))$。思想是：**通过比较$\pi,\mu$两种策略在St中执行行为At的概率大小，来判断两种策略对于该状态行为的评估是否相近**。若该比值小，则表明如果按照被评估的策略，选择At的机会很小，这时候在更新St价值的时候就不能过多的考虑基于当前策略得到的状态St+1的价值。比值大于1的时候道理类似。
-  基于TD（0）的Q-learning：$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha(R_{t+1}+\gamma Q(S_{t+1},A')-Q(S_t,A_t))$。其中$R_{t+1}+\gamma Q(S_{t+1},A')$为基于另一个策略产生的行为A’得到的Q价值。Q学习的最主要的表现形式是：个体遵循的策略是基于当前状态行为价值函数Q（s，a）的一个ε-贪婪策略，而目标策略是是基于当前状态行为价值函数Q（s，a）不包含ε的单纯greedy策略。

![image-20200808170114720](C:\Users\martin gzz17\AppData\Roaming\Typora\typora-user-images\image-20200808170114720.png)

![image-20200808170151934](C:\Users\martin gzz17\AppData\Roaming\Typora\typora-user-images\image-20200808170151934.png)