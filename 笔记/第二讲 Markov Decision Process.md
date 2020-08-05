# 第二讲 Markov Decision Process

## Markov Process

-  Markov Property：某一状态信息包含了所有相关历史，只要当前状态已知，所有的历史不再需要。
-  **Markov Process**：AKA Markov Chain. 是一个无记忆的随机过程，可有用元组<S,P>表示，其中S为有限数量的状态集，P为状态转移概率矩阵

## Markov Reward Process

-  **Markov Reward Process**：继承Markov Process的定义，再加上奖励R和衰减系数γ：<S，P，R，γ>。R是奖励函数，S状态下的奖励是t时刻处在状态S下，在t+1时刻能获得的奖励期望$R_s=E[R_{t+1}|S_t=S]$

-  Discount Factor：$ \gamma\in[0,1]$，用于表达对模型预测未来可信度的参数，γ越大说明目光越长远，γ越小说明越看重眼前的reward

-  return：收获$G_t$为一个MRP从t时刻开始往后所有的奖励的有衰减的总和，即$G_t=\sum_{k=0}^\inf \gamma^kR_{t+k+1}$ 。衰减系数体现了未来的奖励再当前时刻的价值比例

-  Value Function：价值函数给出了某个state或者action的长期价值，即$v(s)=E[G_t|S_t=s]$

-  Bellman Equation：For MRP, Bellman Equation is: $v(s)=E[R_{t+1}+\gamma v(S_{t+1})|S_t=s]$，即可以将v(s)

   分为即时奖励以及下一步的价值期望。矩阵形式的Bellman Equation为：$v=R+\gamma Pv$，因此可以直接解线性方程求出v，但是由于计算复杂度为$O(n^3)$，所以一般采用迭代的方法

## Markov Decision Process

-  **Markov Decision Process**：继承MRP的定义，并且增加了一个行为集合A，是这样的元组：<S,A,P,R,γ>。此时的R和P都必须与对应的action对应。定义为：$P_{ss'}^a=P[S_{t+1}=s'|S_t=s,A_t=a]$以及$R_s^a=E[R_{t+1}|S_t=s,A_t=a]$

-  policy：策略$\pi$是概率的集合或分布，其元素$\pi(a|s)=P[A_t=a|S_t=s]$为对过程中的某一状态s采取行为a的概率。policy至于当前的state有关，与历史无关。某一个确定的policy是静态的，与时间无关。奖励函数表示如下：$R_s^\pi=\sum_{a\in A}\pi(a|s)R_s^a$ 

-  Value Function Based on policy $\pi $：定义状态价值函数$v_\pi(s)=E_\pi[G_t|S_t=s]$，表示从状态S开始，遵循策略$\pi$所获得的reward的期望，也可理解为再策略$\pi$下状态S的价值。定义行为价值函数$q_\pi(s,a)=E_\pi[G_t|S_t=s,A_t=a]$，表示再状态S根据策略$\pi$执行行为a所获得的reward的期望

-  Bellman Expectation Equation：

   状态价值函数：$v_\pi(s)=E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]$

   行为价值函数：$q_\pi(s,a)=E_\pi[R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})|S_t=s,A_t=a]$

   $v_\pi(s)$和$$q_\pi(s,a)$$的关系:

   $v_\pi(s)=\sum_{a\in A}\pi(a|s)q_\pi(s,a)$

   $q_\pi(s,a)=R_s^a+\gamma \sum_{s'\in S}P_{ss'}^av_\pi(s')$

   最终可以得到：

   $v_\pi(s)=\sum_{a\in A}\pi(a|s)(R_s^a+\gamma \sum_{s'\in S}P_{ss'}^av_\pi (s'))$

   $q_\pi(s,a)=R_s^a+\gamma \sum_{s'\in S}P_{ss'}^a\sum_{a'\in A}\pi(a'|s')q_\pi(s',a')$

   ==核心思想是，当前的状态（行为）的价值可以用下一步的行为（状态）的价值的根据概率加权求平均再折现得到==

-  Optimal value function：最优价值函数$q_*(s,a)$指从所有策略产生的行为价值函数中，选取使得状态行为<s,a>价值最大的函数：$q_*(s,a)=max_\pi q_\pi(s,a)$。最优价值函数明确了MDP的最优可能表现，当给出了最优价值函数，就能知道每个状态的最优价值，从而解决该MDP

-  最优策略：

   定理：对于任何MDP，下面几点成立

   1. 存在最优策略，比任何其他策略更好或至少相等
   2. 所有的最优策略有相同的最优价值函数
   3. 所有的最优策略具有相同的行为价值函数
   4. （补充）对于同一个MDP，最优策略不一定唯一

   寻找最优策略：

   可以通过最大化最优行为价值函数来找到最优策略 $\pi_*(a|s)=1$ if $a=argmax_{a\in A}q_*(s,a)$ else 0

-  Bellman Optimality Equation：

   针对$v_*$，一个状态的最优价值等于从该状态出发采取的所有行为产生的行为价值中最大的那个行为价值：$v_*(s)=max_a q_*(s,a)$。

   针对$q_*$，在某个状态s下，采取某个行为的最优价值由离开状态s的即刻奖励与下一步的状态的最优价值根据概率加权折现求得：$q_*(s,a)=R_s^a+\gamma \sum_{s'\in S}P_{ss'}^aV_*(s')$。

   将两个式子组合起来，可以得到：

   $v_*(s)=max_aR_s^a+\gamma \sum _{s'\in S}P_{ss'}^av_*(s')$

   $q_*(s,a)=R_s^a+\gamma\sum_{s'\in S}P_{ss'}^amax_{a'}q_*(s',a')$

## 小结

​	这一讲主要介绍了MRP，MDP，以及value function的定义。在求解MDP的过程中，最重要的目标就是求出最优价值函数。针对状态价值函数的推导，则考虑下一步发生的行为的最优价值的平均。针对行为价值函数的推导，则考虑离开该状态的即时奖励以及下一步到达的状态的最优价值的平均（这里要考虑到由环境导致的不同行为到达不同状态的概率），而Bellman Equation则将这些推导用公式表达出来。求解Bellman Equation，最直观的就是求解线性方程，但是复杂度到了n3的级别，所以只能求解小型的MDP问题。一般，在求解MDP问题的时候会使用迭代的方法。

​	