arXiv:2407.00681v1 [eess.SY] 30 Jun 2024

# Safe Reinforcement Learning for Power System Control: A Review

Peipei Yu, Student Member, IEEE, Zhenyi Wang, Student Member, IEEE, Hongcai Zhang, Senior Member, IEEE, and Yonghua Song, Fellow, IEEE

# Abstract

The large-scale integration of intermittent renewable energy resources introduces increased uncertainty and volatility to the supply side of power systems, thereby complicating system operation and control. Recently, data-driven approaches, particularly reinforcement learning (RL), have shown significant promise in addressing complex control challenges in power systems, because RL can learn from interactive feedback without needing prior knowledge of the system model. However, the training process of model-free RL methods relies heavily on random decisions for exploration, which may result in “bad” decisions that violate critical safety constraints and lead to catastrophic control outcomes. Due to the inability of RL methods to theoretically ensure decision safety in power systems, directly deploying traditional RL algorithms in the real world is deemed unacceptable. Consequently, the safety issue in RL applications, known as safe RL, has garnered considerable attention in recent years, leading to numerous important developments. This paper provides a comprehensive review of the state-of-the-art safe RL techniques and discusses how these techniques can be applied to power system control problems such as frequency regulation, voltage control, and energy management. We then present discussions on key challenges and future research directions, related to convergence and optimality, training efficiency, universality, and real-world deployment.

# Index Terms

Safe reinforcement learning, frequency regulation, voltage control, energy management, power systems.

to smart grids, becoming more flexible and intelligent. Specifically, unpredictable fluctuations and distributed model complexity brought by this transformation require real-time monitoring and strategic control for power systems, based on advanced information and communication technology and intelligent control methods.

In particular, reinforcement learning (RL) has been considered a prominent approach to overcome these challenges in smart grids. On the one hand, RL can learn from interaction feedback without prior knowledge of the system model. On the other hand, RL can utilize neural networks to establish data-driven models for uncertain environment descriptions, enabling well-trained agents to adapt to environment changes continually. Over the past decade, RL has achieved great success in complex control problems of power systems, e.g., frequency regulation and voltage control. Generally, before deploying RL controllers on real-world systems, existing RL-related studies in power systems can only train RL controllers on high-fidelity simulations while not directly on real-world systems. This is because traditional RL training cannot theoretically guarantee decision safety in a real-world system. However, considering the system gap between simulations and the real world, controllers that are trained completely based on simulations cannot ensure effectiveness in real-world systems. Hence, it is necessary to achieve safe training on real-world systems to bridge the above gap from theory to practice, which has triggered research on the safe RL.

# I. INTRODUCTION

With the intensifying change in global climate, promoting carbon neutrality has become a consensus worldwide. As a critical energy infrastructure, it is necessary to drive energy transition in power systems to reduce carbon emissions. Energy transition mainly facilitates two changes to traditional power systems: 1) The system energy structure changes, where renewables (e.g., solar photovoltaic and wind power) will probably become the main power source with its increasing penetration. This change leads to greater uncertainty and volatility on the power supply side, further complicating the real-time match with the dynamic power demand side. 2) With the integration of distributed energy resources (e.g., energy storage systems and electric vehicles), the traditional centralized and large-scale control is undergoing a shift to a distributed and collaborative one. Moreover, complex system characteristics of massive resources make it hard to model the network, such as unknown model parameters. Therefore, to cope with the above uncertainty, volatility, distribution, and complexity, power systems are undergoing a transformation.

Safe RL is considered a sub-field within RL that is envisioned to compensate for the limitations of traditional RL in safety issues, which was first defined by J. García in 2015. Recently, safe RL has attracted surging attention in power system control. In 2020, authors first applied safe RL techniques in power systems, for electric vehicle (EV) charging scheduling. Although safe RL applications have been mentioned in a few reviews (e.g., energy system), to our knowledge, this is the first paper to provide a review of safe RL techniques applied in power systems. We first summarize various state-of-the-art safe RL techniques, then exemplify how these techniques are applied to control problems in power systems, and finally discuss the key challenges and perspectives. Overall, the key contributions of this work are threefold: 1) We present a comprehensive and structural overview of safe RL techniques, in terms of basic concepts and theoretical fundamentals, summarizing two technical categories.

This paper is funded in part by the Science and Technology Development Fund of Macau SAR (File no. 001/2024/SKL, and File no. 0053/2022/AMJ).

(Corresponding author: Hongcai Zhang.)






# Control barrier function (CBF)-based projection

# Model predictive control (MPC)-based projection

# Action projection

# Parametrized model analysis-based projection

# Safe layer

# Action replacement

# Human intervention

| Yes                  |                                      |                                               |               |
| -------------------- | ------------------------------------ | --------------------------------------------- | ------------- |
| Safe                 | On-line                              | Decoupling with RL                            | RL framework? |
| No                   | Policy optimization criterion        | TRPO/PPO/SAC                                  | Lagrange      |
| Lagrange multipliers | Constrained soft actor-critic (CSAC) | Reward constrained policy optimization (RCPO) |               |
| Trust region method  | First-order approximation            | Penalized proximal policy optimization        | FOCOPS        |
| Lyapunov method      | Second-order approximation           | Constrained policy optimization (CPO)         |               |
| Offline RL           | Safety-guided exploration method     | Projection-based CPO                          |               |

Fig. 1: The structure of safe RL methodology.

egories from whether the safe module is coupled with the traditional RL framework. 2) We present the effective appli- cation of safe RL techniques in modeling, safe module design, and implementation, by selecting three key applications in power systems, i.e., frequency regulation, voltage control, and energy management. 3) We discuss the key challenges and perspectives for applying safe RL in power systems regarding convergence and optimality, training efficiency, universality, and real-world deployment.

The rest of this paper is organized as follows: Section II introduces the basic concept of RL, and describes two cate- gories of the state-of-the-art safe RL techniques. Section III provides a comprehensive review of RL applications to three critical power system problems, i.e., frequency regulation, voltage control, and energy management, illustrating typical mathematical models. Section IV discusses the key challenges (e.g., optimality, efficiency, and universality) and potential future directions of safe RL techniques. Finally, Section V presents our conclusions.

# METHODOLOGY OF SAFE REINFORCEMENT LEARNING

This section first introduces the basic formulation and key variables of the traditional RL algorithm in subsection II-A, which is the preliminary of safe RL. Then, two categories of state-of-the-art safe techniques are introduced in detail, which consider the safety in online RL through a safe layer (in subsection II-B) and transforming policy optimization criterion (in subsection II-C). For each category, we summarize the key ideas of different techniques and analyze their applicable scenarios. Fig. 1 shows the overall classification of safe RL techniques summarized in this paper, along with relevant literature references. It is noted that, offline RL learns optimal strategy from a static dataset, which avoids interactions with the physical environment. Although avoiding direct interac- tions can guarantee constraint safety during the RL training process, the problem of distribution shift between the static dataset and the real-world environment also brings challenges to strategy solving [10]. At present, the convergence and optimality of offline RL algorithms lack theoretical guarantees.

The majority of research in offline RL focuses on addressing the problem of distribution shift, rather than on constraint safety considerations [11], [12]. Hence, as a special category of safe RL, this paper does not delve into a detailed discussion on offline RL.

# Basic formulation of reinforcement learning

RL is a branch of machine learning that focuses on training a controller to find optimal sequential decisions in an uncertain environment. As a typical modeling technique, the Markov Decision Process (MDP) is often employed to describe sequen- tial decision-making problems. The mathematical framework provided in an MDP includes four essential components to describe the interaction between the environment and agent, which are the state space s ∈ S, the action space a ∈ A, the transition function P(·|s, a) : S × A → P (S), and the reward function r : S × A → R, respectively. Here, P (S) denotes a distribution on the state space. Hence, in power systems, a specific control problem is first described as an MDP mathematically, by defining the four aforementioned components. Then, the MDP can be solved by different RL algorithms.

To show the continuous interaction in an MDP, Fig. 2 displays the relationship between the four components &#x3C; S, A, P, r >. For each time step t = {0, 1, ...}, the environ- ment captures the current system operation state st through defined observed variables, and sends it to the agent. Then, based on the received state information st, the agent makes an action decision at and executes it. Further, the environment returns the reward rt as feedback for the action. Finally, the environment goes to the next state st+1, following the transition probability P(st+1|st, at). The agent’s rule to take what action given a certain state st is called policy π(a|s) : S → P (A), mapping from the state space to a distribution on the action space. For the agent in an MDP, the optimal policy π∗ means the policy can help the agent receive the maximum expected cumulative reward Jπ:

π∗ : arg max Jπ = Eτ∼π[ R ∞ X γtr(st, at)],  (1)

where τ = {s0, a0, s1, a1, ...} is a state-action trajectory and τ ∼ π means that the action in τ is selected based on the policy.







# 3

# Action a,

# Reward r,

# Agent State Environment

Fig. 2: Illustration of a Markov Decision Process.

π(·|st). Parameter γ ∈ [0, 1) is a discount factor that considers the reward from future steps. Traditional RL techniques for solving Eq. (1) have been summarised in [5].

To solve the optimal policy of MDPs, the model-free RL algorithm is the most popular in power systems, because model-based RL algorithms and dynamic programming rely heavily on perfect environmental assumptions (e.g., accurate state transition probability P) and high-complexity computation [13]. Generally, in the initial policy learning stage, model-free RL algorithms introduce a random process for action explorations to collect adequate experiences. However, such conventional RL methods are only suitable for inherently safe systems or simulators, allowing agents to engage in unconstrained “trial-and-error”. This is because in real-world power grids, random action explorations could lead to extremely dangerous situations, or even result in significant safety incidents [14]. Therefore, the safe concept is introduced into RL, forming safe RL, to address the security considerations of RL applications in real-world power systems.

# B. Safe reinforcement learning by adding safe layer

For the RL application in power systems, the agent’s action following the learned policy is safe if the system’s state satisfies its operation constraints after action execution, especially for hard constraints [7]. However, during the training process, the agent learns more about the environment only through explorations, where most exploration techniques in RL are blind to the risk of actions (e.g., heuristics or ε−greedy) [15]. Exploration is essential for the agent’s training, so the random perturbation to actions is hard to avoid in RL. Therefore, a straightforward idea is to design a safe layer before executing every action. The safe layer is expected to verify whether the action is safe for all constraints, and tune unsafe actions to safe ones. Based on the verified result, the safe layer can modify the action through different techniques to ensure the final executed action is safe.

Adding a safe layer is a main tendency in safe RL, because this is a universal technique decoupling with the traditional RL framework. That is, the introduced safe layer can combine with various RL algorithms, whether based on the actor-critic scheme [16], policy gradient [17], or policy optimization [18]. Fig. 3 displays the combination scheme of the safe layer and the RL agent, which involves two key steps: 1) At each time step, the safe layer intercepts unsafe actions at to become safe ones asafe; 2) As the agent’s feedback, the safe layer modifies the original reward to reflect the penalty of the interception degree, from rt to rmod. Based on the safe layer scheme, many researchers have proposed different design techniques to tune actions, including two main categories: action replacement and action projection [19], which are introduced in detail as follows.

# 1) Action replacement:

Fig. 3(a) shows the concept of action replacement, where the key technique is how to obtain safe actions asafe in a safe action space Asafe to replace the original unsafe ones at. Recently, researchers have proposed to obtain a single safe action via human feedback [20] or from a shielding/blocked mechanism [21], [22] (e.g., a failsafe planner [23]).

# 1) Human intervention:

For safety-critical scenarios, human expert experiences can ensure great safety to avoid unsafe actions made by RL agents [24]. Hence, leveraging human expertise to guide the exploration of agents is a natural idea to enhance safety, where the most common methods include the interruption mechanism and expert guidance. The interruption mechanism aims to interrupt the final executed action directly when it is considered dangerous by drawing on human experiences, and then replace with a safe action. This method can cope with “catastrophic actions” that the human overseer deems unacceptable under any circumstances, which only relies on human experts but requires no model information. However, RL training iteration is up to millions; it is not practical for a human expert to constantly supervise an RL agent for a million timesteps. For this consideration, achieving automating oversight using human expert experiences has been paid attention in recent years. Saunders et al. [24] proposed to train a “blocker” to mimic human interruption, which includes two steps: 1) manual supervision stage to collect binary label of “whether the manual interruption is implemented”; 2) “blocker” training stage to mimic the human interruption. It should be noted that the manual supervision stage cannot stop until the ”blocker” performs well on the testing dataset. However, the limitation is that the “blocker” can only handle relatively simple accidents. When facing more complex environments, the “blocker” takes more than one year to implement the manual supervision stage, with high time costs. To solve the above issue, Prakash et al. [25] proposed a hybrid safe RL scheme for reducing the time cost, by combining a model-based prediction module with the “blocker”. In addition to training a “blocker” to identify dangerous actions, humans prefer to stop the action immediately when confronted with potential danger. Inspired by this, Sun et al. [26] and Eysenbach et al. [27] both developed a safe RL framework for early response to potential dangers, where






the former introduced a solution to early terminated MDP (ET-MDP) and the latter proposed to automatically reset the environment. The expert guidance introduces curriculum learning [28] into the conventional RL framework to guarantee safety, whose main idea is to imitate the learning process of humans, from simple tasks to difficult tasks [29], speeding up the training efficiency. In 2020, Turchetta et al. [30] adopted curriculum learning to ensure safety in RL for the first time, where an agent (i.e., student) learns from the automatic guidance of a supervisor (i.e., teacher). The supervisor needs to automatically design the course for the agent, according to the agent’s learning progress and behavioral data distribution. Hence, the safe exploration of the agent completely depends on whether its supervisor is well-trained. However, the challenge in curriculum learning is how to train the supervisor with limited samples, when designing complex learning tasks. To cope with this issue, Peng et al. [31] proposed an expert-guided policy optimization, which combines offline RL to stabilize the training of the supervisor through the off-policy partial demonstration. Nonetheless, high-cost offline human intervention increases the reliance on experts. Then, in 2022, Li et al. [32] developed a novel human-in-the-loop learning by designing a special mechanism to mitigate the delayed feedback error, which can effectively reduce the reliance on experts over time and improve the supervisor’s autonomy. With this self-learning method, one may cause a quite conservative policy.

In summary, human intervention requires experts to help the agent become self-learning automatically. Although this approach can be both safe during training and deployment, the high cost of human intervention should be taken into account in real-world applications.

# 2) Shielding:

The concept of “shielding” in RL was first proposed by Alshiekh et al. [21] in 2018, to ensure constraint safety during RL training. The key idea is that, the shielding process will be triggered when the output action is unsafe, and then an alternate safe action is used to override the original one. Hence, the implementation of the shielding mainly involves two significant works: 1) the design of the shielding trigger, and 2) the design of backup (safe) policies. The design of the shielding trigger is hard to design. Generally, the shielding method is more suitable for scenarios when safety conditions and constraints can be clearly defined (e.g., no speeding), because the shielding trigger would be easier to design. For complex scenarios with constraints that are hard to define, some researchers have proposed model predictive shielding (MPS) to handle deterministic closed-loop environment dynamics [33], [34] or stochastic environment dynamics [35], [36]. This promising MPS method can perform shielding on-the-fly instead of ahead-of-time, by checking whether a single state is safe in real-time [37]. For common deterministic shielding methods, there are only two system states, either safe or unsafe. However, the same action may lead to different state results following ambient uncertainties. Hence, probabilistic shielding is further proposed to cope with uncertainties [38] through formal verification to compute the probabilities of critical decisions.

The design of backup policies is also one of the research focuses in shielding, where MPC-based backup controllers are the most common choice. Li and Bastani [35] used a robust nonlinear MPC to compute a backup policy in stochastic environment dynamics. To improve the safety of the backup policy, Bastani [33] further defined the backup policy with two choices: an invariant and a recovery policy. The invariant strategy can keep the agent moving around the safe equilibrium point, and the recovery strategy can move the agent to the safe equilibrium point. The key idea is that the controller can determine which backup policy to use based on the distance between the agent and the safe equilibrium point.

In addition, as step 2 shown in Fig. 3, the reward after shielding rmod needs to provide feedback of shielding interception for agents. One usually has two different designs: 1) assigning a large punishment to learn that selecting at a state st is unsafe, rmod &#x3C; rt; 2) remaining the same with original reward, rmod = rt. For the former approach, the agent can learn from the punishment feedback, so the shield is no longer needed in the execution phases for well-trained agents. For the latter, the agent cannot learn to avoid unsafe actions, which are always corrected to safe ones by the shield without feedback. Hence, the shield is still needed in the execution phases even for well-trained agents.

Therefore, compared with human intervention-based methods, shielding is an automated security mechanism with low costs, which dynamically adjusts the decision space through predefined rules or real-time calculated risk assessments. However, the limitation of shielding is the low adaptability in dynamic and complex environments, which is a model-based technique. For complex tasks, it is difficult to provide sufficient prior knowledge to build comprehensive shielding from all dangers [39], as human experts do.

# 2) Action projection:

After reviewing the research work that applies safe RL to power systems, we find out that action projection is the most popular method to deal with constraint safety issues [40]–[43]. As shown in Fig. 4(b), action projection aims to project the original unsafe action to an action in the safe space Asafe that is closest to itself, where the projection rule design relies on model-based optimization programming. In a theoretical way, the common design of projection rules can be categorized broadly based on three techniques: CBF, MPC, and parametrized model analysis (as summarized in Fig. 1). Among the above three designs, the similarity is that, they all need to obtain the closest safe action asafe by solving an optimization problem at every time step t. The difference is how to define the objective and constraints of the optimization problem, and what the system model assumptions of physical environments are. Three projection rule designs are introduced in detail as follows.

1) Control barrier function: The basic idea of CBF is to define a safe region (so-called “safe set”) by creating a “barrier”, which can effectively prevent the agent from stepping into the unsafe region and staying inside the safe set. This technique can effectively deal with hard constraints that must be satisfied, because the safe set defined by CBF possesses forward invariance property [44]. Specifically, a safe set C is required to be defined by the super-level set






# Action set

# Safe action set

# Projected/replaced action

(a) Action replacement (b) Action projection

Fig. 4: Concept and differences between the action projection and action replacement. Here, action replacement (a) replaces unsafe actions with self-defined actions from the safe action space. Action projection (b) projects the agent’s unsafe actions to the closest action in the safe action space.

of a continuously differentiable function h :  Rn → R, by C : {st ∈ Rn : h(st) ≥ 0}. To maintain the constraint safety during every training step, the RL agent can only be limited to learning and exploring within the safe set C. Here, one challenge is how to design functions h in CBF that can guarantee the safe set C possesses forward invariant property. For instance, Cheng et al. [45] proposed an affine CBF based on discrete-time formulations. Emam et al. [46] further extended the design of h from an affine form to a finite union form of convex hulls, which can effectively capture non-convex disturbances in environment systems.

Here, we simply give an example in an affine environment system to show the corresponding design of safe sets, which also supports a more general form. If there exists η ∈ [0, 1] satisfying the condition: supat∈A[h(st+1)+(η −1)h(st)] ≥ 0, the differentiable function h is defined as a discrete-time CBF. Then, the proposed safe set can compensate for the unsafe action by ∆at, based on the CBF controller πCBF as:

asafe = at + πCBF(st, at) = at + ∆at,

where ∆at represents the compensation value at time step t. Further, in the safe layer, the objective of the projection is designed to provide minimal control intervention to original actions at. The constraints consist of the upper/lower limits of final actions and the predefined safe set. Finally, at each time step, the model-based optimization programming for the action projection is formulated as:

(∆at, ε) = arg min ∥∆at∥2 + Kεε,

s.t.: h(st+1) ≥ (1 − η)h(st) − ε,

asafe = at + ∆at,

a ≤ asafe ≤ at,

where ε is a slack variable for the safe set, and Kε is a large constant (e.g., 1012) for penalization when safety cannot be enforced. Note that, different CBF designs will derive different constraint formulations in Eq. (3b), where the safety of CBF needs to be reproved theoretically [45]–[47]. Although CBF can guarantee safety for an infinite time, finding a suitable and well-designed CBF is not easy in practice.

Conventionally, CBF is a model-based method that requires the model of system dynamics (e.g., in Eq. (3b), variable h(st+1) should be expressed). Hence, without prior knowledge of the system model, integrating the model-based CBF with a model-free RL framework is another challenge. Currently, researchers have proposed several model estimation methods to compensate for the unknown system dynamics, such as the Gaussian process [45], [48], the iterative search algorithm [49], sparse optimization [50], and state transformation [51]. The model estimation accuracy of system dynamics is also a significant factor for the constraint safety effects, where a high-accuracy model brings a high-probability safety guarantee. However, measurement errors and environmental noise are hard to be considered perfectly in constraint models.

# Model predictive control

MPC is considered a common methodology for constrained control, which can exploit the data reliably to take safety constraints into account [52]. Recently, researchers have proved that the combination of the MPC and RL algorithms can achieve a safe and high-performance system operation. Compared with the CBF technique, MPC-based safe RL also designs the action projection by a model-based optimization programming. However, the MPC technique guarantees constraint safety through a predictive safety filter, but not through defined safe sets. Hence, unlike the required forward invariant property in CBF, an MPC-based learning process has better universality without design requirements. Specifically, based on the MPC technique, at each time step t, the projected safe action for original action at is solved by the following optimization problem:

min ∥at − asafe∥0,t,

s.t.: s0,t = st,

sN,t ∈ Ssafe,

sj+1,t = f(aj,t, sj,t), ∀j ∈ J[0,N−1],

aj,t ∈ A, ∀j ∈ J[0,N−1],

sj,t ∈ S, ∀j ∈ J[0,N−1]

where sj,t is the j time steps-ahead state prediction, computed at time t (i.e., s0,t = st); Ssafe is a terminal safe set; f(a, s) is the system dynamic model; N is the prediction horizon. The problem (4) solves an N-step input sequence {asafe} that drives the system to the terminal safe set, guaranteeing safety for all future time steps (detailed proof in [53]). Then, the solved first input asafe is selected as the safe action at time t, which is expected to be as “close” as possible to the agent’s original output.






However, MPC-based safe RL methods rely heavily on the system model f(a, s) shown in Eq. (4d), because the model accuracy directly influences the prediction result of future N-steps state safety. Hence, one drawback of MPC is that it is usually not robustly safe, because it cannot encode uncertainties (e.g., environment noises and measurement errors) into the optimization problems, especially for nonlinear systems [52]. Currently, some works have focused on improving the robustness of MPC-based RL in safety guarantees, including linear [37] and non-linear systems [54]. Although virtually any RL controller can be enhanced with safety guarantees using MPC, the resulting performance of the overall system remains to be investigated.

# 3) Parameterized model analysis:

Theoretically, for constraint safety, the most reliable method is to design an action projection rule through the analysis of known parameterized models. This is because the parameterized model-based optimization can predict the future operation state of systems more accurately to judge the safety [46]. However, the system model assumption becomes quite strong for most practical scenarios, requiring an accurate parameterized model of system operation constraints.

Recently, in power systems, some researchers have tried to apply parameterized model analysis forming an action projection in the safe layer, including solving the optimal power flow problem [55], the voltage control problem [56], demand-side resource management [41], [57], and electricity market bidding [43]. Although systems constraint models in the above-mentioned scenarios vary, their proposed objectives for the optimization programming are the same, which aim to minimize the distance between corrective safe actions asafe and original unsafe actions a.

While an advantage is that parameterized model-based optimization can effectively handle hard system constraints, the drawbacks are: 1) the projection design in safe layers relies heavily on the parameterized system model, which is not universal and cannot directly extend to different scenarios; 2) the system uncertainty is hard to be parameterized and considered into the optimization problem, especially involving stochastic human behaviors.

Therefore, the key problem in action projection is formulating the constraint based on different system conditions. Specifically, the CBF-based method requires little system model knowledge, while putting a high requirement on the barrier function design, such as Lipschitz condition. The effectiveness of MPC-based methods depends on the accuracy of the applied model, where the robustness in non-linear systems remains to be investigated. The parameterized model-based method is more reliable for constraint safety, while the model assumption is quite strong in practice. However, when a system is a black box to its controller, the optimization problem cannot be formulated to apply the action projection technique.

Safety by changing the agent’s policy optimization criteria, which is coupled with RL algorithms. As shown in Eq. (1), the goal of conventional RL is to maximize cumulative rewards, ignoring the damage that constraint violations cause to the agent. That is, the objective function in MDP formulation lacks the description of constraint violation risks or losses. Hence, to describe the constraint violation mathematically in safe RL problems, the traditional formulation of MDP in RL is reformulated into the following CMDP.

# 1) Extended formulation of safe reinforcement learning:

A CMDP extends the MDP framework &#x3C; S, A, P, r > by introducing constraints to restrict the allowable policies [58]. Specifically, the MDP is augmented with an auxiliary cost function C and the corresponding threshold d, where the cost function C : S × A → R maps station-action pair to violation costs. Similar to the expected cumulative reward Jπ in Eq. (1), the expectation over the violation cost Jπ is denoted by:

Jπ = E [X∞ γtC(s, a)] (5).

Thus, the reformulated policy optimization problem for a CMDP becomes:

π∗ : arg max Jπ = Eτ∼π[ X∞ γtr(st, at)] (6a)

s.t.: Jπ ≤ d. (6b)

As discussed in subsection II-B, for those safety-critical problems with hard constraints (i.e., constraint needs to be enforced at each time step), the model-based safe layer is more advantageous to ensure safety with the help of model information. However, the cost function in Eq. (5) is a model-free formulation by defining the safety based on the expectation, which is more suitable for complex control tasks with multiple soft constraints in power systems, such as energy/demand management [59], [60]. Essentially, Eqs. (6) aim to solve a constrained optimization problem, where the main challenge is that both the objective and constraints are non-convex for the RL agent. To effectively transform the original constrained optimization problem into an unconstrained one, commonly used methods in power systems can be generally summarized as follows: the Lagrange multipliers method, the trust region method, the Lyapunov method, and the safety-guided exploration method.

# 2) Lagrange multipliers method:

Lagrangian relaxation is a common solution for constrained optimization problems [61]. In the RL framework, a Lagrange multiplier λ ≥ 0 is introduced to manage a trade-off between the reward and constraint violation costs. Specifically, the original constrained optimization problem in Eqs. (6) are converted into an unconstrained one as:

min max L(π, λ) = λ≥0 π (JR − λ(JC − d)). (7)

# C. Safe reinforcement learning by transforming policy optimization criterion

The above safe layer-based technique adopts an extra safe layer that is decoupled with RL algorithms. This subsection presents another safe RL technique considering constraint. With the increasing of λ, the solution of Eq. (7) converges to the result of the original problem in Eqs. (6). Note that, during training, the update iteration for the policy π is suggested to adopt a faster timescale than that for Lagrange multiplier λ. That is, as k-th iteration, assuming that λk is constant, the policy π is updated for several iterations by solving Eq. (7).






to maximize L(·, λk). Then, the λk is increased in a slower timescale to satisfy the constraint and repeat the iteration process. The update of λk is set as:

λk+1 = [λk − ηλ(Jπ − d)]+.

where ηλ is the step size for updating λk, and [·]+ projects λk into a non-negative real number. Based on the continuous iteration of updating the policy π and Lagrange multiplier λ, this method can guarantee the convergence to a local optimal and feasible solution when the following three assumptions hold [62], [63]: 1) Jπ is bounded for all policies π; 2) every local minimum of Jπ is feasible; 3) ηλ = ∞ and ηθ = ∞ and η2 + C η2 ≤ ∞. Here, ηθ denotes the step size of the policy neural network. Hence, the design of the step size and initial value for λ is significant for the feasibility of the local optimal solution, where the hyperparameter tuning process is also one of the challenges in practice.

The key idea of the Lagrange multipliers method is not complex to implement, so this method has been applied in various control tasks in power grids [64]–[66]. In addition, this method has been similarly extended to different state-of-the-art RL algorithms, such as soft actor-critic (SAC) [67], trust region policy optimization (TRPO) [68], and proximal policy optimization (PPO) [69]. However, as shown in Eq. (6b), the constraint safety is defined on all possible states’ expectations, leading to a fatal flaw: each specific state is allowed to be unsafe as long as the expectation of states satisfies the constraint. That is, the expectation-based constraint safety cannot prevent some worst cases in safety-critical domains.

To address this issue, some researchers recently have tried to introduce a chance constraint [70] or conditional value-at-risk (CVaR) [71] to describe the tail risk of constraint violations, for improving the policy robustness. The improved methods can effectively take extreme scenarios into account, according to different risk requirements or preferences. In practice, the aforementioned methods all require quite strict mathematical assumptions for converging to local saddle points. Then, under mild assumptions, Tessler et al. [72] have proposed reward constrained policy optimization (RCPO) and proven it can converge almost surely to a constraint-satisfying solution. Besides, the RCPO algorithm is reward agnostic and does not require prior knowledge. The disadvantage of the RCPO is that multiple learning rates are involved, which are difficult to adjust in practice. As discussed before, most Lagrange multipliers-based methods can only cope with soft constraints, because they cannot prove to achieve zero constraint violation.

To explore whether it is possible to achieve the optimal sublinear convergence rate with zero constraint violations, Bai et al. [73] designed a conservative stochastic primal-dual algorithm (CSPDA) by utilizing the conservative idea of reducing the regret [74], and gave the theoretical convergence analysis.

In summary, the Lagrangian multipliers method transmutes the constrained optimization problem into an unconstrained one by introducing an auxiliary penalty component, thereby enabling the solution to satisfy the constraints and maximize rewards. This method can assure constraint safety as the policy asymptotically converges. Despite its advantages, there are still several limitations: 1) Substantial computation burden for solving a saddle point optimization problem, which equals solving a succession of MDPs; 2) Significant hyperparameter tuning overhead caused by the sensitivity to the initial values and learning rates of the Lagrange multipliers; 3) The convergence rate of the iteration solution cannot be guaranteed, because the objective of the Lagrangian multiplier problem is neither convex nor concave.

# 3) Trust region method:

Different from the penalty of the Lagrange multiplier, the trust region method solves the constrained optimization problem in Eq. (6) through direct modification of the policy gradient, by enforcing a trust region constraint [18]. Specifically, in the policy iteration, the range of policy parameter changes is limited within a neighborhood of the most recent iterate (i.e., the trust region). This trust region constraint ensures that each step’s change is not too large, thereby maintaining the safety and reliability of the policy optimization process. The enforced constraint is formulated as follows:

πk+1 = arg max Jπ

s.t.: Jπ ≤ d,

D(π, πk) = ||θ − θk||2 ≤ δ,

where D is the distance measurement; δ ≥ 0 is a step size; θ denotes the network parameters of policy π. At each iteration k, solving policy πk+1 is difficult because it is required to evaluate whether a policy is feasible for trust region constraint. To address this challenge, Achiam et al. [75] extended the TRPO method and first proposed a general policy search method, called constrained policy optimization (CPO). The key idea of CPO is: firstly, conducting surrogate functions to approximate the non-convex objective function Jπ; secondly, expanding the problem in Eq. (6) into a convex optimization problem by Taylor second order.

In CPO, it adopts backtracking techniques to search for new policies, significantly slowing down training efficiency. To improve the efficiency issue, Yang et al. [76] introduced two steps (i.e., reward improvement step and projection step) for policy searching, and proposed the projection-based constrained policy optimization (PCPO). To solve the trust region problem, CPO and PCPO both need to calculate the inversion of the Fisher information matrix (FIM). However, when facing high-dimensional policies, calculating FIM becomes impractically expensive, requiring low-cost approximation for FIM. To reduce the approximation error of FIM, Zhang et al. [77] proposed first-order constrained optimization in policy space (FOCOPS), whose main idea is to use the primal-dual gradient to solve the trust region problem. Compared with CPO and PCPO, FOCOPS only uses linear approximation and does not need to solve the inversion of FIM, which is more efficient and practical in computation.

Although the simulation results on the high dimensional continuous control task show that the first-order approximation’s performance in FOCOPS is better than that of the complex second-order approximation (e.g., CPO), this observation






# Fit models to estimate reward and safety

# Safety estimation with Gaussian process

Action-value function:

Safety estimation:

Generate and collect samples

Improve the policy

Fig. 5: The safety-guided RL framework for safe explorations.

has not been theoretically substantiated. Similarly, based on the traditional proximal policy optimization (PPO), Zhang et al. [78] proposed penalized proximal policy optimization (P3O) algorithm to handle the difficulty of calculating inverse FIM. In P3O, the cost constraint is transformed into an unconstrained optimization problem by the exact penalty function and solved by first-order optimization, which avoids quadratic approximation and high-dimensional Hessian matrix inversion in large CMDP problems. The aforementioned methods are all trust region methods for solving CMDPs, where the approximated constraints are enforced in every policy update round. Hence, the converged policy can ensure constraint safety during training. Because the above methods are related to TRPO, it is not difficult to apply them to the PPO for constraint optimization. However, it is still not clear how to combine the above methods with RL frameworks that are not of the proximal policy gradient type, such as the RL algorithms with actor-critic framework. In addition, there are some limitations to the above methods: 1) The convex approximation of non-convex policy optimization will produce non-negligible approximation errors, so whether the first or second-order approximation can only learn the policy that is close to satisfying the constraints. 2) When the original problem is not feasible under a certain initial policy, we need to restore the policy to the feasible set through interaction with the environment, causing a low sampling efficiency. 3) The second-order approximation involves matrix inversion, which is costly in a high-dimensional environment and is not suitable for solving large-scale CMDP problems.

4) Lyapunov method: The Lyapunov function, essentially a concept in the analysis of the system stability, is used to measure the “distance” of a system state relative to some stable point or set. In the context of safe RL, during the process of exploring, a Lyapunov function can be introduced to ensure the system state is not far from the predefined safe state [79]. Generally, adopting the Lyapunov method for solving a CMDP problem includes three key steps [80], [81]: First, construct a Lyapunov function L ∈ Lπ(s0, d) that maps the system state to a real value, which measures the “distance” between the system state and the predefined safe state. Second, reformulate a constrained optimization problem to add Lyapunov constraints satisfying the defined Lyapunov function. Last, propose a policy update approach to embed the Lyapunov constraints into the policy network. However, the most necessary but challenging step in practice is how to design a satisfactory Lyapunov function based on the known system dynamic, where the Lyapunov function is required to possess several properties (e.g., positive definiteness, decay property, and Lipschitz continuity condition) [82]. Hence, the design of a Lyapunov function requires a deep understanding and analysis of the system dynamic, and can only be combined with model-based RL algorithms. In addition, for different systems, it is required to design the system’s own specific Lyapunov function that is not universal, especially for complex or highly nonlinear systems. Once an analytic Lyapunov function is designed properly, this method can effectively guarantee the stability of the system with optimal control performance, which is critical and a research hotspot for the RL application in power systems (e.g., frequency control [83], [84]).

5) Safety-guided exploration method: To cope with the design issue of the Lyapunov function, reference [85] formulated a model-free function for safety costs as the candidate Lyapunov function, and modeled its derivative with a Gaussian process which provides statistical guarantees. This model-free method steers the policy search in a direction that decreases the safety costs and increases the objective reward, which can effectively solve power system control tasks involving multiple complex systems [86].

The traditional RL algorithm updates the policy only by estimating the objective reward through the action-value function Qπ(s, a) = Eτ∼π hPT γtrt(s, a)|s =s,a =ai. Hence, the conventional policy gradient direction follows:

∇θJπ = Es∼ρπ [∇θπ(s)∇aQ(s, a)|a=π(s)],

where ρπ R θ. To achieve safety-guided exploration, a model-free safety estimation Gπ(s, a) with the Gaussian process is introduced to ensure safety. The safety estimation is defined as Gπ(s, a) = E hPT γtc (s, a)| i, which is approximated in practice with a deep neural network. Then, the original policy gradient direction is re-derived considering both the action-value function Qπ(s, a) and safety estimation Gπ(s, a), rewritten as:

∇θJπ = Es∼ρπ [∇θπ(s)∇aQ(s, a)|a=π(s)∇aG(s, a)|a=π(s)].

Hence, the whole safety-guided RL framework is that: 1) the collected samples are used to fit models Qπ(s, a) and Gπ(s, a), which estimate the objective reward and safety cost, respectively; 2) the Gaussian process estimation is updated in every iteration; 3) the policy π is finally optimized following the rewritten gradient ∇θJπ in Eq. (11) which combines.






Power Systems Increase T&#x26;D system uncertainty

Buildings Batteries EVs Traditional PVs Wind Gird Demand response Generators Supply Demand

Action Stable frequency &#x26; voltage, Energy balance Observation (Power flow, frequency, voltage, lines, etc.)

Safety RL-based control policy Safety consideration

Fig. 6: The application of safe RL in power systems.

The reward and safety estimations. However, this method can only combine with the RL algorithm using the actor-critic framework, while not for all RL frameworks. Although the stability certificates of Gaussian process estimation can provide high-probability trajectory-based safety guarantees for unknown environments, how the initial knowledge influences the efficacy of this method remains to be explored.

# III. APPLICATIONS OF SAFE REINFORCEMENT LEARNING IN POWER SYSTEMS

With the development of artificial intelligence and Internet of Things technologies, model-free RL-based control methods are widely applied for complex tasks in power systems, to cope with operation environments with high uncertainties. Traditional RL methods rely heavily on large neural networks with millions of parameters, which seem the inexplicable “black box” and cannot ensure system safety. Hence, for safety-critical problems in power systems, safe RL techniques become an appealing complement to the application of traditional RL. As illustrated in Fig. 6, the safe RL scheme considers the “safety” concept before the final decision is executed, which relieves the exploration risk of control decisions but still converges to optimal control policies. Specifically, in power systems, the constraint safety of the frequency and voltage is the most critical indicator of system operation, and real-time power balance requires reliable energy management [5]. Therefore, this section focuses on the following three key applications: frequency regulation, voltage control, and energy management, as summarized in Tables I, II, and III, respectively. For power systems, frequency regulation is a continuous control problem with hard constraints that must be satisfied, while energy management is usually a sequential decision-making problem with soft constraints. Voltage control probably has both hard and soft constraints according to system scales and types. Facing different scenario requirements, various safe RL techniques are adopted to tackle safety challenges that cannot be solved in traditional RL frameworks. In the following subsections, we elaborate on the detailed design of MDPs and safe modules in different applications.

# A. Frequency regulation

Frequency regulation (FR) is a critical aspect of power system operation and stability. It aims to maintain the system frequency within acceptable limits, essential for preventing blackouts or equipment damage. There are typically three timescales associated with FR control [87]: 1) Primary control, known as “droop control,” is an automatic response triggered by speed governors, which occurs within seconds after a disturbance; 2) Secondary control, known as “automatic generation control” (AGC), operates over a time frame of several minutes after droop control, which is usually achieved by adjusting the setpoints of generators; 3) Tertiary control, known as “economic dispatch,” operates over a time frame of tens of minutes to hours, which is responsible for optimizing the generation to meet the demand with minimum costs. Hence, the tertiary control does not focus on the stability and safety of the frequency, which can correspond to grid-level power dispatch discussed in Section III-C.

Many studies have applied RL methods to implement FR control, to cope with the increasing penetration of uncertain renewables [5]. In practice, the safety of FR control policy is vital for power systems, such as the stability of closed-loop systems and satisfying power flow constraints. However, most traditional RL methods cannot handle the safety issue properly. Therefore, some recent works have improved the RL framework and proposed safe RL techniques for FR control, summarised in Table I. In this subsection, we take secondary control as an example, because it is the most typical FR problem and has been extensively studied [65], [88], [89]. We will first present the generalized system dynamic model formulation, state/action spaces, and reward/cost functions. Then, the safe RL techniques and key issues in safe RL-based FR are further analyzed.

1) State and action spaces: When a generator outage occurs, the system frequency dynamics reflect the relationship between the power balance and the frequency fluctuation ∆f, which can be expressed as:

2H d ∆f + D∆f = ∆P gap, (12)






# 10

# TABLE I: Literature summary of frequency regulation.

| Reference    | Problem                                        | Constraint                                                           |
| ------------ | ---------------------------------------------- | -------------------------------------------------------------------- |
| \[84] (2021) | Primary FR                                     | System stability                                                     |
| \[90] (2022) |                                                |                                                                      |
| \[65] (2022) | Load frequency control                         | Line power flows                                                     |
| \[88] (2022) | Multi-area microgrid FR                        | Operation constraints for generators and ESSs                        |
| \[11] (2022) | Thermostatically controlled loads providing FR | Voltage magnitude limits                                             |
| \[42] (2023) | General solution to control problems, e.g., FR | Frequency threshold                                                  |
| \[91] (2023) | Thermostatically controlled loads providing FR | Regulation performance score and users’ temperature comfort          |
| \[89] (2023) | Load frequency control                         | Frequency stability and regulation bound of renewable energy sources |
| \[83] (2024) | Transient frequency control                    | Stability condition                                                  |

# Methodology

Key Features
Safe layer, Lyapunov function: Train the neural Lyapunov function to satisfy the positive definiteness of its value and the negative definiteness of its Lie derivative.
CSAC, Lagrange method: Restrict the entire policy space to a smaller allowable safe space using the Lagrange multiplier method.
NN-based safe layer: Propose a safe module consisting of two components: a safety evaluation network and an action guidance network.
Offline(batch) RL: Leverage historical network measurements to train the offline RL controller, and reduce voltage magnitude constraint violations.
Shielding, CBF: Complement the RL action by the optimal solution guaranteeing the safety of systems, based on a Gaussian process model.
Safe layer, CBF: 1. Utilize previous CBF controllers to avoid repeatedly taking unsafe actions; 2. Propose a neural network-based method to achieve high-efficiency calculation.
Safe layer, CBF: Design a self-tuning CBF-based compensator to realize the optimal safety compensation under different risk conditions.
Safe layer, Lyapunov function: Define the search space of distributed control policies that guarantee asymptotically stable and transient-safe closed-loop systems.

where H and D are the system inertia constant and load damping constant, respectively; ∆P gap denotes the imbalanced power gap in power systems. It can be seen that the power imbalance is the key factor influencing the system frequency, so the action is defined as the power regulation of the traditional generation and renewable energy resources (RENs) by:

aF = P gen, P ren|k ∈ NG, i ∈ NREN ⊺ ,

where NG and NREN are the number of controllable generators and RENs, respectively; P gen and P ren are command output power of each generator k and REN i. Generally, RESs are connected to the power system through power electronic devices, thus the action space is continuous whose scale increases with more generators/RENs being controlled. The design of the state captures the current frequency change and power outputs as:

sF = ∆f, P gen, P ren|k ∈ NG, i ∈ NREN ⊺. If other power injections can be controlled for regulation services, such as demand-side loads and energy storage systems (ESSs), their output powers can also be included in the action space [88].

2) Reward and cost functions: The reward function design determines the control objective of the training RL agent, which is significant for successful RL applications. For FR control problems, the key aim is to maintain the frequency at a nominal value after disturbances. Hence, to maximize the expected reward, the minus of frequency deviation is commonly used to design the reward as:

rF = -∆t · Xi∈N |fi - fref|,

where NB is the number of the bus; fi and f ref are the measured frequency at bus i and the nominal value, respectively.

For multi-area FR, the tie-line power flow is also required to be considered in Eq. (14), introducing weight factors for two or more items [92]. In addition, the generation regulation cost [93], the large penalty for crossing deviation limits [89], and the square or exponential form for deviation frequency [94] are also an alternative in reward functions. There is no definitive assertion about which reward function is better, and the reward design is contingent upon the specific scenario.

The constraints for FR mainly come from two aspects, one is the safe range for fluctuated frequency and the other is the acceptable regulation range of control objects, which can be generalized as:

f ≤ f ≤ f ,

P gen ≤ P gen ≤ P gen, ∀k ∈ N ,

P REN ≤ P REN ≤ P REN, ∀i ∈ N ,

where f and f denote the upper and lower bounds of system frequency; P gen/P REN and P gen/P REN represent the upper and lower bounds of output power in generator and RENs, respectively. If more types of controllable objects are considered, there shall be more physical operating constraint limits, e.g., state-of-charge and charging rate in ESSs [88], comfort requirement in demand-side loads [91], etc. Then, the cost function is the total penalty for all the constraint violations:

cF = X βm · [max(0, bm − bm) + max(0, bm − bm)]2,

where m is the constraint number in Eq. (15); β is responding weight factor; bm, bm, and bm are the formal variables to denote the specific three variables, three lower limits, and






# 3) Safe RL techniques:

Apart from batch RL (i.e., offline RL), most published studies choose to apply safe layer-based approaches in FR problems, because a safe layer needs to check the system safety at every time step, which is more effective for hard and critical constraints. The specific design of the safe layer in FR mainly includes three ideas:

1. With an accurate constraint model, the next state is directly and mathematically derived to judge whether the state is safe. Reference [84] used the physical system model to derive the structure of the stabilizing RL controller based on Lyapunov theory.
2. With a poor constraint model, the system model is approximated through the Gaussian process or other methods, then the safe area is defined by proposed barrier functions. For instance, reference [89] proposed an adaptive and safe-certified RL algorithm for FR control, where the safe layer is designed based on Gaussian process regression and CBF-based compensator. Reference [42] proposed a more general solution for power system control problems (including FR), which certifies the neural barrier function that perseveres barrier conditions. This method does not rely on the model approximation as the first step.
3. Without any model information, an NN-based safety layer is introduced to judge the state safety and correct the action. More specifically, reference [88] proposed a safety evaluation network as a safety monitor, and an action guidance network for safe action-guiding. In this manner, the training of the proposed safe layer relies on the dataset instead of the physical model.

# 4) Discussion:

- Stability: The integration of RL agents with traditional existing controllers requires the closed-loop system dynamics to be stable, which is different from constraint safety and rarely considered in the current works. Cui et al. [90] have tried to propose a Lyapunov regularized RL approach for FR control to cope with the transient stability, where the Lyapunov function is parameterized through NNs. However, this work only focuses on stability ignoring constraint safety, so combining stability and safety is a potential research direction for RL applications in FR control.
- Demand-side regulation services: The aforementioned design of state and action spaces mainly focuses on controlling generators to regulate the power supply. With increasing demand-side resources participating in electrical markets, ubiquitous load devices are also an alternative as a compensatory to FR control. Reference [91] proposed to control large-scale district cooling systems for FR services, which satisfy users’ thermal comforts by introducing CBF in safe layers. Besides, in [11], thermostatically controlled loads are controlled through a batch RL method to provide FR, which ensures constraint safety by offline training. More potential load control for FR services remains to be explored.

Fluctuations into distribution grids. The voltage control problem in power systems aims to keep network voltages within an allowable range and reduce system losses, by determining the control actions for all voltage regulating and var control devices. Many published works have developed model-free RL methods for voltage control to tackle the challenges of renewables’ uncertainties and unknown accurate system models [95], [96]. However, traditional RL applications do not have an explicit mechanism to ensure the safety of operation constraints, e.g., voltage limits and line flow limits. This is because most RL controllers are defined by large-scale neural networks (NNs), which is considered as a “black box” and trained through the feedback of “trial and error”. To this end, considering constraint safety, some recent studies have tried to explore a safe model-free RL framework for voltage control, which is summarised in Table II. We will present the most commonly used definitions of state/action spaces, reward/cost functions, and the application of safety techniques.

# 1) State and action spaces:

For voltage control in distribution systems, different controllable devices possess different operational characteristics, which can be classified into discrete control and continuous control. Discrete control devices include switchable capacitor banks (SCBs), on-load tap changing transformers (OLTCs), and voltage regulators (VRs), which are controlled in a slow timescale on an hourly or daily basis. Continuous control devices, such as battery storage systems (ESSs), distributed resources (DRs), and static Var compensators (SVCs), can regulate their active/reactive power in a fast timescale by seconds. Since the voltage control problems involve both the distribution network and controllable devices, the observed system state s shall capture the significant information of both above two sectors, which can be defined as:

sVol = [t, (Pi, Qi, vi), τSCB, τVR, τOLTC, PDG, QDR, QSVC, EBSS]⊺,

where NB is the number of the bus. State st consists of four parts: (1) the time information of current time step t; (2) the network information of net active power Pi, net reactive power Qi, and voltage magnitude vi at bus i; (3) the device information of the discrete tap positions of SCBs/OLTCs/VRs by τSCB, τVR, and τOLTC, respectively; (4) the device information of the active power outputs of DGs PDG, reactive power outputs of DGs/SVCs by QDG, QSVC, and stored energy in ESSs by EBSS.

The action space of the voltage control problem depends on the selected controllable devices. When all the aforementioned devices are considered, the RL agent’s action a can be defined as:

aVol = [∆τSCB, ∆τVR, ∆τOLTC, PDG, QDG, QSVC, PBSS]⊺,

where ∆τSCB, ∆τVR, ∆τOLTC denote the discrete changes of corresponding tap positions. Parameter PBSS represents the control variable of a BSS, where a positive value means the BSS is charging and a negative value represents discharging. With the increasing integration of renewable energy sources, the highly uncertain renewables have brought rapid voltage fluctuations into distribution grids.






# TABLE II: Literature summary of voltage control.

| Reference     | Action Space                                                                        | Constraint                                                                  | Methodology                                         | Key Features                                                                                                                                                                                      |
| ------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| \[64] (2020)  | Set tap on/off position                                                             | Nodal voltage limits                                                        | CSAC, Lagrange multiplier method                    | Use an ordinal network structure to encode the natural ordering between actions of voltage regulating devices, and the inductive bias is introduced to accelerate the learning.                   |
| \[56] (2020)  | Reactive power adjustments of hybrid distribution transformer                       | Voltage state constraint                                                    | Model-based safe layer                              | Use sensitivity matrix to predict the bus voltages change, and compose an additional safety layer by a convex problem on top of the RL framework.                                                 |
| \[97] (2021)  | Load shedding ratio                                                                 | Voltage magnitude limits                                                    | Reward shaping with CBF                             | A well-designed barrier function is included in the reward function to guide the learning without model knowledge.                                                                                |
| \[82] (2021)  | Nodal reactive power                                                                | Voltage stability (Lipschitz constraint)                                    | Engineer the NN structure of RL controllers         | Optimize the set of Lipschitz bounds to enlarge the search space of RL controllers, and train locally at each bus with policy gradient algorithm.                                                 |
| \[98] (2021)  | Load shedding amount                                                                | Voltage recover criteria                                                    | Lagrange multiplier method                          | Formulate a Lagrangian function involving both the normal reward function and the safety function with a multiplier.                                                                              |
| \[99] (2022)  | Battery storage systems, capacitor banks, OLTCs, voltage regulators                 | Substation/BSS capacity constraints and nodal voltage/branch loading limits | CPO                                                 | 1. Design a stochastic policy to handle mixed discrete and continuous action space; 2. Employ the CPO algorithm to handle the operational constraints.                                            |
| \[100] (2022) | OLTCs, voltage regulators                                                           | Voltage magnitude limits                                                    | NN-based safe layer                                 | Propose a quadratic programming-based safe layer based on neural network architecture to enhance the safety.                                                                                      |
| \[101] (2023) | Load shedding of each bus                                                           | Voltage recovery criteria                                                   | Offline(batch) RL                                   | Use a surrogate model to generate roll-outs in the RL training stage, and add imitation learning to reduce the early unsafe exploration of the policy training.                                   |
| \[102] (2023) | Reactive power of mobile ESSs and PV smart inverters                                | State of charge limits, voltage magnitude limits                            | Model-based action replacement                      | Propose two safety modules with plug-and-play functionality to ensure a safe exploration, based on the system physical model.                                                                     |
| \[57] (2023)  | Reactive power injections of PV inverters                                           | Nodal voltage limits and power flow limits                                  | Model-based safe layer                              | A safety projection is added to analytically solve an action correction formulation per each state, which guides the exploratory actions in the direction of feasible policies.                   |
| \[103] (2023) | Voltage magnitude, local loads, active and reactive power outputs from PV inverters | Voltage magnitude limits                                                    | Lagrange method, graph convolutional networks (GCN) | 1. Use the GCN module to denoise graph signals and improve robustness against flawed input data; 2. Transform the original problem into an unconstrained min-max problem by Lagrange multipliers. |
| \[104] (2024) | Reactive power injections of PV inverters                                           | Voltage magnitude limits and voltage unbalance                              | Human intervention                                  | An online switch mechanism between action exploration and human intervention is designed to facilitate the learning process towards human actions.                                                |

Since the mixed discrete and continuous action space cannot be handled by traditional RL methods directly, most published safe RL techniques in voltage control only consider either continuous devices (e.g., inverter-based DGs) [57], [103] or discrete devices (e.g., OLTCs, VRs) [64], [100]. Nevertheless, the recent studies [99], [105] propose safe RL for a hybrid action space, taking into account both the safety of discrete and continuous actions simultaneously.

Efforts [56], [100]. Hence, the commonly used reward function can be generalized as:

rVol = −βCloss Σi∈NB Pi − (1 − β)Creg ΣA ∆a2,

where β is the weight factor; Closs and Creg are cost coefficients of network loss and device regulation; ∆a denotes the change of device state after executing actions. Note that,

2) Reward and cost functions: The reward design directly guides the achievement of the voltage control objective. Since a key objective of voltage control is to reduce network losses, the negative of the total network loss in the distribution network is the most commonly used reward [57], [99], [103]. On this basis, considering the device loss brought by frequent switching (especially for slow timescale devices), some researchers also introduce an extra penalty item to reduce control.

The key constraint for voltage control is to limit the voltage magnitudes within an acceptable range or stay at a nominal value. Based on the required upper and lower limits v, v





of nodal voltages, the cost function is typically defined as the voltage violation value. For example, in [56], [64], [100], [103], the cost function c is formulated in a generalized way as:

X

cVol = ∑i∈NB (vi − v + vi − v − v − v)

Moreover, some recent studies have begun to take more system operation constraints into account, such as the branch power flow and thermal current constraints [57], substation/BSS capacity constraints [99], and other necessary sophisticated constraints [102].

# 3) Safe RL techniques:

Both the safe layer-based and policy optimization-based methods have been applied for voltage control:

- Safe layer: As described in Table II, for the safe layer-based method, some works design model-based quadratic programming as the safe layer [56], [57], [102]. For instance, in [57], constraints of the safe layer require accurate knowledge of the branch power flow model, to solve safe action projection. In [56], the next voltage state is represented by a first-order linear approximation to obtain an explicit sensitivity matrix, which can analytically describe how much the bus voltages change after executing actions. Moreover, an alternative is to utilize NNs to approximate the unknown system model in the safe layer [82], [100], when there is no straightforward analytical solution to the quadratic problem with multiple inequality constraints. Two key challenges of the safe layer-based method in voltage control include 1) the next voltage state is hard to predict without models, which hinders the judgment of constraint safety; 2) a well-designed safe layer is difficult to generalize to another distribution network, especially when state/action spaces changes.
- Policy optimization: Compared with the safe layer, the CMDP-based policy optimization is model-free and more universal to various distribution networks, which only requires the collection of historical cost value. Typically, the cost function Jπ can be formulated as a penalty term in the objective C via a Lagrange multiplier [64], [98], [103], whose limitation is that manually tuning the Lagrange multiplier λ usually requires a tedious process of trial-and-error. To solve the CMDP in a completely self-adaptive way, the average KL-Divergence DKL(πt−1, πt) is introduced in TRPO-class algorithms to measure the searching area of policy, which releases the requirement on system models [99]. Then, an advantage function Aπ(s, a) = r(s, a, s′) + γV (s′) − V (s) is introduced to formulate the constraint bounds of cost Jπ, leading to a convex quadratic optimization that can be solved analytically with a guarantee of global optimum. Nonetheless, the limitation is the huge computation burden to obtain the analytical solution of the optimization, such as calculating the inversion of the FIM.

# 4) Discussion:

Some key issues of applying safe RL to voltage control are discussed below.

- Distributed Multi-device Collaboration: As the network scale increases, centralized voltage control requires a central controller leading to a heavy communication burden, which is vulnerable to single-point failures. Moreover, the controllable devices for voltage control may belong to different entities, involving data privacy and multi-device collaboration efficiency. Hence, the commonly used “centralized single-agent safe RL” is challenged by the urgent concerns of communication failure, privacy, and scalability. Reference [103] proposed a multi-agent safe RL to reduce the necessity for communication, based on decentralized partially observable MDPs. Reference [106] used federated learning to cope with data privacy in voltage control. However, handling both the scalability and privacy in safe RL should be further investigated, especially when a multi-agent collaboration is required for large-scale networks.
- Initial Data Collection: Voltage control involves the complex network and power flow, thus current safe RL techniques mainly design safe layers and policy optimization criteria through dynamic neural network approximation. This could still be problematic in real-world power systems because NN-based approximation is unreliable at the early stage, which takes effect when collecting enough data. Two possible remedies are 1) training the policy on digital twin simulators to collect initial data, and 2) applying transfer learning or sim-to-real techniques [107] to generate initial sample data.
- System Operation Constraint: Apart from the voltage magnitude limits, more system operation constraints should be taken into account in practice, such as power factor/branch line constraints and device capacity. Although reference [57] formulated power flow constraints into the optimization problem in the safe layer, it requires an accurate network operation model, which is impractical. Reference [99] has tried to consider all constraints in one cost function, while it is difficult to balance the trade-off between different constraints. One may use hierarchical RL to assign different constraints to different layers or sub-tasks to extend the safe RL satisfying multiple operation constraints.

# C. Energy management

Energy management is a broad research area in power systems that aims to maintain the power balance in an economical and reliable manner, which is significant for the large-scale integration of distributed energy resources and a decarbonized future. To this end, the concept of energy management systems (EMSs) is proposed to achieve real-time system control and optimization. However, in practice, stochastic user demands and weather-relied renewables supply cause significant uncertainties. Thus, among massive optimization approaches, model-free RL is paid more attention nowadays in EMSs because it can handle highly uncertain environments without prior model knowledge. Unfortunately, energy management problems usually involve multiple control devices with several local operation constraints and global power balance requirements. It is difficult for conventional RL to capture the physical constraints, destroying secure system operation. Therefore, as summarised in Table III, many recent works explore safe RL techniques for energy management. In the rest of this subsection, we classify the research about EMSs into three categories according to the control objects, and introduce the corresponding state, action, constraints, and reward function.




# 14

# TABLE III: Literature summary of energy management.

| Reference     | Problem                                   | Constraint                                                      | Methodology                        |
| ------------- | ----------------------------------------- | --------------------------------------------------------------- | ---------------------------------- |
| \[8] (2019)   | EV charging scheduling                    | Energy requirement                                              | CPO                                |
| \[108] (2019) | Building HVAC scheduling                  | Temperature comfort                                             | Safe layer, MPC                    |
| \[109] (2020) | Microgrid power management                | Voltage/current flow limits and other operational limits        | Model-based policy gradient        |
| \[110] (2021) | Microgrid energy management               | Voltage/line/ESS/power flow constraints                         | CPO                                |
| \[111] (2022) | Residential energy management             | Temperature comfort                                             | NN-based safe layer                |
| \[86] (2022)  | Energy hub                                | Electricity power limit and the heat balance constraint         | Safety-guided exploration          |
| \[112] (2022) | Optimal power flow                        | Generator/bus voltage/line flow limits                          | Primal-dual method                 |
| \[113] (2022) | Microgrid energy dispatch                 | Power flow constraint and generator power limits                | Offline RL                         |
| \[114] (2023) | Multi-energy management system            | Electrical and thermal power output constraints                 | Safe layer by action replacement   |
| \[40] (2023)  | Multi-energy microgrids                   | Constraints and limits of power and gas networks                | Physical-informed safety layer     |
| \[41] (2023)  | District cooling system power dispatch    | Service performance for power reduction and temperature comfort | Model-based safe layer             |
| \[55] (2023)  | Optimal power flow                        | Generation/line/voltage limits                                  | Knowledge-data-driven safety layer |
| \[115] (2023) | Microgrid energy management               | Power balance, line capacities, nodal voltage magnitudes        | IPO, CBF                           |
| \[59] (2023)  | Integrated electric-gas system            | Constraints for electricity and gas networks                    | CSAC, Lagrange method              |
| \[60] (2024)  | Demand management in distribution network | Carbon emission allowances                                      | Multi-agent CPO                    |
| \[66] (2024)  | Community integrated energy system        | Constraints for retail energy prices/integrated energy balance  | Primal-dual, Lagrange method       |

# Key Features

1. Use a DNN to learn the constrained optimal policy in an end-to-end manner; 2. Employ the CPO algorithm to ensure safety.
2. Limit the actions within a safe range and the maximum absolute change of actions according to prior knowledge.
3. Employ the gradient information of operational constraints to generate safe and feasible decisions.
4. Employ CPO algorithm to train an NN-based policy to achieve constraint safety.
5. Propose a prediction model-guided safe layer through an online prediction model to evaluate output actions.
6. Add a safety-guided network to avoid physical constraint violations without adding a penalty term to the reward.
7. Combine the primal-dual RL algorithm and power system models to approximate actor gradients by the Lagrangian.
8. Propose a two-stage learning framework: 1) a pre-training stage with imitation learning; 2) an online training stage with action clipping and expert demonstrations.
9. (1) Propose the safe layer and safe fallback policy to increase the policy’s initial utility; (2) Introduce self-improving hard constraints to increase the accuracy of constraints.
10. Learn a security assessment rule to form a safety layer and mathematically solve an action correction formulation.
11. Design a partial model-based safe layer based on a linear program to achieve safety-imposing projection.
12. Propose a model-based safety layer with prior knowledge and updated continuously according to the latest experiences.
13. Employ the IPO algorithm utilizing a logarithmic barrier function to govern the satisfaction of the safety constraints.
14. Add a safety network to update the constraint violation penalties to guide the policy in a safe direction.
15. Proposes a consensus multi-agent CPO approach to satisfy the carbon emission limit and preserving private information.
16. Employ a Lagrangian multiplier to penalize violation cost, using DNN to estimate the policy and action-value function.

respectively. Then the applied safe RL techniques are reviewed based on different frameworks.

1) State and action spaces: Considering different control objects have different device characteristics, we summarize the design of state, action, and constraints from the following two categories: integrated electricity-gas energy system, and grid-level power dispatch.

• Residential energy management: Following the price-based demand response programs (e.g., time-of-use), users are motivated to make optimal schedules of domestic appliances, to minimize electricity costs through residential EMSs. Generally, for end-users, some loads that are essential and cannot be scheduled are considered “non-shiftable loads”, e.g., lighting, television, microwave, refrigerator, etc. Some loads that can be scheduled at different periods but cannot be interrupted are considered “shifted and non-interruptible loads”, such as washing machine and dishwasher. Other flexible loads whose power can be regulated continuously are considered “controllable loads”, where the most common applications are heating, ventilation, and air conditioning (HVAC) and electric vehicles (EVs). The action space is defined for shifted and non-interruptible loads i ∈ NNI and controllable loads j ∈ NC, as






follows:

aRes = [(xNI, PC)i ∈ NNI, j ∈ NC]⊺, (21)

where xNIi is a binary decision variable for shifted and non-interruptible load i (i.e., 1/0 denotes “on/off”), and PjC is the power of controllable load j. Thus, the action space includes both discrete and continuous control variables. The state space shall reflect the operation status of domestic devices in residential EMSs, which is usually defined by a high dimensional vector:

sRes = [t, eout, xNI, oNI, PC, oC)i ∈ NNI, j ∈ NC]⊺, (22)

which is composed of the current time t, operating state/power of different devices xNIi and PCj, and other related states of environments/shifted and non-interruptible loads/controllable loads, i.e., eout, oNI, and oC. For instance, the outdoor temperature, users’ comfort preference, and electricity price can be contained in eout [116]; oNI can include the remaining time required for devices [111]; and oC to complete the task and the task deadline captures temperature deviations for HVAC loads and departure time for EVs [8].

Constraints in the residential energy management problem are mainly from human comfort, such as whether all task processes for non-interruptible loads can be finished, the thermal comfort in HVACs, and the charging target of EVs at the departure time. If we use CRes to denote the generalized constraints in residential EMSs, there are two types of constraints:

CRes = CTarget, If t = T, (23a)

βHSS, βTES ∈ [−1, 1] represent the charging/discharging power rate (positive/negative) of HSS and TES as a percentage of their power capacities. The designed action space is continuous because all the above devices can be controlled continuously. Further, the state space can be defined to capture device operating information as:

sInt = [t, EHSS, ETES, PWG, PPV, λInt, DInt]⊺, (25)

where EHSS, ETES are the measured state-of-charge of HSS and TES, reflecting the environment dynamics after the action; PWG and PPV are the maximal WG and PV generation power determined by stochastic weather conditions; λInt denotes the pre-offered grid prices for various energy types, such as electricity/gas/carbon prices; DInt represents the uncertain energy demands for various energy types, e.g., electricity and heat demands. If more uncertainties are considered in the integrated network, e.g., real-time price, it becomes more challenging to solve the designed MDP.

For the action space in integrated EMSs, decisions for each device are independent without correlation, which may lead to violations of energy balance constraints. Hence, apart from the regular upper/lower limits for single variables, e.g., import/export power capacity, the inner demand-supply balance of heat/electricity is also a significant constraint, which can be generalized as:

Hd = Hs, Pd = Ps, Gd = Gs, (26)

where Hd, Pd, Gd are heat, power, and gas demands; Hs, Ps, Gs denote corresponding heat, power, and gas supplies

CRes ≤ CRes ≤ CRes, ∀0 ≤ t ≤ T, (23b)

where the first constraint type is to check the completion of tasks through the target CTarget at the end of the day; and the second constraint type is to limit the upper/lower bounds by CRes and CRes during the whole management process. Because each managed domestic appliance has its corresponding comfort requirement, the number of constraint limitations in residential energy management problems depends on the controlled device number.

- Integrated electricity-gas energy system: The integrated EMSs can combine electricity, heat, cooling, natural gas, and hydrogen to achieve the efficient synergy of various carriers for meeting energy demands, based on conversion, distribution, and storage technologies. Generally, controllable devices in integrated EMSs are divided into three types: 1) RENs, e.g., wind generator (WG) and solar photovoltaic (PV); 2) storage systems, e.g., hydrogen storage system (HSS) and thermal energy storage (TES); 3) energy conversion devices, e.g., combined heat and power (CHP), electric heat pump (EHP), and gas boiler (GB). Hence, the action space can be designed as:

aInt = [αWG, αPV, αCHP, αEHP, αGB, βHSS, βTES]⊺, (24)

where αWG, αPV, αCHP, αEHP, αGB ∈ [0, 1] represent the magnitude of output heat power of WGs, PVs, CHPs, EHPs, and GBs, as a percentage of their maximum power limits. Variables in integrated EMSs. Note that the generalized formulation only shows the design principle, and specific constraint models should be specified based on the inner structure of integrated energy systems.

- Grid-level power dispatch: Due to the security and economy, the optimal power flow (OPF), is the fundamental tool underlying extensive scenarios of grid-level power dispatch, especially security-constrained OPF [55]. The key control objects in OPF are the power outputs of generators, thus the action space for grid-level power dispatch is generally defined as:

aopf = [Pgen, Qgenk ∈ NG]⊺, (27)

where Pgen and Qgen represent the commands to the active and reactive power outputs of k-th generator. Commonly, the on/off statuses of the generators are assumed to be predetermined and not changed during real-time dispatch. The on/off statuses can also be included as a binary decision in action space [117]. The state space is usually defined as:

sopf = [Pgen, Qgen, v, PD, QD, ˜pre, ˜pre⊺]

where PD and QD are the active and reactive net demand at bus i; ˜pre and Qre are the prediction of the next active and reactive system loads. Decision-making in OPF is highly related to the accuracy of future load forecasting. To improve accuracy, some works tend to increase the length of the forecasting time slots in the state, providing more historical





information [55]. Nevertheless, a large scale of the state space increases the training complexity, which requires a trade-off between information content and complexity.

The constraints considered in OPF problems mainly include the power flow equations, bus voltage/transmission line flow limits, and physical limits of controllable generators, which contain equality and inequality constraints. Hence, the constraint can be generalized as:

PG − PD = vi Xij (Gij cos θij + Bij sin θij), ∀i ∈ N,

QG − QD = vi Xij (Gij sin θij − Bij cos θij),

Pgen ≤ Pgen ≤ Pgen, Qgen ≤ Qgen ≤ Qgen, ∀k ∈ NG,

v ≤ vi ≤ v, ∀i ∈ NB,

where PG and QG are the injections of the active and reactive power at bus i; Gij and Bij are the conductance and susceptance of the transmission line between bus i and bus j; θij is the angle difference between bus i and bus j; Pgen/Pgen and Qgen/Qgen are the corresponding upper/lower bounds of the active and reactive power outputs of k-th generator.

2) Reward and cost functions: The control objective of OPF problems is to minimize the total generation costs of the power system, where the reward function for cost minimization can be generally written as:

rE = − Σt=0∞ Σk∈NG ak (Pgen)2 + bk Pgen + ck,

where ak, bk, and ck are the operation cost coefficients of generation k. Considering that Eq. (29) includes lots of operation constraints, the cost function is usually designed as the total penalization for violations, which can be generalized as:

cE = Σl [wl · ReLU(cl − cl)],

where l represents the index of the constraints defined in Eq. (29), i.e., Σl = 3NB + 2NG; wl denotes the penalty weight of each constraint; function ReLU(x) = max(0, x) is a linear rectification function for violation measurements; cl and cl represent the actual power flow state and required limit value. Taking constraint Eq. (29d) as an example, the cl and cl can be defined by cl = {vi, −vi} and cl = {v, −v}.

3) Safe RL techniques: Energy management covers multiple energy flows among different subjects. Based on different problem characteristics, we present the applied safe RL techniques in EMSs, to show the advantages of various techniques under different scenarios.

- Safe layer: The model-based safe layer can effectively cope with hard constraints in EMSs. For instance, reference [114] proposed OptLayerPolicy to increase the accuracy of the cost function, which can keep a high sample efficiency at the initial stage to solve the closest feasible action in safe layers. Besides, based on a specific and well-designed correction rule, authors in [41] also successfully address the hard constraints.

4) Discussion:

- Insufficient scenario occurrence: Some energy management problems for power systems, such as service restoration, probably have insufficient online training due to the infrequent occurrence (e.g., low outage rates). Hence, the requirement for a large amount of training scenarios is one of the major impediments to applying safe RL, especially for safety-critical but infrequent scenarios. Pre-training can be considered a compensatory for safe RL, as an alternative to the agent’s early training process. To this end, reference [113] proposed a two-stage safe RL framework by introducing the pre-training stage before the online training stage, where the initialized agent can gain a jump-start performance through expert imitation. More online approaches for handling scenario insufficiency issues remain to be explored.
- Multi-agent safe control for multi-EMSs: Energy management problems usually contain multiple devices or distributed.




microgrids for cooperation control, involving global and local constraints. Most aforementioned studies consider all the constraints in one cost function and address them through one RL agent. Reference [109] proposed a multi-agent consensus-based training algorithm for distributed microgrids, which designs multiple autonomous controllers for joint safe control through local communication. To further decouple intractable power-carbon flow constraints for low-carbon EMSs, reference [60] extended the traditional CPO algorithm to a consensus multi-agent CPO, to achieve the safe control for low-carbon demand management.

• Spatial-temporal perception: Accurate perception of the spatial-temporal operating characteristics is significant for EMSs, such as the spatial distribution of power flows and the temporal evolution of the renewables. For the energy management in distributed microgrids, reference [115] proposed an interior-point policy optimization (IPO) algorithm to utilize a logarithmic CBF to ensure constraint safety, by introducing edge-conditioned convolutional and long short-term memory networks. These two feature extraction networks can effectively find out the spatial and temporal dependencies, for better state prediction accuracy.

# Other applications

In addition to the above three critical applications, the safe RL concept is also applied to other problems in power systems, including dynamic distribution network reconfiguration [118], real-time congestion management [119], electricity market [43], setpoint optimization for transmission systems [120], organic Rankine cycle system control [121], resilient proactive scheduling [122], transmission overload relief [123], emergency recovery under line outages [124], etc.

# IV. CHALLENGES AND PERSPECTIVES

Safe RL, as one type of RL variant, naturally faces all challenges in traditional RL, such as data availability and scalability summarised in [5]. In this section, we do not repeat the common challenges in traditional RL, and only present the unique challenges of safe RL applications in power systems, including four aspects: (1) convergence and optimality; (2) training efficiency; (3) universality; and (4) real-world environment deployment. Three future directions are then discussed.

# A. Convergence and optimality

The RL applications aim to find an optimal control policy after training, thereby ensuring the successful convergence of training and policy optimality are two key criteria to evaluate the reliability and effectiveness of the algorithm. For the training convergence in safe RL, introducing safe layers or changing policy update rules both limit the RL agent’s exploration space, and even probably bring wrong feedback, which results in the traditional RL convergence theory no longer being applicable. For instance, the RL agent’s original action is usually corrected in safe layers, which rely on specific design principles for different tasks. Hence, it is impossible to prove that a random safe layer design can bring a successful convergence. This is because a too strong intervention in a safe layer can easily destroy the convergence of training, if the reward feedback is not properly corrected by the safe layer. Currently, some researchers have presented approximate proof of convergence or the theoretical guarantee under certain conditions, e.g., Lyapunov-based and CBF-based safe layer methods [45], [80]. However, these proofs are usually proposed based on certain assumptions, such as an approximation of the environment model, a limitation of the policy space, or a simplification of the optimization process. Therefore, we believe that rigorous global convergence proof remains a challenge for safe RL, and future research needs to explore more general theoretical frameworks to provide stronger guarantees of training convergence.

For the policy optimality in safe RL, because RL training processes update policy mainly relying on the feedback of random explorations, there exists a conflict and trade-off between the conservative policy for safety and aggressive explorations for improvement. This trade-off may result in a poor reward expectation in some scenarios, obtaining a sub-optimal or even bad policy. Specifically, a safe but limited policy-searching area makes the agent unable to explore the entire state/action spaces, which probably causes the algorithm to converge to a locally optimal solution rather than a global one. This phenomenon is similar to the fundamental dilemma between exploration and exploitation in RL. Considering the research on safe RL algorithms is still in its early stage, few papers are handling this challenge effectively through algorithm design or optimization strategy.

Besides, the actual power system control involves not only single object but requires cooperation between several distributed areas. Purely single-agent safe RL may be too hard to handle all global and local constraints and suffer from convergence issues. Currently, there are very few solutions that offer effective learning algorithms for safe multi-agent control problems. Recently, reference [125] proposed the first multi-agent trust region method that successfully attains theoretical guarantees of both reward improvement and satisfaction of safety constraints. Then, as the first safety-aware model-free algorithms, reference [126] extend CPO and Lagrange methods to the multi-agent area with theoretical analysis. Despite limited theoretical work on this subject, investigating multi-agent safe RL for multi-device control on cooperative tasks in power systems is envisioned to be an important future direction.

# B. Training efficiency

The training efficiency for safe RL mainly includes two types: sample efficiency and computation efficiency, where sample efficiency refers to the utilization of data samples (i.e., high sample efficiency can obtain a better policy through fewer data samples), and computation efficiency indicates the utilization of computing resources (e.g., CPU, GPU, and memory). Compared with conventional RL, sample efficiency in safe RL is decreased significantly when facing safety-critical constraints, because variance among collected samples.






becomes smaller when explorations are limited within a safe area. That is, a smaller variance provides less knowledge for the RL agent and slows down its policy update iteration. Moreover, in practice, power systems are considered reward-sparse environments for safety-critical problems, because unsafe data samples are usually not adequate or even limited, such as voltage/frequency violations. The reward-sparse issue also influences the sample efficiency. Currently, authors in [127] proposed a sample-efficient safe RL framework to achieve efficient learning with limited samples through three techniques: 1) avoiding behaving overly conservatively; 2) encouraging safe exploration; 3) treating RL agents as expert demonstrations.

In addition, the introduced safe module increases the computation complexity, which further decreases the computation efficiency of safe RL algorithms. For instance, when using MPC-based methods, one needs to solve an extra linear or quadratic program at every time step t [128]. When using CBF-based methods, one needs to store lots of previous RL policy networks and solve multiple separate quadratic programs in sequence to evaluate each CBF controller [45]. When requiring real-time operations in power systems, computation efficiency becomes an issue for the safe RL application, especially in highly non-linear and complex environments.

# C. Universality

The power system is a large-scale dynamic system whose system characteristics may change according to dynamic demands, such as the changing number of generators/transformers, and the variational topology brought by line faults/maintenance. In particular, the increasing distributed energy resources may more frequently change the grid topology for distribution networks. Hence, the universality of a well-trained agent is necessary for power grids to cope with various operation scenarios. However, most of the current safe RL algorithms have poor universality, leading to low reusability of controllers. Once the power system model changes, it may be necessary to retrain the agent, resulting in expensive computation and training costs. The key reasons for poor universality in different safe RL techniques are not the same.

1. Universality for safe layers: For safe layer-based techniques, all constraints are designed based on an assumed accurate or approximated model, which will directly influence the solved optimization problem results in safe layers at each time step. When the system dynamic changes, the constraint formulation is necessary to be re-derived to fit the new system for safety. Thus, the intervention method for the agent by safe layers is changed, and the agent should be re-trained based on the designed new safe layers.
2. Universality for changing optimization criteria: Although this category of safe RL techniques (introduced in Section II-C) is a totally model-free method without prior model knowledge, its training relies on the collected historical costs for constraint violations. When the system dynamic changes, the relationship between the real-time cost and the system operation state is also changed correspondingly, which is different from the original data distribution. Thus, the original safe policy is probably not safe for the new system, requiring re-training based on newly collected data.

Considering that RL is fundamental and vibrant research that garners significant attention, innovations in RL are emerging at a rapid pace, e.g., inverse RL [129], meta RL [130], hierarchical RL [131], attention-based RL [132], etc. To enhance the algorithm universality, one can fully leverage existing research achievements in reinforcement learning. For instance, hierarchical RL can decompose the control task into multiple subtasks to design dynamic safety constraints and exploration methods, reducing the probability of the policy falling into local optima. Meta RL can be used to collect safety information from multiple different environments, thereby enhancing safety performance in downstream task training without sufficient expert experience.

# D. Real-world deployment

For most published works, case studies are designed in simulated environments based only on small-scale grids with a few control objects. To the best of our knowledge, safe RL has not yet been implemented in any real-world power system, even though it claims to be safer than traditional RL algorithms. One of the crucial limitations in safe RL deployment is the initial policy quality, since an initialized policy without training cannot ensure performance even though it may be safe. Since the control in grids usually involves multi-process coupling and multi-department cooperation, it is difficult to directly deploy a random and poor initial policy in the real world for such a long training time. One potential direction to improve the reliability of initial policies is to obtain a pre-trained policy in simulators first to find satisfactory initial policies [11]. However, because of the lack of theoretical assurance, this idea also faces safety issues caused by the gap between simulators and real-world large-scale systems.

Two state-of-the-art techniques can be combined with safe RL frameworks to further address the reality gap: sim2real [133] and digital twin modeling [134]. Specifically, sim2real is a method for transferring safe RL algorithms trained in simulated environments to real-world applications, and digital twin modeling is a technique for creating a virtual copy of a real-world physical system in a digital space. For instance, simulations in reference [135] are implemented in the digital twin environment to present effective energy management for household demand response, employing safe RL and fuzzy reasoning. The proposed digital twin model presents a real-time consumer interface, including smart devices, energy price signals, smart meters, solar PVs, batteries, electric vehicles, and grid supply.

In addition, exploring the effective integration of online safe RL techniques with the deployment of offline RL presents a promising research direction. This idea is conducted solely on pre-collected offline trajectory datasets, without the need for real-time interaction with the environment. While this approach circumvents unsafe exploration phases, ensuring safety during model training, there still exists a gap between the offline training data and real-world datasets. To date, only the constraints penalized Q-learning (CPQ) method [136]






has proposed using additional cost critic (such as reward critic) to learn constraint values, which effectively bridges the distribution gap between the offline and real-world datasets.

However, CPQ still has a theoretical error bound under mild assumptions, as an offline method.

# V. CONCLUSIONS

This paper provides a comprehensive review of safe RL techniques and applications in power systems for the first time. We summarize two categories of state-of-the-art safe RL techniques, which are based on safe layers and policy optimization criteria. Then, three key applications in power systems are summarized through the detailed design of state, action, reward, cost, and applied safe RL methods. Finally, several key challenges and future directions are discussed.

In summary, although safe RL has been paid more attention to for better application in power systems, there is still quite a long distance from real-world deployment. The most important issue is the lack of theoretical proof for safe RL applications, which cannot ensure the safety and optimality of common scenarios. In fact, as an online training-based control method, explorations cannot be completely avoided in safe RL techniques, which makes it hard to deploy individually for safety-critical scenarios. Considering the advantage of model-free characteristics, combining safe RL with model-based traditional controllers is probably more promising and practical.

# REFERENCES

1. P. Lopion, P. Markewitz, M. Robinius, and D. Stolten, “A review of current challenges and trends in energy systems modeling,” Renewable and sustainable energy reviews, vol. 96, pp. 156–166, 2018.
2. S. Borlase, Smart grids: Advanced technologies and solutions. CRC press, 2017.
3. D. Cao, W. Hu, J. Zhao, G. Zhang, B. Zhang, Z. Liu, Z. Chen, and F. Blaabjerg, “Reinforcement learning and its applications in modern power and energy systems: A review,” Journal of modern power systems and clean energy, vol. 8, no. 6, pp. 1029–1042, 2020.
4. Y. Li, C. Yu, M. Shahidehpour, T. Yang, Z. Zeng, and T. Chai, “Deep reinforcement learning for smart grid operations: algorithms, applications, and prospects,” Proceedings of the IEEE, 2023.
5. X. Chen, G. Qu, Y. Tang, S. Low, and N. Li, “Reinforcement learning for selective key applications in power systems: Recent advances and future challenges,” IEEE Trans. Smart Grid, vol. 13, no. 4, pp. 2935–2958, 2022.
6. M. Pecka and T. Svoboda, “Safe exploration techniques for reinforcement learning–an overview,” in Modelling and Simulation for Autonomous Systems: First International Workshop, MESAS 2014, Rome, Italy, May 5-6, 2014, Revised Selected Papers 1, pp. 357–375, Springer, 2014.
7. J. Garcıa and F. Fern´andez, “A comprehensive survey on safe reinforcement learning,” Journal of Machine Learning Research, vol. 16, no. 1, pp. 1437–1480, 2015.
8. H. Li, Z. Wan, and H. He, “Constrained ev charging scheduling based on safe deep reinforcement learning,” IEEE Trans. Smart Grid, vol. 11, no. 3, pp. 2427–2439, 2019.
9. Y. Wei, M. Tian, X. Huang, and Z. Ding, “Incorporating constraints in reinforcement learning assisted energy system decision making: A selected review,” 2022 IEEE/IAS Industrial and Commercial Power System Asia (I&#x26;CPS Asia), pp. 671–675, 2022.
10. S. Levine, A. Kumar, G. Tucker, and J. Fu, “Offline reinforcement learning: Tutorial, review, and perspectives on open problems,” arXiv preprint arXiv:2005.01643, 2020.
11. A. Lesage-Landry and D. S. Callaway, “Batch reinforcement learning for network-safe demand response in unknown electric grids,” Electric Power Systems Research, vol. 212, p. 108375, 2022.
12. R. F. Prudencio, M. R. Maximo, and E. L. Colombini, “A survey on offline reinforcement learning: Taxonomy, review, and open problems,” IEEE Trans. Neural Networks and Learning Systems, 2023.
13. C. Szepesvári, Algorithms for reinforcement learning. Springer Nature, 2022.
14. K. Srinivasan, B. Eysenbach, S. Ha, J. Tan, and C. Finn, “Learning to be safe: Deep rl with a safety critic,” arXiv preprint arXiv:2010.14603, 2020.
15. T. Yang, H. Tang, C. Bai, J. Liu, J. Hao, Z. Meng, P. Liu, and Z. Wang, “Exploration in deep reinforcement learning: a comprehensive survey,” arXiv preprint arXiv:2109.06668, 2021.
16. I. Grondman, L. Busoniu, G. A. Lopes, and R. Babuska, “A survey of actor-critic reinforcement learning: Standard and natural policy gradients,” IEEE Trans. Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 42, no. 6, pp. 1291–1307, 2012.
17. D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Riedmiller, “Deterministic policy gradient algorithms,” in International conference on machine learning, pp. 387–395, Pmlr, 2014.
18. J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, “Trust region policy optimization,” in International conference on machine learning, pp. 1889–1897, PMLR, 2015.
19. H. Krasowski, J. Thumm, M. Müller, L. Sch¨afer, X. Wang, and M. Althoff, “Provably safe reinforcement learning: A theoretical and experimental comparison,” arXiv preprint arXiv:2205.06750, 2022.
20. S. Gu, A. Kshirsagar, Y. Du, G. Chen, J. Peters, and A. Knoll, “A human-centered safe robot reinforcement learning framework with interactive behaviors,” Frontiers in Neurorobotics, vol. 17, 2023.
21. M. Alshiekh, R. Bloem, R. Ehlers, B. K¨u, U. Topcu, “Safe reinforcement learning via shielding,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, 2018.
22. A. Platzer, “Safe reinforcement learning via formal methods: Toward safe control through proof and learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, 2018.
23. J. Thumm and M. Althoff, “Provably safe deep reinforcement learning for robotic manipulation in human environments,” in 2022 International Conference on Robotics and Automation (ICRA), pp. 6344–6350, 2022.
24. W. Saunders, G. Sastry, A. Stuhlmueller, and O. Evans, “Trial without error: Towards safe reinforcement learning via human intervention,” arXiv preprint arXiv:1707.05173, 2017.
25. B. Prakash, M. Khatwani, N. Waytowich, and T. Mohsenin, “Improving safety in reinforcement learning using model-based architectures and human intervention,” arXiv preprint arXiv:1903.09328, 2019.
26. H. Sun, Z. Xu, M. Fang, Z. Peng, J. Guo, B. Dai, and B. Zhou, “Safe exploration by solving early terminated mdp,” arXiv preprint arXiv:2107.04200, 2021.
27. B. Eysenbach, S. Gu, J. Ibarz, and S. Levine, “Leave no trace: Learning to reset for safe and autonomous reinforcement learning,” arXiv preprint arXiv:1711.06782, 2017.
28. S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor, and P. Stone, “Curriculum learning for reinforcement learning domains: A framework and survey,” The Journal of Machine Learning Research, vol. 21, no. 1, pp. 7382–7431, 2020.
29. X. Wang, Y. Chen, and W. Zhu, “A survey on curriculum learning,” IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 44, no. 9, pp. 4555–4576, 2021.
30. M. Turchetta, A. Kolobov, S. Shah, A. Krause, and A. Agarwal, “Safe reinforcement learning via curriculum induction,” Advances in Neural Information Processing Systems, vol. 33, pp. 12151–12162, 2020.
31. Z. Peng, Q. Li, C. Liu, and B. Zhou, “Safe driving via expert guided policy optimization,” in Conference on Robot Learning, pp. 1554–1563, PMLR, 2022.
32. Q. Li, Z. Peng, and B. Zhou, “Efficient learning of safe driving policy via human-ai copilot optimization,” arXiv preprint arXiv:2202.10341, 2022.
33. O. Bastani, S. Li, and A. Xu, “Safe reinforcement learning via statistical model predictive shielding.,” in Robotics: Science and Systems, pp. 1–13, 2021.
34. W. Zhang, O. Bastani, and V. Kumar, “Mamps: Safe multi-agent reinforcement learning via model predictive shielding,” arXiv preprint arXiv:1910.12639, 2019.
35. S. Li and O. Bastani, “Robust model predictive shielding for safe reinforcement learning with stochastic dynamics,” in 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 7166–7172, IEEE, 2020.
36. Y. Wang, S. S. Zhan, R. Jiao, Z. Wang, W. Jin, Z. Yang, Z. Wang, C. Huang, and Q. Zhu, “Enforcing hard constraints with soft barriers: Safe reinforcement learning in unknown stochastic environments,” in






International Conference on Machine Learning, pp. 36593–36604, PMLR, 2023.

1. K. P. Wabersich and M. N. Zeilinger, “Linear model predictive safety certification for learning-based control,” in 2018 IEEE Conference on Decision and Control (CDC), pp. 7130–7135, IEEE, 2018.
2. N. Jansen, B. Königshofer, S. Junges, A. Serban, and R. Bloem, “Safe reinforcement learning using probabilistic shields,” in 31st International Conference on Concurrency Theory (CONCUR 2020), Schloss-Dagstuhl-Leibniz Zentrum für Informatik, 2020.
3. S. Carr, N. Jansen, S. Junges, and U. Topcu, “Safe reinforcement learning via shielding under partial observability,” in Proceedings of AAAI Conference on Artificial Intelligence, vol. 37, pp. 14748–14756, 2023.
4. Y. Wang, D. Qiu, M. Sun, G. Strbac, and Z. Gao, “Secure energy management of multi-energy microgrid: A physical-informed safe reinforcement learning approach,” Appl. Energy, vol. 335, p. 120759, 2023.
5. P. Yu, H. Zhang, Y. Song, H. Hui, and G. Chen, “District cooling system control for providing operating reserve based on safe deep reinforcement learning,” IEEE Trans. Power Systems, 2023.
6. T. Zhao, J. Wang, and M. Yue, “A barrier-certificated reinforcement learning approach for enhancing power system transient stability,” IEEE Trans. Power Systems, 2023.
7. P. Rokhforoz, M. Montazeri, and O. Fink, “Safe multi-agent deep reinforcement learning for joint bidding and maintenance scheduling of generation units,” Reliability Engineering &#x26; System Safety, vol. 232, p. 109081, 2023.
8. Y. Yang, K. G. Vamvoudakis, H. Modares, Y. Yin, and D. C. Wunsch, “Safe intermittent reinforcement learning with static and dynamic event generators,” IEEE Trans. Neural Networks and Learning Systems, vol. 31, no. 12, pp. 5441–5455, 2020.
9. R. Cheng, G. Orosz, R. M. Murray, and J. W. Burdick, “End-to-end safe reinforcement learning through barrier functions for safety-critical continuous control tasks,” in Proceedings of the AAAI conference on artificial intelligence, vol. 33, pp. 3387–3395, 2019.
10. Y. Emam, P. Glotfelter, Z. Kira, and M. Egerstedt, “Safe model-based reinforcement learning using robust control barrier functions,” arXiv preprint arXiv:2110.05415, 2021.
11. Z. Marvi and B. Kiumarsi, “Safe reinforcement learning: A control barrier function optimization approach,” International Journal of Robust and Nonlinear Control, vol. 31, no. 6, pp. 1923–1940, 2021.
12. L. Wang, E. A. Theodorou, and M. Egerstedt, “Safe learning of quadrotor dynamics using barrier certificates,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 2460–2465, IEEE, 2018.
13. L. Wang, D. Han, and M. Egerstedt, “Permissive barrier certificates for safe stabilization using sum-of-squares,” in 2018 Annual American Control Conference (ACC), pp. 585–590, IEEE, 2018.
14. M. Ohnishi, L. Wang, G. Notomista, and M. Egerstedt, “Barrier-certified adaptive reinforcement learning with applications to brushbot navigation,” IEEE Trans. robotics, vol. 35, no. 5, pp. 1186–1205, 2019.
15. Y. Yang, Y. Yin, W. He, K. G. Vamvoudakis, H. Modares, and D. C. Wunsch, “Safety-aware reinforcement learning framework with an actor-critic-barrier structure,” in 2019 American Control Conference (ACC), pp. 2352–2358, IEEE, 2019.
16. L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger, “Learning-based model predictive control: Toward safe learning in control,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 3, pp. 269–296, 2020.
17. K. P. Wabersich, L. Hewing, A. Carron, and M. N. Zeilinger, “Probabilistic model predictive safety certification for learning-based control,” IEEE Trans. Automatic Control, vol. 67, no. 1, pp. 176–188, 2021.
18. K. P. Wabersich and M. N. Zeilinger, “A predictive safety filter for learning-based control of constrained nonlinear dynamical systems,” Automatica, vol. 129, p. 109597, 2021.
19. Z. Yi, X. Wang, C. Yang, C. Yang, M. Niu, and W. Yin, “Real-time sequential security-constrained optimal power flow: A hybrid knowledge-data-driven reinforcement learning approach,” IEEE Trans. Power Systems, 2023.
20. P. Kou, D. Liang, C. Wang, Z. Wu, and L. Gao, “Safe deep reinforcement learning-based constrained optimal control scheme for active distribution networks,” Appl. Energy, vol. 264, p. 114772, 2020.
21. G. Guo, M. Zhang, Y. Gong, and Q. Xu, “Safe multi-agent deep reinforcement learning for real-time decentralized control of inverter based renewable energy resources considering communication delay,” Appl. Energy, vol. 349, p. 121648, 2023.
22. E. Altman, Constrained Markov decision processes. Routledge, 2021.
23. A. B. Jeddi, N. L. Dehghani, and A. Shafieezadeh, “Lyapunov-based uncertainty-aware safe reinforcement learning,” arXiv preprint arXiv:2107.13944, 2021.
24. W. Cui, J. Li, and B. Zhang, “Decentralized safe reinforcement learning for voltage control,” arXiv preprint arXiv:2110.01126, 2021.
25. Z. Yuan, C. Zhao, and J. Cortés, “Reinforcement learning for distributed transient frequency control with stability and safety guarantees,” Systems &#x26; Control Letters, vol. 185, p. 105753, 2024.






1. W. Cui, Y. Jiang, and B. Zhang, “Reinforcement learning for optimal primary frequency control: A lyapunov approach,” IEEE Trans. Power Systems, vol. 38, no. 2, pp. 1676–1688, 2022.
2. J. Fan and W. Li, “Safety-guided deep reinforcement learning via online gaussian process estimation,” arXiv preprint arXiv:1903.02526, 2019.
3. D. Qiu, Z. Dong, X. Zhang, Y. Wang, and G. Strbac, “Safe reinforcement learning for real-time automatic control in a smart energy-hub,” Appl. Energy, vol. 309, p. 118403, 2022.
4. H. Bevrani and T. Hiyama, Intelligent automatic generation control. CRC press New York, 2017.
5. Y. Xia, Y. Xu, Y. Wang, S. Mondal, S. Dasgupta, A. K. Gupta, and G. M. Gupta, “A safe policy learning-based method for decentralized and economic frequency control in isolated networked-microgrid systems,” IEEE Trans. Sustain. Energy, vol. 13, no. 4, pp. 1982–1993, 2022.
6. X. Wan, M. Sun, B. Chen, Z. Chu, and F. Teng, “Adapsafe: Adaptive and safe-certified deep reinforcement learning-based frequency control for carbon-neutral power systems,” AAAI, 2023.
7. W. Cui and B. Zhang, “Lyapunov-regularized reinforcement learning for power system transient stability,” IEEE Control Systems Letters, vol. 6, pp. 974–979, 2021.
8. P. Yu, H. Zhang, and Y. Song, “District cooling system control for providing regulation services based on safe reinforcement learning with barrier functions,” Appl. Energy, vol. 347, p. 121396, 2023.
9. Z. Yan and Y. Xu, “A multi-agent deep reinforcement learning method for cooperative load frequency control of a multi-area power system,” IEEE Trans. Power Systems, vol. 35, no. 6, pp. 4599–4608, 2020.
10. J. Li and T. Yu, “Deep reinforcement learning based multi-objective integrated automatic generation control for multiple continuous power disturbances,” IEEE Access, vol. 8, pp. 156839–156850, 2020.
11. S. Rozada, D. Apostolopoulou, and E. Alonso, “Load frequency control: A deep multi-agent reinforcement learning approach,” in 2020 IEEE Power &#x26; Energy Society General Meeting (PESGM), pp. 1–5, IEEE, 2020.
12. Y. Gao, W. Wang, and N. Yu, “Consensus multi-agent reinforcement learning for volt-var control in power distribution networks,” IEEE Trans. Smart Grid, vol. 12, no. 4, pp. 3594–3604, 2021.
13. S. Wang, J. Duan, D. Shi, C. Xu, H. Li, R. Diao, and Z. Wang, “A data-driven multi-agent autonomous voltage control framework using deep reinforcement learning,” IEEE Trans. Power Systems, vol. 35, no. 6, pp. 4644–4654, 2020.
14. T. L. Vu, S. Mukherjee, R. Huang, and Q. Huang, “Barrier function-based safe reinforcement learning for emergency control of power systems,” in 2021 60th IEEE Conference on Decision and Control (CDC), pp. 3652–3657, IEEE, 2021.
15. T. L. Vu, S. Mukherjee, T. Yin, R. Huang, J. Tan, and Q. Huang, “Safe reinforcement learning for emergency load shedding of power systems,” in 2021 IEEE Power &#x26; Energy Society General Meeting (PESGM), pp. 1–5, IEEE, 2021.
16. H. Li and H. He, “Learning to operate distribution networks with safe deep reinforcement learning,” IEEE Trans. Smart Grid, vol. 13, no. 3, pp. 1860–1872, 2022.
17. Y. Gao and N. Yu, “Model-augmented safe reinforcement learning for volt-var control in power distribution networks,” Appl. Energy, vol. 313, p. 118762, 2022.
18. R. R. Hossain, T. Yin, Y. Du, R. Huang, J. Tan, W. Yu, Y. Liu, and Q. Huang, “Efficient learning of power grid voltage control strategies via model-based deep reinforcement learning,” Machine Learning, pp. 1–26, 2023.
19. S. Jeon, H. T. Nguyen, and D.-H. Choi, “Safety-integrated online deep reinforcement learning for mobile energy storage system scheduling and volt/var control in power distribution networks,” IEEE Access, 2023.
20. R. Yan, Q. Xing, and Y. Xu, “Multi agent safe graph reinforcement learning for pv inverter s based real-time decentralized volt/var control in zoned distribution networks,” IEEE Trans. Smart Grid, 2023.
21. X. Sun, Z. Xu, J. Qiu, H. Liu, H. Wu, and Y. Tao, “Optimal volt/var control for unbalanced distribution networks with human-in-the-loop deep reinforcement learning,” IEEE Trans. Smart Grid, 2023.
22. H. Liu and W. Wu, “Bi-level off-policy reinforcement learning for volt/var control involving continuous and discrete devices,” arXiv preprint arXiv:2104.05902, 2021.
23. H. Liu and W. Wu, “Federated reinforcement learning for decentralized voltage control in distribution networks,” IEEE Trans. Smart Grid, vol. 13, no. 5, pp. 3840–3843, 2022.
24. D. M. Bossens, “Robust lagrangian and adversarial policy gradient for robust constrained markov decision processes,” arXiv preprint arXiv:2308.11267, 2023.
25. C. Zhang, S. R. Kuppannagari, R. Kannan, and V. K. Prasanna, “Building hvac scheduling using reinforcement learning via neural network based model approximation,” in Proceedings of the 6th ACM international conference on systems for energy-efficient buildings, cities, and transportation, pp. 287–296, 2019.
26. Q. Zhang, K. Dehghanpour, Z. Wang, F. Qiu, and D. Zhao, “Multi-agent safe policy learning for power management of networked microgrids,” IEEE Trans. Smart Grid, vol. 12, no. 2, pp. 1048–1062, 2020.
27. H. Li, Z. Wang, L. Li, and H. He, “Online microgrid energy management based on safe deep reinforcement learning,” in 2021 IEEE Symposium Series on Computational Intelligence (SSCI), pp. 1–8, IEEE, 2021.
28. C. Huang, H. Zhang, L. Wang, X. Luo, and Y. Song, “Mixed deep reinforcement learning considering discrete-continuous hybrid action space for smart home energy management,” Journal of Modern Power Systems and Clean Energy, vol. 10, no. 3, pp. 743–754, 2022.
29. Z. Yan and Y. Xu, “A hybrid data-driven method for fast solution of security-constrained optimal power flow,” IEEE Trans. Power Systems, vol. 37, no. 6, pp. 4365–4374, 2022.
30. Y. Du and D. Wu, “Deep reinforcement learning from demonstrations to assist service restoration in islanded microgrids,” IEEE Trans. Sustain. Energy, vol. 13, no. 2, pp. 1062–1072, 2022.
31. G. Ceusters, M. A. Putratama, R. Franke, A. Nowé, and M. Messagie, “An adaptive safety layer with hard constraints for safe reinforcement learning in multi-energy management systems,” Sustainable Energy, Grids and Networks, vol. 36, p. 101202, 2023.
32. Y. Ye, H. Wang, P. Chen, Y. Tang, and G. Strbac, “Safe deep reinforcement learning for microgrid energy management in distribution networks with leveraged spatial-temporal perception,” IEEE Trans. Smart Grid, 2023.
33. H. Li, Z. Wan, and H. He, “Real-time residential demand response,” IEEE Trans. Smart Grid, vol. 11, no. 5, pp. 4144–4154, 2020.
34. T.-H. Fan and Y. Wang, “Soft actor-critic with integer actions,” in 2022 American Control Conference (ACC), pp. 2611–2616, IEEE, 2022.
35. Y. Gao, W. Wang, J. Shi, and N. Yu, “Batch-constrained reinforcement learning for dynamic distribution network reconfiguration,” IEEE Trans. Smart Grid, vol. 11, no. 6, pp. 5357–5369, 2020.
36. H. Yang, Y. Xu, and Q. Guo, “Dynamic incentive pricing on charging stations for real-time congestion management in distribution network: An adaptive model-based safe deep reinforcement learning method,” IEEE Trans. Sustain. Energy, 2023.
37. M. Tarle, M. Larsson, G. Ingeström, L. Nordström, and M. Björkman, “Safe reinforcement learning for mitigation of model setpoint control,” in 2023 International Conference on Smart Energy Systems and Technologies (SEST), pp. 1–6, IEEE, 2023.
38. L. Zhang, R. Lin, L. Xie, W. Dai, and H. Su, “Event-triggered optimal control for organic rankine cycle systems via safe reinforcement learning,” IEEE Trans. Neural Networks and Learning Systems, 2022.
39. Z. Liang, C. Huang, W. Su, N. Duan, V. Donde, B. Wang, and X. Zhao, “Safe reinforcement learning-based resilient proactive scheduling for a commercial building considering correlated demand response,” IEEE Open Access Journal of Power and Energy, vol. 8, pp. 85–96, 2021.
40. H. Cui, Y. Ye, J. Hu, Y. Tang, Z. Lin, and G. Strbac, “Online preventive control for transmission overload relief using safe reinforcement learning with enhanced spatial-temporal awareness,” IEEE Trans. Power Systems, 2023.
41. X. Weiss, S. Mohammadi, P. Khanna, M. R. Hesamzadeh, and L. Nordström, “Safe deep reinforcement learning for power system operation under scheduled unavailability,” in 2023 IEEE Power &#x26; Energy Society General Meeting (PESGM), pp. 1–5, IEEE, 2023.
42. J. G. Kuba, R. Chen, M. Wen, Y. Wen, F. Sun, J. Wang, and Y. Yang, “Trust region policy optimisation in multi-agent reinforcement learning,” arXiv preprint arXiv:2109.11251, 2021.
43. S. Gu, J. G. Kuba, Y. Chen, Y. Du, L. Yang, A. Knoll, and Y. Yang, “Safe multi-agent reinforcement learning for multi-robot control,” Artificial Intelligence, vol. 319, p. 103905, 2023.
44. H. Chen and C. Liu, “Safe and sample-efficient reinforcement learning for clustered dynamic environments,” IEEE Control Systems Letters, vol. 6, pp. 1928–1933, 2023.
45. T. Koller, F. Berkenkamp, M. Turchetta, and A. Krause, “Learning-based model predictive control for safe exploration,” in 2018 IEEE conference on decision and control (CDC), pp. 6059–6066, IEEE, 2018.






# References

1. Q. Tang, H. Guo, and Q. Chen, “Multi-market bidding behavior analysis of energy storage system based on inverse reinforcement learning,” IEEE Trans. Power Systems, vol. 37, no. 6, pp. 4819–4831, 2022.
2. T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning,” in Conference on robot learning, pp. 1094–1100, PMLR, 2020.
3. I. Jendoubi and F. Bouffard, “Multi-agent hierarchical reinforcement learning for energy management,” Appl. Energy, vol. 332, p. 120500, 2023.
4. J. Xie, A. Ajagekar, and F. You, “Multi-agent attention-based deep reinforcement learning for demand response in grid-responsive buildings,” Appl. Energy, vol. 342, p. 121162, 2023.
5. M. U. Yavas, T. Kumbasar, and N. K. Ure, “A real-world reinforcement learning framework for safe and human-like tactical decision-making,” IEEE Trans. Intelligent Transportation Systems, 2023.
6. H. Xu, J. Wu, H. Pan, J. Gu, and X. Guan, “Delay safety-aware digital twin empowered industrial sensing-actuation systems using transferable and reinforced learning,” IEEE Trans. Industrial Informatics, 2023.
7. H. Liu, Q. Liu, C. Rao, F. Wang, F. Alsokhiry, A. V. Shvetsov, and M. A. Mohamed, “An effective energy management layout-based reinforcement learning for household demand response in digital twin simulation,” Solar Energy, vol. 258, pp. 95–105, 2023.
8. H. Xu, X. Zhan, and X. Zhu, “Constraints penalized q-learning for safe offline reinforcement learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, pp. 8753–8760, 2022.



