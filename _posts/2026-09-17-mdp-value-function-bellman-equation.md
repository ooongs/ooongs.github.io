---
title: "[RL for LLM] 01. MDP, 가치함수와 벨만 방정식"
date: 2026-09-17 09:00:00 +0800
categories: [STUDY, REINFORCEMENT LEARNING]
tags: [reinforcement_learning, llm, mdp, value_function, bellman_equation]
math: true
---

## 들어가며

강화학습(Reinforcement Learning, RL)의 목표는 단순히 바로 다음 보상을 크게 만드는 것이 아니다. 에이전트(agent)는 현재의 행동이 미래에 미칠 영향까지 고려하여 **장기적인 누적 보상**을 최대화해야 한다.

이를 위해서는 다음 세 가지 질문에 답해야 한다.

1. 에이전트와 환경의 상호작용을 어떻게 수학적으로 표현할 것인가?
2. 현재 상태나 행동이 장기적으로 얼마나 좋은지 어떻게 정의할 것인가?
3. 먼 미래의 보상을 현재의 상태와 행동에 어떻게 연결할 것인가?

첫 번째 질문에 답하는 수학적 모델이 **마르코프 결정 과정(Markov Decision Process, MDP)**이다. 두 번째 질문에서 등장하는 것이 **가치함수(value function)**이고, 세 번째 질문에 답하는 재귀적 관계가 **벨만 방정식(Bellman equation)**이다.

이 글에서는 다음 순서로 개념을 살펴본다.

1. MDP
2. 리턴(return)
3. 상태 가치함수와 행동 가치함수
4. 벨만 기대 방정식
5. 벨만 최적 방정식
6. LLM의 토큰 생성을 MDP로 해석하는 방법

> 이 글에서는 Sutton과 Barto의 표기법을 따른다. 시점 $t$에서 행동 $A_t$를 선택한 뒤 환경으로부터 받는 보상을 $R_{t+1}$로 표기한다.
{: .prompt-info }

## 1. 에이전트와 환경

강화학습에서는 에이전트와 환경이 이산적인 시간 단계 $t=0,1,2,\ldots$에서 상호작용한다.

시간 $t$에서 다음 과정이 일어난다.

1. 에이전트가 상태 $S_t$를 관찰한다.
2. 정책 $\pi$에 따라 행동 $A_t$를 선택한다.
3. 환경이 보상 $R_{t+1}$과 다음 상태 $S_{t+1}$을 반환한다.

이를 간단히 나타내면 다음과 같다.

$$
S_t
\xrightarrow{A_t}
(R_{t+1},S_{t+1})
$$

여기에서 시간 인덱스를 주의해야 한다.

- $S_t$: 행동하기 전의 현재 상태
- $A_t$: 현재 상태에서 선택한 행동
- $R_{t+1}$: 그 행동의 결과로 받은 보상
- $S_{t+1}$: 그 행동의 결과로 이동한 다음 상태

## 2. 마르코프 결정 과정

MDP는 일반적으로 다음 튜플로 정의한다.

$$
\mathcal{M}
=
\left(
\mathcal{S},
\mathcal{A},
p,
\gamma
\right)
$$

각 기호의 의미는 다음과 같다.

| 기호 | 의미 |
|---|---|
| $\mathcal{S}$ | 상태 집합 |
| $\mathcal{A}$ | 행동 집합 |
| $p(s',r\mid s,a)$ | 상태 전이와 보상의 결합 확률 |
| $\gamma$ | 할인율, $0\leq\gamma\leq1$ |

유한 MDP에서 환경의 동역학(dynamics)은 다음과 같이 정의한다.

$$
p(s',r\mid s,a)
=
\Pr
\left\{
S_{t+1}=s',
R_{t+1}=r
\mid
S_t=s,A_t=a
\right\}
$$

즉, 현재 상태가 $s$이고 행동 $a$를 선택했을 때 다음 상태가 $s'$이고 보상이 $r$일 확률이다.

가능한 모든 다음 상태와 보상에 대한 확률의 합은 1이어야 한다.

$$
\sum_{s'\in\mathcal{S}^+}
\sum_{r\in\mathcal{R}}
p(s',r\mid s,a)
=
1
$$

여기서 $\mathcal{S}^+$는 종료 상태(terminal state)를 포함한 상태 집합이다.

### 상태 전이 확률

보상 값을 모두 합하면 상태 전이 확률을 얻는다.

$$
p(s'\mid s,a)
=
\sum_r p(s',r\mid s,a)
$$

### 기대 즉시 보상

상태 $s$에서 행동 $a$를 선택했을 때 받을 즉시 보상의 기댓값은 다음과 같다.

$$
r(s,a)
=
\mathbb{E}
\left[
R_{t+1}\mid S_t=s,A_t=a
\right]
$$

결합 확률로 표현하면 다음과 같다.

$$
r(s,a)
=
\sum_{s',r}
r\,p(s',r\mid s,a)
$$

같은 기호 $r$이 보상의 실현값과 기대 보상 함수에 함께 사용되어 혼란스러울 수 있다. 문맥상 $r$은 하나의 보상 값이고, $r(s,a)$는 그 보상의 기댓값이다.

## 3. 마르코프 성질

MDP의 핵심은 **마르코프 성질(Markov property)**이다.

$$
\begin{aligned}
&\Pr
\left(
S_{t+1}=s',
R_{t+1}=r
\mid
S_0,A_0,\ldots,S_t,A_t
\right) \\
&\qquad =
\Pr
\left(
S_{t+1}=s',
R_{t+1}=r
\mid
S_t,A_t
\right)
\end{aligned}
$$

미래의 상태와 보상의 분포를 예측할 때, 현재 상태와 행동만 알면 과거의 전체 기록은 추가 정보를 제공하지 않는다는 뜻이다.

마르코프 성질은 과거가 중요하지 않다는 의미가 아니다. 필요한 과거 정보가 현재 상태 $S_t$에 충분히 포함되어 있어야 한다는 의미다.

예를 들어 언어 모델에서 현재 상태를 마지막 토큰 하나로만 정의하면 이전 대화에 관한 정보가 사라진다. 반면 프롬프트와 지금까지 생성된 토큰 전체를 상태로 정의하면, 다음 토큰 생성에 필요한 과거 정보를 상태에 포함할 수 있다.

## 4. 정책

정책(policy) $\pi$는 상태가 주어졌을 때 각 행동을 선택할 확률을 나타낸다.

$$
\pi(a\mid s)
=
\Pr(A_t=a\mid S_t=s)
$$

모든 행동 확률의 합은 1이다.

$$
\sum_{a\in\mathcal{A}(s)}
\pi(a\mid s)
=
1
$$

결정적 정책(deterministic policy)은 상태마다 하나의 행동을 선택한다. 확률적 정책(stochastic policy)은 여러 행동에 확률을 분배한다.

언어 모델에서는 다음 토큰 확률분포가 정책에 해당한다.

$$
\pi_\theta(a\mid s)
=
p_\theta(\text{next token}=a\mid\text{context}=s)
$$

## 5. 리턴

에이전트의 목표는 즉시 보상 하나가 아니라 미래의 보상까지 포함한 누적 보상을 최대화하는 것이다.

시간 $t$ 이후의 할인된 누적 보상을 **리턴(return)** $G_t$라고 한다.

무한 시간 문제에서는 다음과 같이 정의한다.

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+
\cdots
$$

합 기호로 표현하면 다음과 같다.

$$
G_t
=
\sum_{k=0}^{\infty}
\gamma^k R_{t+k+1}
$$

종료 시간 $T$를 가지는 에피소드 문제에서는 다음과 같다.

$$
G_t
=
\sum_{k=0}^{T-t-1}
\gamma^k R_{t+k+1}
$$

### 할인율의 역할

할인율은 $0\leq\gamma\leq1$의 값을 가진다.

- $\gamma=0$: 바로 다음 보상만 고려한다.
- $\gamma$가 1에 가까움: 먼 미래의 보상도 중요하게 고려한다.
- $\gamma=1$: 유한 에피소드에서 모든 미래 보상을 동일한 비중으로 합한다.

할인은 미래의 불확실성을 표현할 수도 있지만, 그것만이 목적은 아니다. 무한 시간 문제에서 보상이 계속 누적될 때 리턴을 유한하게 만들고, 가까운 보상을 선호하도록 목표를 정의하는 역할도 한다.

### 리턴의 재귀 구조

리턴에서 첫 번째 보상을 분리해 보자.

$$
\begin{aligned}
G_t
&=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+\cdots \\
&=
R_{t+1}
+
\gamma
\left(
R_{t+2}
+
\gamma R_{t+3}
+\cdots
\right)
\end{aligned}
$$

괄호 안은 $G_{t+1}$이므로 다음 관계를 얻는다.

$$
\boxed{
G_t
=
R_{t+1}
+
\gamma G_{t+1}
}
$$

이 식은 벨만 방정식의 출발점이다.

> 벨만 방정식의 핵심은 특별한 계산 기법이 아니라, 전체 리턴을 “한 단계의 보상”과 “다음 상태부터의 리턴”으로 분해하는 것이다.
{: .prompt-tip }

## 6. 상태 가치함수

정책 $\pi$를 따를 때 상태 $s$에서 기대되는 리턴을 **상태 가치함수(state-value function)**라고 한다.

$$
\boxed{
v_\pi(s)
=
\mathbb{E}_\pi
\left[
G_t\mid S_t=s
\right]
}
$$

가치함수는 이미 받은 보상의 합이 아니다. 현재 상태에서 출발해 앞으로 받을 리턴의 기댓값이다.

같은 상태에서도 어떤 정책을 따르는지에 따라 미래의 행동이 달라지므로 가치 역시 달라진다. 따라서 상태 자체의 가치가 아니라 **정책 $\pi$ 아래에서의 상태 가치**라고 이해해야 한다.

리턴의 정의를 대입하면 다음과 같다.

$$
v_\pi(s)
=
\mathbb{E}_\pi
\left[
\sum_{k=0}^{\infty}
\gamma^k R_{t+k+1}
\;\middle|\;
S_t=s
\right]
$$

## 7. 행동 가치함수

상태 $s$에서 먼저 행동 $a$를 선택하고, 이후 정책 $\pi$를 따를 때 기대되는 리턴을 **행동 가치함수(action-value function)**라고 한다.

$$
\boxed{
q_\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s,A_t=a
\right]
}
$$

두 가치함수의 차이는 첫 번째 행동이 정해져 있는지 여부다.

- $v_\pi(s)$: 첫 행동도 정책 $\pi$에서 샘플링한다.
- $q_\pi(s,a)$: 첫 행동은 $a$로 고정하고, 그 이후부터 정책 $\pi$를 따른다.

따라서 상태 가치함수는 행동 가치함수의 정책 가중 평균이다.

$$
\boxed{
v_\pi(s)
=
\sum_a
\pi(a\mid s)q_\pi(s,a)
}
$$

## 8. 벨만 기대 방정식: 상태 가치함수

이제 상태 가치함수의 정의에서 벨만 방정식을 직접 유도해 보자.

출발점은 다음과 같다.

$$
v_\pi(s)
=
\mathbb{E}_\pi
\left[
G_t\mid S_t=s
\right]
$$

리턴의 재귀 관계 $G_t=R_{t+1}+\gamma G_{t+1}$을 대입한다.

$$
v_\pi(s)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma G_{t+1}
\mid
S_t=s
\right]
$$

현재 상태 $s$에서 선택 가능한 행동 $a$, 다음 상태 $s'$, 보상 $r$에 대해 기댓값을 전개한다.

$$
\begin{aligned}
v_\pi(s)
&=
\sum_a
\pi(a\mid s)
\sum_{s',r}
p(s',r\mid s,a) \\
&\qquad\qquad\cdot
\left[
r
+
\gamma
\mathbb{E}_\pi
\left[
G_{t+1}\mid S_{t+1}=s'
\right]
\right]
\end{aligned}
$$

마르코프 성질과 가치함수의 정의에 따라

$$
\mathbb{E}_\pi
\left[
G_{t+1}\mid S_{t+1}=s'
\right]
=
v_\pi(s')
$$

이므로 다음 식을 얻는다.

$$
\boxed{
v_\pi(s)
=
\sum_a
\pi(a\mid s)
\sum_{s',r}
p(s',r\mid s,a)
\left[
r+\gamma v_\pi(s')
\right]
}
$$

이 식이 정책 $\pi$에 대한 **상태 가치함수의 벨만 기대 방정식(Bellman expectation equation)**이다.

문장으로 읽으면 다음과 같다.

> 현재 상태의 가치는 정책이 선택할 행동, 환경이 반환할 보상과 다음 상태를 모두 평균한 “즉시 보상 + 할인된 다음 상태 가치”다.

기댓값 표기로 간단히 쓰면 다음과 같다.

$$
\boxed{
v_\pi(s)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma v_\pi(S_{t+1})
\mid S_t=s
\right]
}
$$

## 9. 벨만 기대 방정식: 행동 가치함수

행동 가치함수도 같은 방법으로 전개할 수 있다.

$$
q_\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s,A_t=a
\right]
$$

리턴을 한 단계 보상과 나머지 리턴으로 분리한다.

$$
q_\pi(s,a)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma G_{t+1}
\mid
S_t=s,A_t=a
\right]
$$

다음 상태의 기대 리턴은 $v_\pi(S_{t+1})$이므로 다음과 같다.

$$
\boxed{
q_\pi(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\left[
r+\gamma v_\pi(s')
\right]
}
$$

$v_\pi(s')$를 행동 가치함수로 다시 표현하면 다음 식도 얻을 수 있다.

$$
\boxed{
q_\pi(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\left[
r
+
\gamma
\sum_{a'}
\pi(a'\mid s')q_\pi(s',a')
\right]
}
$$

두 표현은 같은 의미를 가진다.

## 10. 숫자로 이해하는 벨만 기대 방정식

LLM이 다음 질문에 답한다고 가정하자.

```text
2 + 2를 설명해 줘.
```

설명을 단순화하기 위해 에이전트가 선택할 수 있는 행동을 실제 토큰이 아닌 두 개의 문장 조각으로 제한한다.

초기 상태 $s_0$에서 다음 행동을 선택할 수 있다.

- $a_1$: `덧셈 과정을 살펴보면`
- $a_2$: `답은 5입니다`

$a_1$을 선택하면 중간 상태 $s_1$로 이동하며 즉시 보상은 0이다. $a_2$를 선택하면 바로 종료되고 보상 $-1$을 받는다.

상태 $s_1$에서는 다음 행동을 선택한다.

- $b_1$: `2와 2를 더하면 4입니다` — 보상 $2$
- $b_2$: `2와 2를 더하면 5입니다` — 보상 $-1$

정책이 다음과 같다고 하자.

$$
\pi(a_1\mid s_0)=0.8,
\qquad
\pi(a_2\mid s_0)=0.2
$$

$$
\pi(b_1\mid s_1)=0.75,
\qquad
\pi(b_2\mid s_1)=0.25
$$

할인율은 $\gamma=0.9$로 설정한다.

### 중간 상태의 가치

$s_1$에서는 다음 행동 이후 바로 종료되므로

$$
\begin{aligned}
v_\pi(s_1)
&=
0.75\times2
+
0.25\times(-1) \\
&=
1.25
\end{aligned}
$$

### 초기 상태에서 각 행동의 가치

$a_1$을 선택하면 즉시 보상은 0이고 다음 상태는 $s_1$이다.

$$
\begin{aligned}
q_\pi(s_0,a_1)
&=
0+\gamma v_\pi(s_1) \\
&=
0.9\times1.25 \\
&=
1.125
\end{aligned}
$$

$a_2$를 선택하면 보상 $-1$을 받고 종료된다. 종료 상태의 가치는 0이다.

$$
q_\pi(s_0,a_2)
=
-1+0.9\times0
=
-1
$$

### 초기 상태의 가치

정책이 두 행동을 선택할 확률로 가중 평균한다.

$$
\begin{aligned}
v_\pi(s_0)
&=
0.8q_\pi(s_0,a_1)
+
0.2q_\pi(s_0,a_2) \\
&=
0.8\times1.125
+
0.2\times(-1) \\
&=
0.7
\end{aligned}
$$

이 계산에는 벨만 방정식의 구조가 그대로 나타난다.

$$
\text{현재 가치}
=
\text{즉시 보상}
+
\gamma\times\text{다음 상태의 가치}
$$

## 11. 벨만 방정식의 행렬 표현

유한한 상태 집합에서는 벨만 기대 방정식을 행렬로 표현할 수 있다.

정책 $\pi$ 아래에서 기대 즉시 보상을 다음과 같이 정의한다.

$$
r_\pi(s)
=
\sum_a
\pi(a\mid s)
\sum_{s',r}
p(s',r\mid s,a)r
$$

정책에 의해 만들어지는 상태 전이 확률은 다음과 같다.

$$
P_\pi(s,s')
=
\sum_a
\pi(a\mid s)
\sum_r
p(s',r\mid s,a)
$$

모든 상태의 가치를 벡터 $\mathbf{v}_\pi$, 기대 보상을 벡터 $\mathbf{r}_\pi$, 전이 확률을 행렬 $\mathbf{P}_\pi$로 나타내면

$$
\boxed{
\mathbf{v}_\pi
=
\mathbf{r}_\pi
+
\gamma\mathbf{P}_\pi\mathbf{v}_\pi
}
$$

가 된다. 가치 벡터가 포함된 항을 왼쪽으로 이동하면

$$
\left(
\mathbf{I}
-
\gamma\mathbf{P}_\pi
\right)
\mathbf{v}_\pi
=
\mathbf{r}_\pi
$$

이다. 역행렬이 존재한다면 가치함수는 다음과 같이 계산할 수 있다.

$$
\boxed{
\mathbf{v}_\pi
=
\left(
\mathbf{I}
-
\gamma\mathbf{P}_\pi
\right)^{-1}
\mathbf{r}_\pi
}
$$

따라서 유한 MDP의 환경 모델을 정확히 알고 있다면 연립방정식을 풀어 정책의 가치함수를 계산할 수 있다.

하지만 LLM처럼 상태 공간이 사실상 무한하고 행동 공간이 전체 어휘인 문제에서는 이 행렬을 직접 만들 수 없다. 실제 강화학습에서는 샘플과 신경망을 이용해 가치함수를 근사한다.

## 12. 벨만 연산자와 고정점

벨만 기대 연산자 $\mathcal{T}^\pi$를 다음과 같이 정의하자.

$$
\left(
\mathcal{T}^\pi v
\right)(s)
=
\sum_a
\pi(a\mid s)
\sum_{s',r}
p(s',r\mid s,a)
\left[
r+\gamma v(s')
\right]
$$

벨만 기대 방정식은 가치함수 $v_\pi$가 이 연산자의 고정점(fixed point)이라는 의미다.

$$
\boxed{
v_\pi
=
\mathcal{T}^\pi v_\pi
}
$$

$0\leq\gamma<1$일 때 이 연산자는 최대 노름에서 수축 사상(contraction mapping)이 된다.

$$
\left\|
\mathcal{T}^\pi v
-
\mathcal{T}^\pi w
\right\|_\infty
\leq
\gamma
\left\|
v-w
\right\|_\infty
$$

따라서 임의의 초기 가치함수 $v_0$에서 시작해

$$
v_{k+1}
=
\mathcal{T}^\pi v_k
$$

를 반복하면 유일한 고정점 $v_\pi$로 수렴한다. 이 성질이 반복적 정책 평가(iterative policy evaluation)와 동적 계획법(dynamic programming)의 수학적 기반이다.

> $\gamma=1$인 유한 에피소드 문제에서는 종료 시점으로부터 역순으로 값을 계산하는 backward induction을 사용할 수 있다. $\gamma<1$인 무한 시간 문제의 수축 성질과는 조건이 다르다.
{: .prompt-warning }

## 13. 최적 가치함수

지금까지는 주어진 정책 $\pi$가 얼마나 좋은지를 평가했다. 이제 가능한 정책 중 가장 좋은 정책을 생각해 보자.

최적 상태 가치함수는 다음과 같다.

$$
v_*(s)
=
\max_\pi v_\pi(s)
$$

최적 행동 가치함수는 다음과 같다.

$$
q_*(s,a)
=
\max_\pi q_\pi(s,a)
$$

최적 상태 가치는 현재 상태에서 선택할 수 있는 행동 중 가장 큰 최적 행동 가치와 같다.

$$
\boxed{
v_*(s)
=
\max_a q_*(s,a)
}
$$

## 14. 벨만 최적 방정식

벨만 기대 방정식에서는 정책 $\pi$에 따라 행동의 가중 평균을 계산했다.

$$
v_\pi(s)
=
\sum_a
\pi(a\mid s)q_\pi(s,a)
$$

반면 최적 가치함수에서는 가장 가치가 큰 행동을 선택한다.

$$
\boxed{
v_*(s)
=
\max_a
\sum_{s',r}
p(s',r\mid s,a)
\left[
r+\gamma v_*(s')
\right]
}
$$

이 식이 상태 가치함수에 대한 **벨만 최적 방정식(Bellman optimality equation)**이다.

행동 가치함수에 대해서는 다음과 같다.

$$
\boxed{
q_*(s,a)
=
\sum_{s',r}
p(s',r\mid s,a)
\left[
r
+
\gamma
\max_{a'}
q_*(s',a')
\right]
}
$$

두 종류의 벨만 방정식을 구분해야 한다.

| 구분 | 행동 처리 | 목적 |
|---|---|---|
| 벨만 기대 방정식 | 정책 $\pi$로 가중 평균 | 주어진 정책 평가 |
| 벨만 최적 방정식 | 가장 큰 행동 가치를 선택 | 최적 정책 탐색 |

최적 행동 가치함수를 알고 있다면 다음과 같은 탐욕 정책을 만들 수 있다.

$$
\pi_*(s)
\in
\operatorname*{arg\,max}_a
q_*(s,a)
$$

즉, 각 상태에서 $q_*(s,a)$가 가장 큰 행동을 선택하면 최적 정책을 얻을 수 있다.

## 15. 앞의 예제에서 최적 가치 계산하기

앞의 LLM 예제로 돌아가 보자.

상태 $s_1$에서 가능한 행동의 보상은 다음과 같았다.

$$
q_*(s_1,b_1)=2,
\qquad
q_*(s_1,b_2)=-1
$$

최적 정책은 $b_1$을 선택하므로

$$
v_*(s_1)
=
\max(2,-1)
=
2
$$

초기 상태에서 $a_1$을 선택하면

$$
q_*(s_0,a_1)
=
0+0.9\times2
=
1.8
$$

$a_2$를 선택하면 바로 종료되므로

$$
q_*(s_0,a_2)
=
-1
$$

따라서 초기 상태의 최적 가치는

$$
v_*(s_0)
=
\max(1.8,-1)
=
1.8
$$

기존 정책의 가치는 $v_\pi(s_0)=0.7$이었지만, 최적 가치는 $v_*(s_0)=1.8$이다.

- $v_\pi$: 현재 정책으로 얻을 수 있는 기대 리턴
- $v_*$: 가능한 정책 중 가장 좋은 정책으로 얻을 수 있는 기대 리턴

## 16. LLM의 토큰 생성을 MDP로 표현하기

프롬프트를 $x$, 지금까지 생성한 토큰을 $y_{1:t-1}$라고 하자.

LLM의 상태를 다음과 같이 정의할 수 있다.

$$
S_t
=
(x,y_1,\ldots,y_{t-1})
$$

행동은 다음 토큰이다.

$$
A_t=y_t
$$

정책은 언어 모델이 출력하는 다음 토큰 확률분포다.

$$
\pi_\theta(A_t\mid S_t)
=
p_\theta
\left(
y_t\mid x,y_{1:t-1}
\right)
$$

토큰 $a$를 선택한 뒤의 다음 상태는 기존 문맥에 토큰을 연결한 결과다.

$$
S_{t+1}
=
S_t\oplus a
$$

단일 응답을 생성하는 동안에는 이 상태 전이가 결정적이다. 따라서 다음 상태를 $s\oplus a$로 직접 쓸 수 있다.

즉시 보상을 $r(s,a)$라고 하면 LLM의 벨만 기대 방정식은 다음처럼 단순화된다.

$$
\boxed{
q_\pi(s,a)
=
r(s,a)
+
\gamma v_\pi(s\oplus a)
}
$$

$$
\boxed{
v_\pi(s)
=
\sum_a
\pi(a\mid s)
\left[
r(s,a)
+
\gamma v_\pi(s\oplus a)
\right]
}
$$

답변이 완성된 후에만 보상 모델 점수가 주어진다면 중간 토큰의 보상은 0이다.

$$
r(s_t,a_t)=0,
\qquad t<T-1
$$

마지막 토큰을 생성해 종료 상태에 도달했을 때 보상을 받는다.

$$
r(s_{T-1},a_{T-1})
=
R_T
$$

따라서 중간 상태의 가치는 현재 문맥에서 생성을 계속했을 때 받을 **최종 답변 점수의 기댓값**이 된다.

$$
v_\pi(s_t)
=
\mathbb{E}_\pi
\left[
R_T\mid S_t=s_t
\right]
$$

여기서는 이해를 위해 $\gamma=1$인 유한 에피소드를 가정했다.

## 17. 가치함수는 LLM에서 무엇을 알려주는가?

LLM 관점에서 상태 가치함수는 다음 질문에 답한다.

> 현재까지 작성된 답변을 이어서 완성하면 최종적으로 얼마나 좋은 평가를 받을 것으로 예상되는가?

행동 가치함수는 다음 질문에 답한다.

> 현재 문맥에서 특정 토큰을 선택한 뒤 답변을 완성하면 얼마나 좋은 평가를 받을 것으로 예상되는가?

예를 들어 현재 문맥이 다음과 같다고 하자.

```text
벨만 방정식의 핵심은 현재 상태의 가치가
```

다음 토큰 후보가 두 개 있다고 가정한다.

- `즉시`
- `과거`

`즉시`를 선택하면 “즉시 보상과 다음 상태 가치”라는 올바른 설명으로 이어질 가능성이 높다. 반면 `과거`를 선택하면 잘못된 설명으로 이어질 가능성이 있다.

$$
q_\pi(s,\text{즉시})
>
q_\pi(s,\text{과거})
$$

강화학습은 좋은 최종 결과로 이어지는 행동의 확률을 높이는 방향으로 정책을 업데이트한다.

다만 실제 LLM에서는 어휘 수가 매우 많고 가능한 문맥의 수가 사실상 무한하다. 따라서 모든 $q_\pi(s,a)$를 표 형태로 저장하지 않고, 신경망으로 가치함수를 근사하거나 샘플링한 응답으로 리턴을 추정한다.

## 18. 자주 혼동하는 부분

### 보상과 가치의 차이

보상은 환경이 한 전이 후 제공하는 즉각적인 신호다.

$$
R_{t+1}
$$

가치는 현재 상태 또는 행동 이후에 받을 전체 리턴의 기댓값이다.

$$
v_\pi(s)
=
\mathbb{E}_\pi[G_t\mid S_t=s]
$$

따라서 현재 보상이 0이어도 미래에 큰 보상을 받을 수 있다면 상태 가치는 높을 수 있다.

### 가치함수는 정책에 의존한다

같은 상태라도 좋은 정책을 따르면 가치가 높고, 나쁜 정책을 따르면 가치가 낮다.

$$
v_{\pi_1}(s)
\neq
v_{\pi_2}(s)
$$

### 벨만 방정식은 정의가 아니라 일관성 조건이다

가치함수의 정의는 기대 리턴이다.

$$
v_\pi(s)
=
\mathbb{E}_\pi[G_t\mid S_t=s]
$$

벨만 방정식은 이 가치함수가 만족해야 하는 재귀적 관계다.

$$
v_\pi(s)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma v_\pi(S_{t+1})
\mid S_t=s
\right]
$$

### 벨만 기대 방정식과 최적 방정식은 다르다

정책을 평가할 때는 행동을 정책 확률로 평균한다.

$$
\sum_a\pi(a\mid s)(\cdots)
$$

최적 정책을 구할 때는 가장 가치가 큰 행동을 선택한다.

$$
\max_a(\cdots)
$$

## 정리

MDP는 에이전트와 환경의 상호작용을 다음 확률로 표현한다.

$$
p(s',r\mid s,a)
$$

리턴은 미래 보상의 할인합이다.

$$
G_t
=
\sum_{k=0}^{\infty}
\gamma^kR_{t+k+1}
$$

상태 가치함수는 정책을 따를 때 상태에서 기대되는 리턴이다.

$$
v_\pi(s)
=
\mathbb{E}_\pi
\left[
G_t\mid S_t=s
\right]
$$

행동 가치함수는 상태에서 특정 행동을 선택한 뒤 기대되는 리턴이다.

$$
q_\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t\mid S_t=s,A_t=a
\right]
$$

벨만 기대 방정식은 현재 가치와 다음 상태 가치의 관계를 나타낸다.

$$
v_\pi(s)
=
\sum_a
\pi(a\mid s)
\sum_{s',r}
p(s',r\mid s,a)
\left[
r+\gamma v_\pi(s')
\right]
$$

벨만 최적 방정식은 가장 좋은 행동을 선택했을 때의 가치를 나타낸다.

$$
v_*(s)
=
\max_a
\sum_{s',r}
p(s',r\mid s,a)
\left[
r+\gamma v_*(s')
\right]
$$

결국 벨만 방정식의 핵심은 다음과 같다.

$$
\boxed{
\text{현재 가치}
=
\text{즉시 보상}
+
\text{할인된 미래 가치}
}
$$

이 재귀 구조 덕분에 최종적으로 받은 보상을 이전 상태와 행동에 연결할 수 있다. 다음 글에서는 이 가치함수를 이용해 행동의 상대적인 품질을 측정하는 **어드밴티지 함수(advantage function)**와 정책 경사(policy gradient)를 살펴볼 예정이다.

## 참고 자료

1. Richard S. Sutton and Andrew G. Barto, [Reinforcement Learning: An Introduction, Second Edition](http://incompleteideas.net/book/the-book-2nd.html)
2. Richard S. Sutton and Andrew G. Barto, [Chapter 3: Finite Markov Decision Processes](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
3. Seongmin.C, [가치 함수 및 벨만 방정식 정의 및 증명](https://smcho1201.tistory.com/126)
4. Long Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
