"""
같은 미로를 신경망으로 학습(DQN)

Q-learning.py의 Q표와 비교하면서 읽음
환경, 보상, 행동 번호, 엡실론-그리디, 목표값 식은 그대로!!
바뀌는 것은 Q값을 딕셔너리에 저장하지 않고, 신경망이 계산한다는 점
"""


import random
import sys
from collections import deque

import numpy as np
import torch
from torch import nn, optim

from maze_env import ACTION_NAMES, NUM_ACTIONS, MazeEnv

# ── 설정 ─────────────────────────────────────────────────────
SEED = 2026

LR = 1e-3                 # 학습률 (Q-learning의 알파 자리)
GAMMA = 0.95              # 할인율

EPISODES = 600
TRAIN_MAX_STEPS = 300     # 학습 중 한 판 최대 걸음
EVAL_MAX_STEPS = 200      # 정책 확인(재생)용 최대 걸음 (maze_game과 동일)

EPS_START = 1.0           # 엡실론-그리디: 처음엔 거의 무작위
EPS_END = 0.05
EPS_DECAY = 0.99          # 에피소드마다 곱해서 감소

BUFFER_SIZE = 10000
BATCH_SIZE = 64
LEARN_START = 200         # 경험이 이만큼 쌓이면 학습 시작
TARGET_SYNC = 200         # 이 걸음마다 Online -> Target 복사

# 학습 과정 재생(watch_agent)에서 보여줄 시점
SNAPSHOT_EPISODES = {1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 1. Q-Network: 칸 -> 행동 4개의 Q값 ───────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim, actions=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, actions),
        )

    def forward(self, x):
        return self.net(x)


# ── 2. Replay Buffer ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.data = deque(maxlen=capacity)

    def add(self, s, a, r, ns, done):
        self.data.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.as_tensor(np.asarray(s), dtype=torch.float32),
                torch.as_tensor(a, dtype=torch.long).unsqueeze(1),
                torch.as_tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.as_tensor(np.asarray(ns), dtype=torch.float32),
                torch.as_tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.data)


# ── 3. DQN 에이전트 ──────────────────────────────────────────
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.rows, self.cols = env.rows, env.cols
        self.obs_dim = self.rows * self.cols       # 칸 하나당 입력 1개 (원-핫)

        self.online = QNetwork(self.obs_dim)
        self.target = QNetwork(self.obs_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)
        self.replay = ReplayBuffer()
        self.steps = 0
        self._table = {}                           # {칸: Q값 4개} 읽기용 캐시
        self.refresh_table()

    # 칸 (row, col) -> 원-핫 벡터. Q표의 "칸 하나 = 표 한 줄"과 같은 정보를 입력으로 준다
    def encode(self, state):
        x = np.zeros(self.obs_dim, dtype=np.float32)
        x[state[0] * self.cols + state[1]] = 1.0
        return x

    # 엡실론-그리디 (Q-learning과 같은 규칙)
    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS)
        with torch.no_grad():
            x = torch.as_tensor(self.encode(state)).unsqueeze(0)
            return int(self.online(x).argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay.add(self.encode(state), action, reward, self.encode(next_state), float(done))

    # 목표값: reward + gamma * max Q_target(next) (도착하면 미래 가치 없음)
    def learn(self):
        if len(self.replay) < max(BATCH_SIZE, LEARN_START):
            return None
        s, a, r, ns, d = self.replay.sample(BATCH_SIZE)
        chosen_q = self.online(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target(ns).max(dim=1, keepdim=True).values
            targets = r + GAMMA * (1 - d) * next_q
        loss = nn.functional.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    # ── maze_game.watch_agent()가 쓰는 인터페이스 ──
    # agent.q = snapshot["q"] 로 시점을 바꾸므로, 가중치를 q로 읽고 쓸 수 있게 한다
    @property
    def q(self):
        return {k: v.detach().clone() for k, v in self.online.state_dict().items()}

    @q.setter
    def q(self, weights):
        self.online.load_state_dict(weights)
        self.refresh_table()

    # 모든 칸의 Q값을 한 번에 계산해 둔다 (화면이 매 프레임 읽기 때문)
    def refresh_table(self):
        cells = self.env.walkable_cells()
        x = torch.as_tensor(np.stack([self.encode(c) for c in cells]))
        with torch.no_grad():
            values = self.online(x).numpy()
        self._table = {cell: values[i] for i, cell in enumerate(cells)}

    def q_get(self, state, action):
        return float(self._table[state][action])

    def best_action(self, state, random_tie=True):
        values = self._table[state]
        best = values.max()
        ties = [a for a in range(NUM_ACTIONS) if abs(values[a] - best) < 1e-9]
        return random.choice(ties) if random_tie else ties[0]


# ── 4. 지금 정책으로 목표까지 가는지 확인 (탐험 없음) ────────
def greedy_rollout(agent, env):
    agent.refresh_table()
    state = env.reset()
    seen = {state}
    for step in range(1, EVAL_MAX_STEPS + 1):
        action = agent.best_action(state, random_tie=False)
        next_state, _reward, done = env.step(action)
        if done:
            return True, step
        if next_state == state or next_state in seen:   # 벽에 막힘 / 같은 길 반복
            return False, step
        seen.add(next_state)
        state = next_state
    return False, EVAL_MAX_STEPS


# ── 5. 학습 ──────────────────────────────────────────────────
def train(agent, env, episodes=EPISODES):
    eval_env = MazeEnv()
    successes = []        # 에피소드별 도착 여부 (1/0) -> 성공률 그래프
    snapshots = []        # 학습 과정 재생용

    for episode in range(1, episodes + 1):
        epsilon = max(EPS_END, EPS_START * EPS_DECAY ** (episode - 1))
        state = env.reset()
        reached = False

        for _ in range(TRAIN_MAX_STEPS):
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            agent.steps += 1
            if agent.steps % TARGET_SYNC == 0:
                agent.sync_target()

            state = next_state
            if done:
                reached = True
                break

        successes.append(1 if reached else 0)

        if episode in SNAPSHOT_EPISODES:
            ok, steps = greedy_rollout(agent, eval_env)
            snapshots.append({"episode": episode, "q": agent.q, "reached": ok, "steps": steps})

        if episode % 50 == 0:
            recent = np.mean(successes[-50:])
            print(f"에피소드 {episode:4d} | 최근 50판 성공률 {recent:.2f} | 엡실론 {epsilon:.2f}")

    return successes, snapshots


def print_policy(agent, env):
    agent.refresh_table()
    marks = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    print("\n학습된 정책 (칸마다 Q값이 가장 큰 행동)")
    for row in range(env.rows):
        line = ""
        for col in range(env.cols):
            if env.grid[row][col] == 1:
                line += "#"
            elif (row, col) == env.goal:
                line += "G"
            else:
                line += marks[agent.best_action((row, col), random_tie=False)]
        print(line)


def main():
    env = MazeEnv()
    agent = DQNAgent(env)

    successes, snapshots = train(agent, env)

    final = snapshots[-1]
    agent.q = final["q"]
    print_policy(agent, env)
    ok, steps = greedy_rollout(agent, MazeEnv())
    print(f"\n최종 정책: {'목표 도착' if ok else '도착 실패'} ({steps}걸음)")

    if "--no-gui" in sys.argv:
        return
    try:
        from maze_game import watch_agent
    except ImportError:
        print("pygame이 없어 재생 화면은 건너뜁니다. (pip install pygame)")
        return
    summary = {"alpha": LR, "gamma": GAMMA}
    watch_agent(agent, successes, summary, snapshots)


if __name__ == "__main__":
    main()