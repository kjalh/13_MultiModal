import os
import random
import sys
from collections import deque
from pathlib import Path
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from maze_env import ACTION_NAMES, MazeEnv, NUM_ACTIONS

EPISODES = 600
MAX_STEPS = 40
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.992
LOG_EVERY = 50
RANDOM_SEED = 1
BATCH_SIZE = 64
MEMORY_SIZE = 4000
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
TARGET_UPDATE = 200


def encode_batch(states, rows, cols):
    """
        칸 좌표 여러 개를 원 핫 행렬로 만듦
        - 칸(행, 열)은 길이 rows*cols의 벡어테서 한 자라만 1
        - 표에서 상태마다 칸이 따로 있던 것과 같은 대응
    """
    encoded = torch.zeros(len(states), rows * cols)
    for index, (row, col) in enumerate(states):
        encoded[index, row * cols + col] = 1.0
    return encoded


class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions),
        )

    def forward(self, state_batch):
        return self.layers(state_batch)


class ReplayMemory:
    def __init__(self, capacity):
        self.items = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.items.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.items, batch_size)

    def __len__(self):
        return len(self.items)


class DQNAgent:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        n_inputs = rows * cols
        self.policy_net = QNetwork(n_inputs, NUM_ACTIONS)
        self.target_net = QNetwork(n_inputs, NUM_ACTIONS)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.learn_steps = 0

    def encode(self, states):
        return encode_batch(states, self.rows, self.cols)

    def q_values(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            values = self.policy_net(self.encode([state]))
        return values.squeeze(0)

    def best_action(self, state):
        return int(torch.argmax(self.q_values(state)).item())

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS), True
        return self.best_action(state), False

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self, explain=False):
        if len(self.memory) < BATCH_SIZE:
            return None
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_batch = self.encode(states)
        next_batch = self.encode(next_states)
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32)
        self.policy_net.train()
        predicted = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_max = self.target_net(next_batch).max(dim=1).values
            targets = reward_batch + GAMMA * next_max * (1.0 - done_batch)
        loss = nn.functional.mse_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1

        if self.learn_steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if explain:
            print_one_sample(
                states[0], actions[0], rewards[0], next_states[0], dones[0],
                predicted[0].item(), targets[0].item(),
            )
        return float(loss.item())


def print_one_sample(state, action, reward, next_state, done, predicted, target):
    print()
    print("----- 신경망 업데이트 한 건 -----")
    print(f"상태 s = {state}")
    print(f"행동 a = {ACTION_NAMES[action]}")
    print(f"보상 r = {reward}")
    print(f"다음 상태 s' = {next_state}")
    print(f"종료 = {done}")
    if done:
        print("종료 상태라 미래 가치는 0이고, target은 보상 자체다.")
    else:
        print("target = r + gamma * max Q_target(s', a')")
    print(f"신경망이 예측한 Q(s, a) = {predicted:.4f}")
    print(f"target = {target:.4f}")
    print(f"차이 = target - 예측 = {target - predicted:.4f}")
    print("이 차이를 제곱한 것이 이 경험의 손실이다.")
    print("backward()가 그 손실을 줄이는 쪽으로 가중치를 고친다.")
    print("--------------------------------")
    print()


def follow_policy(agent):
    env = MazeEnv()
    state = env.reset()
    path = [state]
    seen = {state}
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = agent.best_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state)
        if done:
            return path, total_reward, True
        if next_state == state or next_state in seen:
            return path, total_reward, False
        seen.add(next_state)
        state = next_state
    return path, total_reward, False


def train():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    env = MazeEnv()
    agent = DQNAgent(env.rows, env.cols)
    epsilon = EPSILON_START
    rewards = []
    successes = []
    losses = []
    explained = False
    print("DQN으로 같은 미로를 학습합니다.")
    print("target = r + gamma * max Q_target(s', a')")
    print("손실 = (예측 Q(s, a) - target)^2 의 평균")
    n_inputs = env.rows * env.cols
    print(f"입력 {n_inputs} -> 은닉 {HIDDEN_SIZE} -> 출력 {NUM_ACTIONS}")
    print(
        f"gamma={GAMMA}, 학습률={LEARNING_RATE}, 배치={BATCH_SIZE}, "
        f"에피소드={EPISODES}"
    )
    print()

    for episode in range(1, EPISODES + 1):
        current_epsilon = epsilon
        state = env.reset()
        total_reward = 0.0
        reached_goal = False
        episode_losses = []

        for _step in range(1, MAX_STEPS + 1):
            action, _explored = agent.choose_action(state, current_epsilon)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            explain = (not explained) and len(agent.memory) >= BATCH_SIZE
            loss = agent.learn(explain=explain)
            if explain:
                explained = True
            if loss is not None:
                episode_losses.append(loss)
            total_reward += reward
            state = next_state
            if done:
                reached_goal = True
                break
        rewards.append(total_reward)
        successes.append(1 if reached_goal else 0)
        if episode_losses:
            losses.append(sum(episode_losses) / len(episode_losses))
        epsilon = max(EPSILON_MIN, current_epsilon * EPSILON_DECAY)

        if episode % LOG_EVERY == 0:
            recent_reward = sum(rewards[-LOG_EVERY:]) / LOG_EVERY
            recent_success = sum(successes[-LOG_EVERY:]) / LOG_EVERY
            if losses:
                recent_loss = sum(losses[-LOG_EVERY:]) / min(LOG_EVERY, len(losses))
            else:
                recent_loss = 0.0
            print(
                f"에피소드 {episode:4d} | "
                f"최근 평균 보상 {recent_reward:7.1f} | "
                f"성공률 {recent_success * 100:5.1f}% | "
                f"평균 손실 {recent_loss:7.3f} | "
                f"엡실론 {current_epsilon:.3f}"
            )
    path, total_reward, reached = follow_policy(agent)
    print()
    print(f"경험 버퍼에 남은 걸음: {len(agent.memory)}개")
    if reached:
        print(f"탐욕 정책은 목표까지 {len(path) - 1}걸음, 보상 합 {total_reward} 입니다.")
    else:
        print("탐욕 정책이 목표에 도착하지 못했습니다.")
        print("EPISODES를 늘리거나 EPSILON_DECAY를 1에 더 가깝게 바꿔 보세요.")
    return agent, rewards, successes


def watch(agent):
    import pygame
    from maze_game import (
        COLOR_AGENT,
        COLOR_BG,
        COLOR_GOOD,
        COLOR_MUTED,
        COLOR_PANEL,
        COLOR_TEXT,
        MOVE_DELAY_MS,
        STAGE_HOLD_MS,
        blit_lines,
        draw_maze,
        make_font,
        panel_rect,
        window_size,
    )
    pygame.init()
    try:
        env = MazeEnv()
        screen = pygame.display.set_mode(window_size(env))
        pygame.display.set_caption("DQN 미로 에이전트")
        clock = pygame.time.Clock()
        body_font = make_font(17)
        cell_font = make_font(16, bold=True)

        def begin():
            env.reset()
            return {
                "trail": [env.pos],
                "seen": {env.pos},
                "steps": 0,
                "total_reward": 0,
                "finished": False,
                "paused": False,
                "elapsed": 0,
                "hold": 0,
                "message": "학습된 정책으로 이동합니다.",
            }
        play_state = begin()
        running = True
        while running:
            dt = clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        play_state = begin()
                    elif event.key == pygame.K_SPACE:
                        play_state["paused"] = not play_state["paused"]
            arrows = {}
            for cell in env.walkable_cells():
                arrows[cell] = agent.best_action(cell)

            if not play_state["finished"] and not play_state["paused"]:
                play_state["elapsed"] += dt
                if play_state["elapsed"] >= MOVE_DELAY_MS:
                    play_state["elapsed"] = 0
                    before = env.pos
                    action = agent.best_action(before)
                    _state, reward, done = env.step(action)
                    play_state["trail"].append(env.pos)
                    play_state["steps"] += 1
                    play_state["total_reward"] += reward
                    if done:
                        play_state["finished"] = True
                        play_state["message"] = "목표에 도착했습니다."
                    elif env.pos == before:
                        play_state["finished"] = True
                        play_state["message"] = "벽에 막혔습니다."
                    elif env.pos in play_state["seen"]:
                        play_state["finished"] = True
                        play_state["message"] = "같은 길로 돌아왔습니다."
                    elif play_state["steps"] >= MAX_STEPS:
                        play_state["finished"] = True
                        play_state["message"] = "걸음 제한에 도달했습니다."
                    else:
                        play_state["seen"].add(env.pos)
            elif play_state["finished"] and not play_state["paused"]:
                play_state["hold"] += dt
                if play_state["hold"] >= STAGE_HOLD_MS:
                    play_state = begin()
            values = agent.q_values(env.pos)
            planned = None if env.pos == env.goal else agent.best_action(env.pos)
            q_lines = []
            for action in range(NUM_ACTIONS):
                value = float(values[action].item())
                chosen = action == planned
                color = COLOR_GOOD if chosen else COLOR_TEXT
                mark = "  선택" if chosen else ""
                q_lines.append((f"{ACTION_NAMES[action]}  {value:8.2f}{mark}", color))

            if play_state["finished"]:
                pause_text = "잠시 후 다시 재생"
            elif play_state["paused"]:
                pause_text = "일시정지 중"
            else:
                pause_text = "재생 중"
            screen.fill(COLOR_BG)
            panel = panel_rect(screen, env)
            pygame.draw.rect(screen, COLOR_PANEL, panel, border_radius=10)
            agent_color = COLOR_GOOD if env.pos == env.goal else COLOR_AGENT
            draw_maze(screen, env, play_state["trail"], arrows, agent_color, cell_font)
            lines = [
                "DQN 학습 결과",
                "",
                f"상태  {env.pos}",
                f"걸음  {play_state['steps']}",
                f"보상 합  {play_state['total_reward']}",
                "",
                ("지금 칸의 Q값", COLOR_MUTED),
            ]
            lines.extend(q_lines)
            lines.extend([
                "",
                play_state["message"],
                (pause_text, COLOR_MUTED),
                "",
                ("스페이스  일시정지", COLOR_MUTED),
                ("R  이 판 다시", COLOR_MUTED),
                ("ESC  종료", COLOR_MUTED),
            ])
            blit_lines(screen, body_font, lines, panel.x + 16, panel.y + 14)
            pygame.display.flip()
    finally:
        pygame.quit()


def main():
    agent, _rewards, _successes = train()
    print()
    print("학습이 끝났습니다. 창에는 학습된 정책만 보여 줍니다.")
    print("스페이스로 멈추고, R로 다시 보고, ESC로 닫습니다.")
    try:
        watch(agent)
    except ImportError:
        print("pygame이 없어 화면 재생을 열지 못했습니다.")


if __name__ == "__main__":
    main()
