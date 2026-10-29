import os
import random
import sys
from pathlib import Path
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.distributions import Categorical
from maze_env import ACTION_NAMES, MazeEnv, NUM_ACTIONS

GAMMA = 0.95
GAE_LAMBDA = 0.95
MAX_STEPS = 40
REWARD_SCALE = 0.01
ROLLOUT_STEPS = 512
UPDATE_EPOCHS = 4
NUM_UPDATES = 80
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LOG_EVERY = 5
RANDOM_SEED = 1


def encode_batch(states, rows, cols):
    encoded = torch.zeros(len(states), rows * cols)
    for index, (row, col) in enumerate(states):
        encoded[index, row * cols + col] = 1.0
    return encoded


class ActorCritic(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.policy = nn.Linear(HIDDEN_SIZE, n_actions)
        self.value = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, state_batch):
        hidden = self.body(state_batch)
        logits = self.policy(hidden)
        value = self.value(hidden).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.net = ActorCritic(rows * cols, NUM_ACTIONS)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def encode(self, states):
        return encode_batch(states, self.rows, self.cols)

    def act(self, state):
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(self.encode([state]))
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def value_of(self, state):
        self.net.eval()
        with torch.no_grad():
            _logits, value = self.net(self.encode([state]))
        return float(value.item())

    def best_action(self, state):
        self.net.eval()
        with torch.no_grad():
            logits, _value = self.net(self.encode([state]))
        return int(torch.argmax(logits, dim=1).item())

    def action_probs(self, state):
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(self.encode([state]))
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
        return probabilities, float(value.item())

    def update(self, steps, explain=False):
        states = self.encode([step["state"] for step in steps])
        actions = torch.tensor([step["action"] for step in steps], dtype=torch.int64)
        old_log_probs = torch.tensor([step["log_prob"] for step in steps], dtype=torch.float32)
        returns = torch.tensor([step["return"] for step in steps], dtype=torch.float32)
        advantages = torch.tensor([step["advantage"] for step in steps], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.net.train()
        policy_loss_value = 0.0
        value_loss_value = 0.0
        explained = False
        for _epoch in range(UPDATE_EPOCHS):
            logits, values = self.net(states)
            distribution = Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            entropy = distribution.entropy().mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            policy_loss_value = float(policy_loss.item())
            value_loss_value = float(value_loss.item())
            if explain and not explained:
                print_one_ratio(steps[0], ratio[0].item(), advantages[0].item())
                explained = True
        return policy_loss_value, value_loss_value


def returns_for_episode(episode_steps, bootstrap):
    advantage = 0.0
    next_value = bootstrap
    computed = []
    for step in reversed(episode_steps):
        delta = step["reward"] + GAMMA * next_value - step["value"]
        advantage = delta + GAMMA * GAE_LAMBDA * advantage
        computed.append(advantage)
        next_value = step["value"]
    computed.reverse()
    for step, advantage in zip(episode_steps, computed):
        step["advantage"] = advantage
        step["return"] = advantage + step["value"]
    return episode_steps


def run_episode(agent, env):
    state = env.reset()
    episode_steps = []
    total_reward = 0.0
    reached = False

    for _step in range(MAX_STEPS):
        action, log_prob, value = agent.act(state)
        next_state, reward, done = env.step(action)
        episode_steps.append({
            "state": state,
            "action": action,
            "log_prob": log_prob,
            "reward": reward * REWARD_SCALE,
            "value": value,
        })
        total_reward += reward
        state = next_state
        if done:
            reached = True
            break

    if reached:
        bootstrap = 0.0
    else:
        bootstrap = agent.value_of(state)
    returns_for_episode(episode_steps, bootstrap)
    return episode_steps, total_reward, reached


def print_one_ratio(step, ratio, advantage):
    print()
    print("----- PPO 업데이트 한 건 -----")
    print(f"상태 s = {step['state']}")
    print(f"행동 a = {ACTION_NAMES[step['action']]}")
    print(f"보상 r = {step['reward'] / REWARD_SCALE:.0f}")
    print("학습할 때는 이 보상을 100으로 나눈 값을 쓴다.")
    print(f"가치 목표 = {step['return'] / REWARD_SCALE:.4f}")
    print(f"수집할 때의 V(s) = {step['value'] / REWARD_SCALE:.4f}")
    print(f"이득 A = {step['advantage'] / REWARD_SCALE:.4f}")
    print("학습 전에 이득은 평균 0, 표준편차 1로 맞춘다.")
    print(f"맞춘 뒤의 이득 = {advantage:.4f}")
    print(f"ratio = π_new / π_old = {ratio:.4f}")
    print(f"clip 구간 = {1.0 - CLIP_EPS:.2f} ~ {1.0 + CLIP_EPS:.2f}")
    if advantage > 0:
        print("이득이 양수이므로 이 행동의 확률을 올리되, ratio가 상한을 넘으면 더 올리지 않는다.")
    else:
        print("이득이 음수이므로 이 행동의 확률을 내리되, ratio가 하한보다 작으면 더 내리지 않는다.")
    print("------------------------------")
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
    agent = PPOAgent(env.rows, env.cols)
    explained = False
    finished_updates = 0
    recent_rewards = []
    recent_successes = []
    print("PPO로 같은 미로를 학습합니다.")
    print("한 걸음의 차이 delta = r + gamma * V(s') - V(s)")
    print("이득 A = delta + gamma * lambda * 다음 이득")
    print("ratio = π_new(a|s) / π_old(a|s)")
    print("정책 손실 = -min(ratio * A, clip(ratio) * A)")
    print(f"입력 {env.rows * env.cols} -> 은닉 {HIDDEN_SIZE} -> 정책 {NUM_ACTIONS}개, 가치 1개")
    print(
        f"gamma={GAMMA}, lambda={GAE_LAMBDA}, clip={CLIP_EPS}, 학습률={LEARNING_RATE}, "
        f"보상 배율={REWARD_SCALE}, "
        f"수집={ROLLOUT_STEPS}걸음, 업데이트={NUM_UPDATES}회"
    )
    print()

    for update_index in range(1, NUM_UPDATES + 1):
        steps = []
        rollout_rewards = []
        rollout_successes = []
        while len(steps) < ROLLOUT_STEPS:
            episode_steps, total_reward, reached = run_episode(agent, env)
            steps.extend(episode_steps)
            rollout_rewards.append(total_reward)
            rollout_successes.append(1 if reached else 0)
        policy_loss, value_loss = agent.update(steps, explain=not explained)
        explained = True
        finished_updates = update_index
        recent_rewards.extend(rollout_rewards)
        recent_successes.extend(rollout_successes)

        if update_index % LOG_EVERY == 0:
            window = recent_rewards[-20:]
            success_window = recent_successes[-20:]
            print(
                f"업데이트 {update_index:3d} | "
                f"최근 평균 보상 {sum(window) / len(window):7.1f} | "
                f"성공률 {sum(success_window) / len(success_window) * 100:5.1f}% | "
                f"정책 손실 {policy_loss:7.3f} | "
                f"가치 손실 {value_loss:7.3f}"
            )
    path, total_reward, reached = follow_policy(agent)
    print()
    print(f"업데이트 횟수: {finished_updates}")
    if reached:
        print(f"확률이 가장 큰 행동만 고르면 목표까지 {len(path) - 1}걸음, 보상 합 {total_reward} 입니다.")
    else:
        print("학습된 정책이 목표에 도착하지 못했습니다.")
        print("NUM_UPDATES를 늘리거나 ENTROPY_COEF를 조금 키워 보세요.")
    return agent


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
        screen = pygame.display.set_mode(window_size(env, 760))
        pygame.display.set_caption("PPO 미로 에이전트")
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
            arrows = {cell: agent.best_action(cell) for cell in env.walkable_cells()}

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
            probabilities, state_value = agent.action_probs(env.pos)
            planned = None if env.pos == env.goal else agent.best_action(env.pos)
            prob_lines = []
            for action in range(NUM_ACTIONS):
                probability = float(probabilities[action].item())
                chosen = action == planned
                color = COLOR_GOOD if chosen else COLOR_TEXT
                mark = "  선택" if chosen else ""
                prob_lines.append((f"{ACTION_NAMES[action]}  {probability:6.2f}{mark}", color))

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
                "PPO 학습 결과",
                "",
                f"상태  {env.pos}",
                f"걸음  {play_state['steps']}",
                f"보상 합  {play_state['total_reward']}",
                f"가치 V  {state_value / REWARD_SCALE:8.2f}",
                "",
                ("지금 칸의 행동 확률", COLOR_MUTED),
            ]
            lines.extend(prob_lines)
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
    agent = train()
    print()
    print("학습이 끝났습니다. 창에는 학습된 정책만 보여 줍니다.")
    print("스페이스로 멈추고, R로 다시 보고, ESC로 닫습니다.")
    try:
        watch(agent)
    except ImportError:
        print("pygame이 없어 화면 재생을 열지 못했습니다.")


if __name__ == "__main__":
    main()
