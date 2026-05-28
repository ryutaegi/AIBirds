import os
import sys
import numpy as np
import time
import tensorflow as tf

# Apple Silicon GPU (MPS) 활성화
print("=" * 70)
print("TensorFlow GPU/MPS 설정")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
print(f"GPU 디바이스 감지: {len(gpus)}개")
if gpus:
    for gpu in gpus:
        print(f"  {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
    print("  GPU 메모리 동적 할당 활성화")
else:
    print("  GPU 감지 안 됨 (CPU 사용)")

cpus = tf.config.list_physical_devices('CPU')
print(f"CPU 디바이스: {len(cpus)}개")

try:
    tf.config.optimizer.set_jit_compilation_enabled(True)
    print("  XLA 컴파일 활성화")
except:
    print("  XLA 컴파일 비활성화")

print("=" * 70 + "\n")

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.demo.v2_rl.env import ScienceBirdsEnv
from src.demo.v2_rl.agent import DQNAgent
from src.demo.v2_rl.stats import Statistics
from src.demo.v2_rl.model import ClassicConvStem, DuelingQHead, SpatialAttention

# ── 저장 경로 ──────────────────────────────────────────────────────────
OUT_DIR    = os.path.join(current_dir, "out", "angry_birds", "ab_agent")
MODELS_DIR = os.path.join(OUT_DIR, "checkpoints")

# ── 주기 설정 ───────────────────────────────────────────────────────────
PRINT_STATS_PERIOD     = 500
PLOT_SAVE_STATS_PERIOD = 2500
CHECKPOINT_SAVE_PERIOD = 1000

# ── 학습 설정 ───────────────────────────────────────────────────────────
REPLAY_PERIOD           = 4      # 4 환경 스텝마다 1회 학습 (1 -> 4)
REPLAY_SIZE_MULTIPLIER  = 4      # 매 학습 시 4배 반복 (원본과 동일)
UPDATE_TARGET_EVERY     = 2000   # 타겟 네트워크 동기화 주기 (500 -> 2000)

def find_latest_checkpoint(models_dir):
    """가장 최신 체크포인트 찾기"""
    if not os.path.exists(models_dir):
        return None, 0

    ckpt_files = sorted(
        [f for f in os.listdir(models_dir) if f.startswith("agent_step_") and f.endswith(".h5")],
        key=lambda f: int(f.replace("agent_step_", "").replace(".h5", ""))
    )

    if not ckpt_files:
        return None, 0

    latest_file = ckpt_files[-1]
    latest_step = int(latest_file.replace("agent_step_", "").replace(".h5", ""))
    latest_path = os.path.join(models_dir, latest_file)

    return latest_path, latest_step

def main():
    print("=" * 70)
    print("Hybrid RL Training (Score Maximization + All Birds)")
    print("=" * 70)
    print("  기법: Double DQN + Dueling + PER + N-step + Exp-Epsilon + Exp-LR")
    print("  목표: 점수 최대화 + 모든 새 사용")

    os.makedirs(MODELS_DIR, exist_ok=True)

    env   = ScienceBirdsEnv(agent_id=2888)
    agent = DQNAgent(
        num_actions    = env.num_actions,
        learning_rate  = 0.00025,    # 0.0001 -> 0.00025
        lr_half_life   = 160000,     # 100000 -> 160000
        gamma          = 0.97,       # 0.90 -> 0.97 (delta reward에 맞춤)
        epsilon_start  = 1.0,
        epsilon_end    = 0.05,       # 0.01 -> 0.05 (200-액션 공간에서 최소 5% 탐색 유지)
        eps_half_life  = 25000,      # 5000 -> 25000 (전체 학습 기간에 걸쳐 탐색 유지)
        n_step         = 3,
        memory_capacity= 80000,      # 30000 -> 80000
    )

    stats = Statistics()
    stats.start_timer()

    batch_size     = 32             # 128 -> 32 (DQN 표준, gradient update 4배 증가)
    total_steps    = 100000

    # 체크포인트에서 재개
    latest_ckpt, resumed_step = find_latest_checkpoint(MODELS_DIR)

    if latest_ckpt:
        print(f"\n 체크포인트 발견! {latest_ckpt}")
        print(f"   Step {resumed_step}부터 재개합니다...\n")
        custom_objects = {
            'ClassicConvStem': ClassicConvStem,
            'DuelingQHead': DuelingQHead,
            'SpatialAttention': SpatialAttention,
        }
        agent.model = tf.keras.models.load_model(latest_ckpt, custom_objects=custom_objects)
        agent.target_model.set_weights(agent.model.get_weights())
        agent.step_count = resumed_step
        # 체크포인트 복원 시 epsilon도 step_count 기준으로 재계산
        # (안 하면 epsilon=1.0으로 시작 → 리플레이 버퍼 찰 때까지 완전 랜덤)
        agent.epsilon = max(
            agent.epsilon_end,
            agent.epsilon_start * (0.5 ** (agent.step_count / agent.eps_half_life)),
        )
        global_step = resumed_step
        episode = (resumed_step // 6)
        stats.load(OUT_DIR + "/")
        # ★ 리플레이 버퍼 초기화 (reward 구조 변경 후 이전 데이터와 충돌 방지)
        agent.memory = __import__('src.demo.v2_rl.memory', fromlist=['PrioritizedReplayBuffer']).PrioritizedReplayBuffer(capacity=agent.memory.capacity)
        agent._nstep = __import__('src.demo.v2_rl.memory', fromlist=['NStepBuffer']).NStepBuffer(agent.n_step, agent.gamma)
        print(f"Stats 로드 완료: {stats.get_num_episodes()} episodes, {stats.get_num_cycles()} cycles")
        print(f"Epsilon 복원: {agent.epsilon:.4f} (step {resumed_step})")
        print(f"리플레이 버퍼 초기화 완료 (reward 구조 변경)\n")
    else:
        print("\n 새로 시작합니다 (체크포인트 없음)\n")
        global_step = 0
        episode = 0

    episode_done = True
    consecutive_timeouts = 0

    while global_step < total_steps:
        if episode_done:
            reset_ok = False
            for reset_attempt in range(2):
                try:
                    state = env.reset()
                    episode_reward = 0
                    episode_score  = 0
                    shots_this_ep  = 0
                    episode_done = False
                    timeout_occurred = False
                    reset_ok = True
                    break
                except Exception as e:
                    print(f"\n reset() 실패 (시도 {reset_attempt+1}/2): {e}")
                    try:
                        env.ar._clear_buffer()
                    except Exception:
                        pass
                    time.sleep(1)
            if not reset_ok:
                print("reset() 실패 - 게임 재시작...")
                try:
                    env.restart_game()
                    state = env.reset()
                    episode_reward = 0
                    episode_score  = 0
                    shots_this_ep  = 0
                    episode_done = False
                    timeout_occurred = False
                except Exception as e:
                    print(f"재시작 후에도 실패: {e}")
                    print("Java/Science Birds를 수동으로 확인하세요")
                    sys.exit(1)

        # 한 스텝 실행
        t_action_start = time.time()
        action = agent.get_action(state, action_mask=env.get_action_mask())
        t_action_end = time.time()
        if t_action_end - t_action_start > 1.0:
            print(f"  get_action() 느림: {(t_action_end - t_action_start)*1000:.0f}ms")

        next_state, reward, done, info = env.step(action)

        # N-step 버퍼를 거쳐 PER에 저장
        agent.store_transition(state, action, reward, next_state, done)

        state          = next_state
        episode_reward += reward
        episode_score   = max(episode_score, info.get("score", 0))
        shots_this_ep  += 1
        global_step    += 1

        # 타임아웃 발생 시 플래그 설정
        if info.get("state") == "timeout":
            timeout_occurred = True
            episode_done = True

        # Replay: REPLAY_PERIOD 스텝마다, REPLAY_SIZE_MULTIPLIER 회 반복 학습
        if global_step % REPLAY_PERIOD == 0 and len(agent.memory) >= batch_size:
            for _ in range(REPLAY_SIZE_MULTIPLIER):
                loss = agent.train(batch_size)
            if loss is not None:
                lr_val = agent.optimizer.learning_rate
                if hasattr(lr_val, '__call__'):
                    lr_val = float(lr_val(agent.step_count))
                else:
                    lr_val = float(lr_val)
                stats.denote_learning_stats(loss, lr_val)

            if agent.step_count % UPDATE_TARGET_EVERY == 0:
                agent.update_target_model()

        # 에피소드 종료
        if done:
            episode_done = True

        # 에피소드가 끝난 후 통계 기록
        if episode_done and shots_this_ep > 0:
            if timeout_occurred:
                consecutive_timeouts += 1
                print(f"\n [타임아웃] Episode {episode + 1} 스킵 (연속: {consecutive_timeouts}회)")
                print(f"  게임 창을 확인하세요!")
            else:
                consecutive_timeouts = 0

                win = (info.get("state") and hasattr(info["state"], "name")
                       and info["state"].name == "WON")
                stats.denote_episode_stats(
                    ret=episode_reward,
                    score=episode_score,
                    shots=shots_this_ep,
                    win=win,
                )

            ep = episode + 1
            print(f"Episode {ep}: Reward={episode_reward:.2f}, Score={episode_score}, "
                  f"Shots={shots_this_ep}, Win={timeout_occurred and 'timeout' or (info.get('state') and hasattr(info['state'], 'name') and info['state'].name == 'WON')}, "
                  f"Epsilon={agent.epsilon:.4f}, GlobalStep={global_step}/{total_steps}")

            if global_step % PRINT_STATS_PERIOD == 0:
                stats.print_stats(global_step, total_steps, agent.epsilon)

            if global_step % PLOT_SAVE_STATS_PERIOD == 0:
                stats.plot_stats(OUT_DIR + "/plots/")
                stats.save(OUT_DIR + "/")
                print(f"  -> Stats & plots saved to {OUT_DIR}")

            if global_step % CHECKPOINT_SAVE_PERIOD == 0:
                ckpt_path = os.path.join(MODELS_DIR, f"agent_step_{global_step}.h5")
                agent.model.save(ckpt_path)
                print(f"  -> Model saved: {ckpt_path}")

            episode += 1

    # 학습 완료 후 최종 저장
    stats.plot_stats(OUT_DIR + "/plots/")
    stats.save(OUT_DIR + "/")
    agent.model.save(os.path.join(MODELS_DIR, "agent_final.h5"))
    print("Training complete.")


if __name__ == "__main__":
    main()
