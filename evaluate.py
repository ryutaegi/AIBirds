"""
모델 평가 스크립트
================================
  - SavedModel 또는 .h5 체크포인트를 로드해 평가
  - 원본 evaluate_ab.py와 동일한 레벨 풀 (FILTERED_TRAIN_LEVELS 200개)
  - 평가 시 config.xml 자동 교체 → 평가 후 학습용 자동 복구
결과 저장:
  - out/angry_birds/<모델명>/evaluation_results.csv
  - out/angry_birds/<모델명>/evaluation_summary.txt
  - out/angry_birds/<모델명>/score_distribution.png

사용법:
  python3 evaluate.py               # 그냥 실행하면 끝 (config 자동 처리)
"""

import os
import sys
import csv
import re
import shutil
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.demo.v2_rl.env import ScienceBirdsEnv
from src.demo.v2_rl.model import ClassicConvStem, DuelingQHead, NoisyDense, SpatialAttention
from client.agent_client import GameState

# ─────────────────────────────────────────────────────────────
# 평가 설정
# ─────────────────────────────────────────────────────────────

MODEL_NAME = "ab_agent_v1"
SIM_SPEED = 50
AGENT_ID = 2888

# ─────────────────────────────────────────────────────────────
# 원본과 동일한 FILTERED_TRAIN_LEVELS (200개)
# ─────────────────────────────────────────────────────────────

# type00-givenLevel (1-200) 중 제외된 67개 (원본과 동일)
_GIVEN_LEVEL_EXCLUDED = frozenset([
     3,  8, 11, 23, 25, 32, 38, 44, 45, 65,
    80, 81, 82, 87, 89, 93, 94, 96,101,106,
   108,109,110,111,112,113,114,115,116,117,
   118,119,120,121,122,123,124,125,126,127,
   128,129,130,131,132,133,134,135,136,137,
   138,139,140,141,142,143,144,145,146,147,
   148,149,150,151,152,153,154,155,156,157,
   192,194,195,196,197,198,199,
])
_n_type01 = (len(_GIVEN_LEVEL_EXCLUDED) + 1) // 2   # 34
_n_type13 = len(_GIVEN_LEVEL_EXCLUDED) // 2          # 33

FILTERED_TRAIN_LEVELS = sorted(
    [i for i in range(1, 201) if i not in _GIVEN_LEVEL_EXCLUDED]
    + list(range(201, 201 + _n_type01))
    + list(range(301, 301 + _n_type13))
)  # 200개

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(current_dir, "out", "angry_birds")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "ab_agent", "checkpoints")
CONFIG_PATH = os.path.join(current_dir, "ScienceBirds", "MacOS", "config.xml")
CONFIG_BACKUP = CONFIG_PATH + ".train_backup"

CUSTOM_OBJECTS = {
    'ClassicConvStem': ClassicConvStem,
    'DuelingQHead': DuelingQHead,
    'NoisyDense': NoisyDense,
    'SpatialAttention': SpatialAttention,
}

# Action space (원본과 동일)
PHI = 10
PSI = 40
ANGLE_RESOLUTION = 20
TAP_TIME_RESOLUTION = 10
MAXIMUM_TAP_TIME = 4000


def action_to_params(action):
    a = np.unravel_index(action, (ANGLE_RESOLUTION, TAP_TIME_RESOLUTION))
    alpha = PHI + int(a[0] * (180 - PHI - PSI) / (ANGLE_RESOLUTION - 1))
    tap_time = int(a[1] / TAP_TIME_RESOLUTION * MAXIMUM_TAP_TIME)
    return alpha, tap_time


# ─────────────────────────────────────────────────────────────
# config.xml 자동 교체/복구
# ─────────────────────────────────────────────────────────────

def _read_all_level_paths(config_path):
    """config.xml에서 모든 레벨 경로를 추출 (중복 제거, 순서 유지)."""
    with open(config_path, 'r', encoding='utf-16') as f:
        content = f.read()
    paths = re.findall(r'<game_levels\s+level_path="([^"]+)"\s*/>', content)
    # 중복 제거 (학습용은 500개 × N반복)
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _write_eval_config(config_path, level_paths):
    """평가용 config.xml 생성 (레벨 목록만 1회 순차)."""
    lines = [
        '<?xml version="1.0" encoding="utf-16"?>',
        '<evaluation>',
        '    <novelty_detection_measurement step="1" measure_in_training="False" measure_in_testing="True" />',
        '    <trials>',
        '        <trial id="0" number_of_executions="1" checkpoint_time_limit="2147483647" checkpoint_interaction_limit="2147483647" notify_novelty="False">',
        '            <game_level_set mode="training" time_limit="999999" total_interaction_limit="999999" attempt_limit_per_level="1" allow_level_selection="False">',
    ]
    for path in level_paths:
        lines.append(f'                <game_levels level_path="{path}" />')
    lines += [
        '            </game_level_set>',
        '        </trial>',
        '    </trials>',
        '</evaluation>',
    ]
    with open(config_path, 'w', encoding='utf-16') as f:
        f.write('\n'.join(lines))


def setup_eval_config():
    """학습용 config.xml을 백업하고, 평가용(FILTERED 200개)으로 교체."""
    print("[config] 평가용 config.xml 생성 중...")

    # 1. 학습용 config 백업
    if not os.path.exists(CONFIG_BACKUP):
        shutil.copy2(CONFIG_PATH, CONFIG_BACKUP)
        print(f"  학습용 config 백업: {CONFIG_BACKUP}")
    else:
        print(f"  백업 이미 존재: {CONFIG_BACKUP}")

    # 2. 500개 레벨 경로 추출
    all_paths = _read_all_level_paths(CONFIG_BACKUP)
    print(f"  전체 레벨 경로: {len(all_paths)}개")

    # 3. FILTERED_TRAIN_LEVELS 인덱스로 필터링 (1-indexed → 0-indexed)
    eval_paths = []
    for level_idx in FILTERED_TRAIN_LEVELS:
        if level_idx <= len(all_paths):
            eval_paths.append(all_paths[level_idx - 1])
    print(f"  평가용 레벨: {len(eval_paths)}개 (원본 FILTERED_TRAIN_LEVELS)")

    # 4. 평가용 config 생성
    _write_eval_config(CONFIG_PATH, eval_paths)
    print(f"  config.xml 교체 완료")

    return len(eval_paths)


def restore_train_config():
    """학습용 config.xml 복구."""
    if os.path.exists(CONFIG_BACKUP):
        shutil.copy2(CONFIG_BACKUP, CONFIG_PATH)
        print(f"\n[config] 학습용 config.xml 복구 완료")
    else:
        print(f"\n[config] 백업 없음 — 복구 건너뜀")


# ─────────────────────────────────────────────────────────────
# 모델 로딩
# ─────────────────────────────────────────────────────────────

def load_saved_model(model_name):
    saved_model_path = os.path.join(OUT_DIR, model_name, "saved_model")
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(
            f"\n[오류] SavedModel을 찾을 수 없습니다.\n"
            f"경로: {saved_model_path}\n"
            f"먼저 export_savedmodel.py 를 실행하세요.\n"
        )
    print(f"  SavedModel 로드 중: {saved_model_path}")
    loaded = tf.saved_model.load(saved_model_path)
    predict_fn = loaded.signatures["serving_default"]
    print("  모델 로드 완료!")
    return predict_fn


def load_h5_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    if not os.path.exists(checkpoint_dir):
        return None
    h5_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    if not h5_files:
        return None
    ckpt_path = os.path.join(checkpoint_dir, h5_files[-1])
    print(f"  .h5 체크포인트 로드 중: {ckpt_path}")
    model = tf.keras.models.load_model(ckpt_path, custom_objects=CUSTOM_OBJECTS)
    for layer in model.layers:
        if hasattr(layer, 'set_noisy'):
            layer.set_noisy(False)
    print("  모델 로드 완료!")
    return model


def get_action_savedmodel(predict_fn, image_arr, bird_arr):
    image_t = tf.expand_dims(tf.cast(image_arr, tf.float32), 0)
    bird_t = tf.expand_dims(tf.cast(bird_arr, tf.float32), 0)
    output = predict_fn(image=image_t, bird=bird_t)
    logits = list(output.values())[0]
    return int(tf.argmax(logits[0]).numpy())


def get_action_keras(model, image_arr, bird_arr):
    image_t = tf.expand_dims(image_arr, 0)
    bird_t = tf.expand_dims(bird_arr, 0)
    q_values = model([image_t, bird_t], training=False)
    return int(tf.argmax(q_values[0]).numpy())


# ─────────────────────────────────────────────────────────────
# 평가 루프
# ─────────────────────────────────────────────────────────────

def evaluate_model(model_name=None, sim_speed=50, agent_id=AGENT_ID):
    print("=" * 60)
    print(f"  AngryBirds 모델 평가 (원본 200 레벨 순차)")
    print("=" * 60)

    # 1. 평가용 config.xml 생성
    num_eval_levels = setup_eval_config()

    # 2. 모델 로드
    use_savedmodel = False
    if model_name:
        saved_path = os.path.join(OUT_DIR, model_name, "saved_model")
        if os.path.exists(saved_path):
            print(f"\n[1/3] SavedModel 로드: {model_name}")
            predict_fn = load_saved_model(model_name)
            use_savedmodel = True
        else:
            print(f"\n[1/3] SavedModel 없음 -> .h5 체크포인트 로드")
            keras_model = load_h5_checkpoint()
            if keras_model is None:
                raise FileNotFoundError("모델을 찾을 수 없습니다.")
    else:
        print(f"\n[1/3] .h5 체크포인트 로드")
        keras_model = load_h5_checkpoint()
        if keras_model is None:
            raise FileNotFoundError("모델을 찾을 수 없습니다.")

    # 3. 환경 초기화 (평가용 config로 게임 시작)
    print(f"\n[2/3] ScienceBirds 환경 초기화...")
    env = ScienceBirdsEnv(agent_id=agent_id)
    env.ar.set_game_simulation_speed(sim_speed)
    print(f"      시뮬레이션 속도: {sim_speed}x")
    print(f"      평가 레벨 수: {num_eval_levels}개")

    # 4. 레벨 순차 평가
    print(f"\n[3/3] {num_eval_levels}개 레벨 평가 시작\n")
    print(f"{'#':>4} | {'Level':>6} | {'Score':>8} | {'Result':>6} | {'Shots':>5} | {'Time':>7}")
    print("-" * 55)

    results = []

    for i in range(num_eval_levels):
        level_result = evaluate_single_level(
            env=env,
            predict_fn=predict_fn if use_savedmodel else None,
            keras_model=None if use_savedmodel else keras_model,
            use_savedmodel=use_savedmodel,
        )
        results.append(level_result)

        status = "Pass" if level_result["passed"] else "Fail"
        print(f"{i+1:>4} | {level_result['level']:>6} | {level_result['score']:>8,} | "
              f"{status:>6} | {level_result['num_shots']:>5} | "
              f"{level_result['elapsed_sec']:>5.1f}s")

    summary = compute_summary(results)
    return results, summary


def evaluate_single_level(env, predict_fn=None, keras_model=None,
                          use_savedmodel=True):
    state = env.reset()
    level_num = env.current_level_num or 0

    start_time = time.time()
    total_score = 0
    num_shots = 0
    done = False
    level_won = False

    while not done:
        image_arr, bird_arr = state

        if use_savedmodel:
            action = get_action_savedmodel(predict_fn, image_arr * 255.0, bird_arr)
        else:
            action = get_action_keras(keras_model, image_arr, bird_arr)

        alpha, tap_ms = action_to_params(action)
        print(f"    shot#{num_shots+1}  action={action:>3}  "
              f"alpha={alpha:>3}deg  tap={tap_ms}ms")

        state, reward, done, info = env.step(action)
        total_score = info.get("score", total_score)
        if hasattr(total_score, '__len__'):
            total_score = int(total_score[0]) if len(total_score) > 0 else 0
        else:
            total_score = int(total_score)

        game_state = info.get("state")
        if game_state == GameState.WON:
            level_won = True

        num_shots += 1
        if num_shots >= 20:
            break

    elapsed = time.time() - start_time

    return {
        "level": level_num,
        "score": total_score,
        "passed": level_won,
        "num_shots": num_shots,
        "elapsed_sec": elapsed,
    }


def compute_summary(results):
    scores = [r["score"] for r in results]
    passed = [r["passed"] for r in results]
    shots = [r["num_shots"] for r in results]

    return {
        "total_levels":    len(results),
        "passed_levels":   sum(passed),
        "failed_levels":   len(results) - sum(passed),
        "win_rate":        sum(passed) / len(results) * 100,
        "avg_score":       float(np.mean(scores)),
        "max_score":       int(np.max(scores)),
        "min_score":       int(np.min(scores)),
        "std_score":       float(np.std(scores)),
        "avg_shots":       float(np.mean(shots)),
    }


# ─────────────────────────────────────────────────────────────
# 결과 출력 / 저장
# ─────────────────────────────────────────────────────────────

def print_summary(summary, model_name):
    print("\n" + "=" * 60)
    print(f"  평가 결과 요약: {model_name}")
    print("=" * 60)
    print(f"  평가 레벨 수       : {summary['total_levels']}개")
    print(f"  클리어 성공        : {summary['passed_levels']}개")
    print(f"  클리어 실패        : {summary['failed_levels']}개")
    print(f"  클리어율(Win Rate) : {summary['win_rate']:.1f}%")
    print(f"  평균 점수          : {summary['avg_score']:,.0f}점")
    print(f"  최고 점수          : {summary['max_score']:,}점")
    print(f"  최저 점수          : {summary['min_score']:,}점")
    print(f"  점수 표준편차      : {summary['std_score']:,.0f}점")
    print(f"  평균 발사 수       : {summary['avg_shots']:.1f}발/레벨")
    print("=" * 60)


def save_results_to_csv(results, model_name):
    out_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "evaluation_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EvaluationIndex", "LevelIndex", "Score", "LevelStatus", "NumShots"])
        for i, r in enumerate(results, 1):
            status = "Pass" if r["passed"] else "Fail"
            writer.writerow([i, r["level"], r["score"], status, r["num_shots"]])

    print(f"\n  CSV 저장: {csv_path}")
    return csv_path


def save_summary_to_txt(summary, model_name):
    out_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "evaluation_summary.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"모델: {model_name}\n")
        f.write("=" * 40 + "\n")
        for key, val in summary.items():
            if isinstance(val, float):
                f.write(f"{key}: {val:.2f}\n")
            else:
                f.write(f"{key}: {val}\n")

    print(f"  요약 저장: {txt_path}")
    return txt_path


def plot_score_distribution(results, summary, model_name):
    out_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    levels = [r["level"] for r in results]
    scores = [r["score"] for r in results]
    colors = ["#2ecc71" if r["passed"] else "#e74c3c" for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"AngryBirds 평가 결과 - {model_name}", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.bar(range(len(levels)), scores, color=colors, alpha=0.85)
    ax1.axhline(y=summary["avg_score"], color="navy", linestyle="--", linewidth=1.5,
                label=f"평균: {summary['avg_score']:,.0f}점")
    ax1.set_xlabel("평가 순서")
    ax1.set_ylabel("점수")
    ax1.set_title(f"레벨별 점수 (Win Rate: {summary['win_rate']:.1f}%  |  "
                  f"평균 점수: {summary['avg_score']:,.0f}점  |  "
                  f"평균 발사: {summary['avg_shots']:.1f}발)")
    ax1.set_xticks(range(0, len(levels), max(1, len(levels) // 20)))
    pass_patch = mpatches.Patch(color="#2ecc71", label=f"Pass ({summary['passed_levels']}개)")
    fail_patch = mpatches.Patch(color="#e74c3c", label=f"Fail ({summary['failed_levels']}개)")
    ax1.legend(handles=[pass_patch, fail_patch, ax1.lines[0]], loc="upper right")

    ax2 = axes[1]
    ax2.hist(scores, bins=min(20, len(scores)), color="#3498db", edgecolor="white", alpha=0.8)
    ax2.axvline(x=summary["avg_score"], color="navy", linestyle="--", linewidth=1.5,
                label=f"평균: {summary['avg_score']:,.0f}점")
    ax2.set_xlabel("점수")
    ax2.set_ylabel("레벨 수")
    ax2.set_title("점수 분포 히스토그램")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "score_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  그래프 저장: {plot_path}")


# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        results, summary = evaluate_model(
            model_name=MODEL_NAME,
            sim_speed=SIM_SPEED,
        )

        print_summary(summary, MODEL_NAME)
        save_results_to_csv(results, MODEL_NAME)
        save_summary_to_txt(summary, MODEL_NAME)
        plot_score_distribution(results, summary, MODEL_NAME)

        print("\n평가 완료!")

    finally:
        # 에러가 나더라도 반드시 학습용 config 복구
        restore_train_config()
