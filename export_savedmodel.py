"""
.h5 체크포인트 -> SavedModel 변환 스크립트
============================================
원본 evaluate_ab.py에서 사용할 수 있는 형식으로 변환합니다.

변환 내용:
  - image 입력: 0~255 float32 -> 0~1 정규화 (원본 evaluate가 uint8->float32 cast만 하므로)
  - NoisyNet 노이즈 비활성화 (greedy 평가용)

사용법:
  # 5차원 모델 (기본 - 앞으로 학습할 모델)
  python3 export_savedmodel.py

  # 9차원 모델 (기존 학습된 모델 - bird 5->9 래핑 포함)
  python3 export_savedmodel.py --bird9

  # 특정 체크포인트 지정
  python3 export_savedmodel.py --checkpoint path/to/model.h5

  # 모델 이름 지정 (out/angry_birds/<이름>/saved_model 에 저장)
  python3 export_savedmodel.py --name ab_agent_v1
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.demo.v2_rl.model import ClassicConvStem, DuelingQHead, NoisyDense, SpatialAttention

CUSTOM_OBJECTS = {
    'ClassicConvStem': ClassicConvStem,
    'DuelingQHead': DuelingQHead,
    'NoisyDense': NoisyDense,
    'SpatialAttention': SpatialAttention,
}

MODELS_DIR = os.path.join(current_dir, "out", "angry_birds", "ab_agent", "checkpoints")


def find_latest_checkpoint(models_dir):
    if not os.path.exists(models_dir):
        return None
    h5_files = sorted(
        [f for f in os.listdir(models_dir) if f.endswith(".h5")],
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f))
    )
    return os.path.join(models_dir, h5_files[-1]) if h5_files else None


class SavedModelWrapper5(tf.Module):
    """5차원 bird 모델용 래퍼 (이미지 정규화만)"""

    def __init__(self, keras_model):
        super().__init__()
        self.model = keras_model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32, name='image'),
        tf.TensorSpec(shape=[None, 5], dtype=tf.float32, name='bird'),
    ])
    def __call__(self, image, bird):
        image_normalized = image / 255.0
        q_values = self.model([image_normalized, bird], training=False)
        return q_values


class SavedModelWrapper9(tf.Module):
    """9차원 bird 모델용 래퍼 (이미지 정규화 + bird 5->9 변환)"""

    def __init__(self, keras_model):
        super().__init__()
        self.model = keras_model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32, name='image'),
        tf.TensorSpec(shape=[None, 5], dtype=tf.float32, name='bird'),
    ])
    def __call__(self, image, bird):
        image_normalized = image / 255.0

        # Bird 5차원 -> 9차원 변환
        # [0:5] = one-hot (그대로)
        # [5] = is_explosive = bird[3] (Black)
        # [6] = splits = bird[1] * 3.0 (Blue)
        # [7] = has_projectile = bird[4] (White)
        # [8] = speed_multiplier = bird[2] * 1.5 (Yellow)
        skill = tf.stack([
            bird[:, 3],           # is_explosive
            bird[:, 1] * 3.0,    # splits
            bird[:, 4],           # has_projectile
            bird[:, 2] * 1.5,    # speed_multiplier
        ], axis=1)
        bird_9 = tf.concat([bird, skill], axis=1)

        q_values = self.model([image_normalized, bird_9], training=False)
        return q_values


def disable_noise(model):
    """NoisyNet 노이즈 비활성화 (greedy 평가용)"""
    for layer in model.layers:
        if hasattr(layer, 'set_noisy'):
            layer.set_noisy(False)


def main():
    parser = argparse.ArgumentParser(description='.h5 -> SavedModel 변환')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='체크포인트 경로 (None이면 최신 자동 선택)')
    parser.add_argument('--name', type=str, default='ab_agent_v1',
                        help='모델 이름 (저장 경로: out/angry_birds/<name>/saved_model)')
    parser.add_argument('--bird9', action='store_true',
                        help='9차원 bird 모델용 래퍼 사용 (기존 모델)')
    args = parser.parse_args()

    # 체크포인트 탐색
    ckpt_path = args.checkpoint or find_latest_checkpoint(MODELS_DIR)
    if ckpt_path is None:
        print(f"[오류] 체크포인트를 찾을 수 없습니다: {MODELS_DIR}")
        return

    output_dir = os.path.join(current_dir, "out", "angry_birds", args.name, "saved_model")
    bird_dim = 9 if args.bird9 else 5

    print("=" * 60)
    print(f"  .h5 -> SavedModel 변환")
    print("=" * 60)
    print(f"  체크포인트 : {ckpt_path}")
    print(f"  Bird 차원  : {bird_dim} ({'래퍼로 5->9 변환' if args.bird9 else '원본 호환 5차원'})")
    print(f"  저장 경로  : {output_dir}")
    print()

    # 1. 모델 로드
    print("[1/4] 체크포인트 로드 중...")
    model = tf.keras.models.load_model(ckpt_path, custom_objects=CUSTOM_OBJECTS)
    print(f"  모델 입력: {[inp.shape for inp in model.inputs]}")
    print(f"  모델 출력: {model.output.shape}")

    # 2. NoisyNet 비활성화
    print("[2/4] NoisyNet 노이즈 비활성화...")
    disable_noise(model)

    # 3. 래퍼 생성
    print(f"[3/4] SavedModel 래퍼 생성 (bird {'5->9 변환' if args.bird9 else '5차원 직접'})")
    if args.bird9:
        wrapper = SavedModelWrapper9(model)
    else:
        wrapper = SavedModelWrapper5(model)

    # 테스트 실행
    test_image = tf.constant(np.random.randint(0, 256, (1, 128, 128, 3), dtype=np.uint8),
                             dtype=tf.float32)
    test_bird = tf.constant([[1.0, 0, 0, 0, 0]], dtype=tf.float32)  # Red bird
    result = wrapper(test_image, test_bird)
    print(f"  테스트 출력 shape: {result.shape}")
    print(f"  테스트 argmax action: {tf.argmax(result[0]).numpy()}")

    # 4. SavedModel 저장
    print(f"[4/4] SavedModel 저장...")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    tf.saved_model.save(
        wrapper,
        output_dir,
        signatures={
            'serving_default': wrapper.__call__.get_concrete_function(
                tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32, name='image'),
                tf.TensorSpec(shape=[None, 5], dtype=tf.float32, name='bird'),
            )
        }
    )

    # 검증
    print("\n[검증] SavedModel 다시 로드...")
    loaded = tf.saved_model.load(output_dir)
    predict_fn = loaded.signatures['serving_default']
    out = predict_fn(image=test_image, bird=test_bird)
    logits = list(out.values())[0]
    print(f"  출력 keys: {list(out.keys())}")
    print(f"  출력 shape: {logits.shape}")
    print(f"  argmax action: {tf.argmax(logits[0]).numpy()}")

    print(f"\n{'=' * 60}")
    print(f"  변환 완료!")
    print(f"  SavedModel: {output_dir}")
    print(f"\n  원본 evaluate_ab.py에서 사용:")
    print(f'  MODEL_NAME = "{args.name}"')
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
