import numpy as np
import tensorflow as tf
from .model import build_model
from .memory import PrioritizedReplayBuffer, NStepBuffer


class DQNAgent:
    """
    Double DQN + Dueling + PER + N-step Returns
    + Exponential Epsilon Decay + Exponential LR Decay
    + Data Augmentation (일반화 향상)
    (AIBirds원본 기법 전체 적용)
    """

    def __init__(self,
                 num_actions,
                 # Learning rate & decay (AIBirds원본 ParamScheduler exp, half_life=100000)
                 learning_rate=0.0003,
                 lr_half_life=100000,
                 # Discount
                 gamma=0.9,
                 # Epsilon (exp decay, half_life=25000, min=0.1)
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 eps_half_life=25000,
                 # N-step returns (AIBirds원본 n_step)
                 n_step=3,
                 # Noisy Nets
                 noise_std_init=0.5,
                 # Replay memory
                 memory_capacity=30000):

        self.num_actions   = num_actions
        self.gamma         = gamma
        self.n_step        = n_step
        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.eps_half_life = eps_half_life
        self.step_count    = 0

        self.model        = build_model(num_actions, noise_std_init=noise_std_init)
        self.target_model = build_model(num_actions, noise_std_init=noise_std_init)
        self.target_model.set_weights(self.model.get_weights())

        # 지수 감소 Learning Rate: lr = lr_init * 0.5^(step / half_life)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=max(learning_rate, 0.0002),  # 최소 0.0002
            decay_steps=lr_half_life,
            decay_rate=0.5,
            staircase=False,
        )
        try:
            self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
            print("Using legacy Adam optimizer (Apple Silicon optimized)")
        except AttributeError:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            print("Using standard Adam optimizer")

        self.memory    = PrioritizedReplayBuffer(capacity=memory_capacity)
        self._nstep    = NStepBuffer(n_step, gamma)

        # @tf.function으로 학습 루프 컴파일 (Metal GPU 활용 극대화)
        self._train_step = tf.function(self._train_step_impl)

    # ------------------------------------------------------------------
    # 데이터 증강 (일반화 향상 — 처음 보는 맵 대응)
    # ------------------------------------------------------------------
    def _augment_batch(self, images):
        """학습 시 이미지 증강: CNN이 픽셀값이 아닌 형태/구조를 학습하도록 유도"""
        batch_size = images.shape[0]

        # 1) 랜덤 밝기 (±15%)
        brightness = np.random.uniform(0.85, 1.15, size=(batch_size, 1, 1, 1))
        images = images * brightness

        # 2) 랜덤 대비 (±15%): 평균을 기준으로 스케일
        means = np.mean(images, axis=(1, 2, 3), keepdims=True)
        contrast = np.random.uniform(0.85, 1.15, size=(batch_size, 1, 1, 1))
        images = (images - means) * contrast + means

        # 3) 가우시안 노이즈 (std=0.01)
        noise = np.random.normal(0, 0.01, size=images.shape)
        images = images + noise

        # 4) 랜덤 색상 채널 이동 (각 채널 ±5%)
        color_shift = np.random.uniform(0.95, 1.05, size=(batch_size, 1, 1, 3))
        images = images * color_shift

        return np.clip(images, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 전이 저장 — N-step 버퍼를 거쳐 PER에 추가
    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """에피소드 루프에서 이 메서드를 사용하세요 (memory.add 대신)."""
        transitions = self._nstep.add(state, action, reward, next_state, done)
        for t in transitions:
            self.memory.add(*t)

    # ------------------------------------------------------------------
    # 행동 선택 (epsilon-greedy)
    # ------------------------------------------------------------------
    def get_action(self, state, training=True, action_mask=None):
        if training and np.random.rand() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                return np.random.choice(valid_actions)
            return np.random.randint(self.num_actions)

        try:
            image, bird_info = state
        except (TypeError, ValueError) as e:
            print(f"  [에러] state 언팩 실패: {e}")
            raise

        image     = np.expand_dims(image,     axis=0)
        bird_info = np.expand_dims(bird_info, axis=0)

        # 노이즈는 noise_active 플래그로 제어 (training 플래그와 무관)
        # 학습 시: 노이즈 ON (탐험), 평가 시: set_noisy(False)로 끔
        q_values = self.model([image, bird_info], training=False)[0].numpy()

        if action_mask is not None:
            q_values[action_mask == 0] = -np.inf

        return np.argmax(q_values)

    def set_noisy(self, active: bool):
        """모든 NoisyDense 레이어의 노이즈 ON/OFF (평가 시 False로 설정)."""
        for layer in self.model.layers:
            if hasattr(layer, 'set_noisy'):
                layer.set_noisy(active)

    # ------------------------------------------------------------------
    # 학습 (Double DQN + PER + N-step gamma)
    # ------------------------------------------------------------------
    def _train_step_impl(self, s_img, s_bird, a, r, ns_img, ns_bird, d, weights):
        """@tf.function으로 컴파일되는 내부 학습 스텝."""
        gamma_n = tf.cast(self.gamma ** self.n_step, tf.float32)

        # Double DQN: 온라인 네트워크로 액션 선택
        next_q_online = self.model([ns_img, ns_bird], training=False)
        next_actions  = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

        # 타겟 네트워크로 가치 평가
        next_q_target = self.target_model([ns_img, ns_bird], training=False)
        batch_idx     = tf.range(tf.shape(a)[0])
        next_q_sel    = tf.gather_nd(next_q_target,
                                     tf.stack([batch_idx, next_actions], axis=1))
        target_q      = r + (1.0 - d) * gamma_n * next_q_sel

        with tf.GradientTape() as tape:
            current_q  = self.model([s_img, s_bird], training=True)
            selected_q = tf.reduce_sum(
                current_q * tf.one_hot(a, self.num_actions), axis=1
            )
            td_errors  = tf.abs(target_q - selected_q)
            # Huber loss (delta=1.0): 큰 TD error에 강건 (MSE 대신)
            loss = tf.reduce_mean(weights * tf.keras.losses.huber(
                target_q, selected_q, delta=1.0))

        grads = tape.gradient(loss, self.model.trainable_variables)
        # Gradient clipping: max_norm=10으로 gradient 폭발 방지
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, td_errors

    def _reset_all_noise(self):
        """모든 NoisyDense 레이어의 노이즈를 리셋."""
        for layer in self.model.layers:
            if hasattr(layer, 'reset_noise'):
                layer.reset_noise()
        for layer in self.target_model.layers:
            if hasattr(layer, 'reset_noise'):
                layer.reset_noise()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        # 매 학습 스텝마다 노이즈 리셋 (원본과 동일)
        self._reset_all_noise()

        (s_img, s_bird), a, r, (ns_img, ns_bird), d, indices, weights = \
            self.memory.sample(batch_size)

        # 데이터 증강 (현재 상태 & 다음 상태 독립적으로 증강)
        s_img  = self._augment_batch(s_img)
        ns_img = self._augment_batch(ns_img)

        # numpy → tensor 변환
        s_img   = tf.constant(s_img,   dtype=tf.float32)
        s_bird  = tf.constant(s_bird,  dtype=tf.float32)
        a       = tf.constant(a,       dtype=tf.int32)
        r       = tf.constant(r,       dtype=tf.float32)
        ns_img  = tf.constant(ns_img,  dtype=tf.float32)
        ns_bird = tf.constant(ns_bird, dtype=tf.float32)
        d       = tf.constant(d,       dtype=tf.float32)
        weights = tf.constant(weights, dtype=tf.float32)

        loss, td_errors = self._train_step(s_img, s_bird, a, r, ns_img, ns_bird, d, weights)

        # 우선순위 업데이트
        self.memory.update_priorities(indices, td_errors.numpy() + 1e-6)

        # 지수 감소 Epsilon: ε = max(ε_min, ε_start * 0.5^(step / half_life))
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start * (0.5 ** (self.step_count / self.eps_half_life)),
        )
        self.step_count += 1

        return loss.numpy()

    # ------------------------------------------------------------------
    # 타겟 네트워크 동기화
    # ------------------------------------------------------------------
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
