import numpy as np
import random


class NStepBuffer:
    """
    N-step Returns 버퍼 (AIBirds원본 n_step 파라미터와 동일).
    n개의 transition을 모아 n-step return을 계산한 뒤 PER에 저장.
    """
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf = []   # list of (state, action, reward, next_state, done)

    def add(self, state, action, reward, next_state, done):
        """
        transition 추가. 준비된 n-step transition 목록을 반환.
        done=True이면 남은 전이를 모두 flush하여 반환.
        """
        self.buf.append((state, action, reward, next_state, done))
        ready = []

        if done:
            # 에피소드 종료 → 버퍼에 남은 모든 transition 처리
            while self.buf:
                ready.append(self._compute(0))
                self.buf.pop(0)
        elif len(self.buf) >= self.n:
            ready.append(self._compute(0))
            self.buf.pop(0)

        return ready

    def _compute(self, start: int):
        """start 인덱스부터 최대 n-step return 계산."""
        state  = self.buf[start][0]
        action = self.buf[start][1]

        n_step_return = 0.0
        steps = min(self.n, len(self.buf) - start)
        for i in range(steps):
            n_step_return += (self.gamma ** i) * self.buf[start + i][2]

        last = min(start + self.n - 1, len(self.buf) - 1)
        next_state = self.buf[last][3]
        done       = self.buf[last][4]

        return (state, action, n_step_return, next_state, done)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).
    AIBirds원본 ReplayMemory의 우선 순위 샘플링 방식과 동일.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity       = capacity
        self.alpha          = alpha
        self.beta           = beta
        self.beta_increment = beta_increment
        self.pos            = 0
        self.buffer         = []
        self.priorities     = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs  = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total   = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states_img  = np.array([s[0][0] for s in samples])
        states_bird = np.array([s[0][1] for s in samples])
        actions     = np.array([s[1]    for s in samples])
        rewards     = np.array([s[2]    for s in samples], dtype=np.float32)
        next_img    = np.array([s[3][0] for s in samples])
        next_bird   = np.array([s[3][1] for s in samples])
        dones       = np.array([s[4]    for s in samples], dtype=np.float32)

        return (states_img, states_bird), actions, rewards, (next_img, next_bird), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)
