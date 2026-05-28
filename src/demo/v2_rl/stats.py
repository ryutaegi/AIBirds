"""
AIBirds원본 Statistics 클래스 포팅 (독립 구현, 외부 의존성 없음)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless (no display required)

# ★ macOS 한글 폰트 설정
matplotlib.rcParams['font.sans-serif'] = ['AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

import matplotlib.pyplot as plt
import os
import pickle
import time


def sec2hhmmss(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def get_moving_avg_lst(values, n):
    if len(values) < n or n == 0:
        return np.array([])
    a = np.cumsum(values, dtype=float)
    a[n:] = a[n:] - a[:-n]
    a = a[n - 1:] / n
    mov_avg = np.full(len(values), np.nan)
    mov_avg[n // 2:n // 2 + len(a)] = a
    return mov_avg


def get_moving_avg_val(values, window_size):
    if len(values) >= window_size:
        return float(np.average(values[-window_size:]))
    return float('nan')


def get_moving_quantile(x, y, window_size, quantile):
    if len(y) < window_size or window_size == 0:
        return np.array([]), np.array([])
    overhang = len(y) % window_size
    if overhang:
        y, x = y[:-overhang], x[:-overhang]
    y_q = np.array([np.quantile(y[i:i+window_size], 1 - quantile)
                    for i in range(0, len(y), window_size)])
    x_q = np.array([np.mean(x[i:i+window_size])
                    for i in range(0, len(x), window_size)])
    return x_q, y_q


def _finalize_plot(title, x_label, y_label, out_path, legend=True, logarithmic=False):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if logarithmic:
        plt.yscale('log')
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


class Statistics:
    WINDOW_SIZE_EPISODES = 500
    WINDOW_SIZE_CYCLES = 10

    # episode_stats column indices
    TRANSITION   = 0
    SECONDS      = 1
    RETURN       = 2
    SCORE        = 3
    TIME         = 4   # shots per episode
    WIN          = 5
    RETURN_RECORD = 6
    SCORE_RECORD  = 7

    # cycle_stats column indices
    LOSS          = 2
    LEARNING_RATE = 3

    def __init__(self):
        self.episode_stats = np.zeros((100000, 8), dtype='float32')
        self.cycle_stats   = np.zeros((500000, 4), dtype='float32')
        self.episode_ptr   = 0
        self.cycle_ptr     = 0
        self._total_timer  = 0.0
        self._comp_timer   = 0.0
        self._timer_started = False

    def start_timer(self):
        self._total_timer = time.time() - self._total_timer
        self._comp_timer  = time.time()
        self._timer_started = True

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def denote_episode_stats(self, ret, score, shots, win):
        if self.episode_ptr >= len(self.episode_stats):
            self.episode_stats = np.vstack(
                [self.episode_stats, np.zeros((100000, 8), dtype='float32')])

        trans   = int(self.get_current_transition()) + shots
        seconds = time.time() - self._total_timer

        prev_ret_rec   = self.episode_stats[self.episode_ptr - 1, self.RETURN_RECORD]   if self.episode_ptr > 0 else -1e9
        prev_score_rec = self.episode_stats[self.episode_ptr - 1, self.SCORE_RECORD]    if self.episode_ptr > 0 else 0

        self.episode_stats[self.episode_ptr] = [
            trans, seconds, ret, score, shots, float(win),
            max(prev_ret_rec, ret), max(prev_score_rec, score)
        ]
        self.episode_ptr += 1

    def denote_learning_stats(self, loss, learning_rate):
        if self.cycle_ptr >= len(self.cycle_stats):
            self.cycle_stats = np.vstack(
                [self.cycle_stats, np.zeros((500000, 4), dtype='float32')])

        trans   = self.get_current_transition()
        seconds = time.time() - self._total_timer
        self.cycle_stats[self.cycle_ptr] = [trans, seconds, loss, learning_rate]
        self.cycle_ptr += 1

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def get_num_episodes(self):    return self.episode_ptr
    def get_num_cycles(self):      return self.cycle_ptr

    def get_current_transition(self):
        if self.episode_ptr == 0:
            return 0
        return int(self.episode_stats[self.episode_ptr - 1, self.TRANSITION])

    def get_returns(self):         return self.episode_stats[:self.episode_ptr, self.RETURN]
    def get_scores(self):          return self.episode_stats[:self.episode_ptr, self.SCORE].astype(int)
    def get_times(self):           return self.episode_stats[:self.episode_ptr, self.TIME].astype(int)
    def get_wins(self):            return self.episode_stats[:self.episode_ptr, self.WIN].astype(bool)
    def get_return_records(self):  return self.episode_stats[:self.episode_ptr, self.RETURN_RECORD]
    def get_score_records(self):   return self.episode_stats[:self.episode_ptr, self.SCORE_RECORD].astype(int)
    def get_losses(self):          return self.cycle_stats[:self.cycle_ptr, self.LOSS]
    def get_learning_rates(self):  return self.cycle_stats[:self.cycle_ptr, self.LEARNING_RATE]

    def get_records(self):
        if self.episode_ptr > 0:
            return (float(self.episode_stats[self.episode_ptr - 1, self.RETURN_RECORD]),
                    int(self.episode_stats[self.episode_ptr - 1, self.SCORE_RECORD]))
        return float('nan'), 0

    # ------------------------------------------------------------------
    # Console output (AIBirds원본 print_stats 포맷)
    # ------------------------------------------------------------------
    def print_stats(self, step, total_steps, epsilon):
        comp_time  = time.time() - self._comp_timer
        total_time = time.time() - self._total_timer

        ma_ws    = self.WINDOW_SIZE_EPISODES
        ma_score = get_moving_avg_val(self.get_scores(), ma_ws)
        ma_wins  = get_moving_avg_val(self.get_wins().astype(float), ma_ws)
        ma_loss  = get_moving_avg_val(self.get_losses(), self.WINDOW_SIZE_CYCLES)
        _, score_rec = self.get_records()
        lr = float(self.get_learning_rates()[-1]) if self.cycle_ptr > 0 else float('nan')

        pct    = step / total_steps * 100
        filled = int(30 * step // total_steps)
        bar    = '█' * filled + '░' * (30 - filled)
        eta    = sec2hhmmss(total_time / step * (total_steps - step)) if step > 0 else '--:--:--'

        print(
            f"\n진행률: [{bar}] {pct:.1f}%  (스텝 {step} / {total_steps}  |  ETA {eta})"
            f"\n   # Ep. | Trans.:              {self.episode_ptr} | {self.get_current_transition()}"
            f"\n   Epsilon:                   {epsilon:.3f}"
            f"\n   Score ({ma_ws} MA | record):  {ma_score:.1f} | {score_rec}"
            f"\n   Win-ratio ({ma_ws} MA):        {ma_wins:.3f}"
            f"\n   Loss ({self.WINDOW_SIZE_CYCLES} MA):              {ma_loss:.4f}"
            f"\n   Learning rate:             {lr:.6f}"
            f"\n------"
            f"\n   Comp time (last period):   {int(comp_time)} s"
            f"\n   Total time:                {sec2hhmmss(total_time)}"
        )
        self._comp_timer = time.time()

    # ------------------------------------------------------------------
    # Plotting (AIBirds원본 plot_stats 포맷)
    # ------------------------------------------------------------------
    def plot_stats(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        self._plot_episode_stat(self.get_scores(), self.get_score_records(),
                                "Score history", "Score", out_path + "scores.png")
        self._plot_episode_stat(self.get_returns(), self.get_return_records(),
                                "Return history", "Return", out_path + "returns.png")
        self._plot_win_ratio(out_path + "wins.png")
        self._plot_loss(out_path + "loss.png")
        self._plot_score_dist(out_path + "score_dist.png")

    def _plot_episode_stat(self, y, y_rec, title, y_label, out_path):
        if len(y) < 2:
            return
        ma_ws = max(1, len(y) // 40)
        y_ma  = get_moving_avg_lst(y, ma_ws)
        x     = np.arange(len(y_ma))
        x_up, y_up = get_moving_quantile(x, y, ma_ws, quantile=0.95)
        x_lo, y_lo = get_moving_quantile(x, y, ma_ws, quantile=0.05)

        plt.figure()
        plt.plot(x, y_ma, label=f"Moving avg (ws={ma_ws})")
        if len(x_up) > 0:
            plt.fill_between(x_up, y_lo, y_up, alpha=0.3, label="p5|p95")
        plt.plot(y_rec, label="Record", linestyle='--')
        _finalize_plot(title, "Episode", y_label, out_path)

    def _plot_win_ratio(self, out_path):
        wins = self.get_wins().astype(float)
        if len(wins) < 2:
            return
        ma_ws = max(1, len(wins) // 40)
        y_ma  = get_moving_avg_lst(wins, ma_ws)
        plt.figure()
        plt.plot(y_ma, label=f"Win ratio (ws={ma_ws})")
        plt.ylim(0, 1)
        _finalize_plot("Win ratio history", "Episode", "Win proportion", out_path)

    def _plot_loss(self, out_path):
        losses = self.get_losses()
        if len(losses) < 2:
            return
        ma_ws = max(1, len(losses) // 40)
        y_ma  = get_moving_avg_lst(losses, ma_ws)
        plt.figure()
        plt.plot(y_ma, label=f"Loss MA (ws={ma_ws})")
        _finalize_plot("Training loss history", "Train cycle", "Loss",
                       out_path, logarithmic=True)

    def _plot_score_dist(self, out_path):
        scores = self.get_scores()[-1000:]
        if len(scores) < 10:
            return
        plt.figure()
        plt.hist(scores, bins=30)
        _finalize_plot("Recent score distribution", "Score", "Count",
                       out_path, legend=False)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(path + "stats.pkl", 'wb') as f:
            pickle.dump({
                'episode_stats': self.episode_stats,
                'cycle_stats':   self.cycle_stats,
                'episode_ptr':   self.episode_ptr,
                'cycle_ptr':     self.cycle_ptr,
            }, f)

    def load(self, path):
        fpath = path + "stats.pkl"
        if not os.path.exists(fpath):
            print(f"[stats] No saved stats at {fpath}")
            return
        with open(fpath, 'rb') as f:
            d = pickle.load(f)
        self.episode_stats = d['episode_stats']
        self.cycle_stats   = d['cycle_stats']
        self.episode_ptr   = d['episode_ptr']
        self.cycle_ptr     = d['cycle_ptr']
        if self.episode_ptr > 0:
            self._total_timer = float(self.episode_stats[self.episode_ptr - 1, self.SECONDS])
        print(f"[stats] Loaded {self.episode_ptr} ep, {self.cycle_ptr} cycles from {fpath}")
