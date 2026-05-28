import numpy as np
import cv2
import sys
import os
import time
import socket
import random
import subprocess
import shutil
import signal

# Adjust path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import re

try:
    from client.agent_client import AgentClient, GameState
    from computer_vision.GroundTruthReader import GroundTruthReader
except ImportError:
    from src.client.agent_client import AgentClient, GameState
    from src.computer_vision.GroundTruthReader import GroundTruthReader

# XML bird type string → one-hot index (BirdRed=0, Blue=1, Yellow=2, Black=3, White=4)
XML_BIRD_TYPE_IDX = {
    'BirdRed': 0, 'BirdBlue': 1, 'BirdYellow': 2, 'BirdBlack': 3, 'BirdWhite': 4
}

# Constants from AIBirds원본
PHI = 10
PSI = 40
ANGLE_RESOLUTION = 20
TAP_TIME_RESOLUTION = 10
MAXIMUM_TAP_TIME = 4000

def _find_java():
    """Java 실행 파일 경로를 찾습니다."""
    java = shutil.which("java")
    if java:
        return java
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home:
        candidate = os.path.join(java_home, "bin", "java")
        if os.path.isfile(candidate):
            return candidate
    return None


def _kill_process_by_name(name_fragment):
    """프로세스 이름에 name_fragment가 포함된 프로세스를 모두 kill."""
    try:
        if sys.platform == "darwin":
            # macOS: pkill로 프로세스 종료
            subprocess.run(["pkill", "-f", name_fragment],
                           capture_output=True, timeout=5)
        else:
            subprocess.run(["pkill", "-f", name_fragment],
                           capture_output=True, timeout=5)
    except Exception:
        pass


class ScienceBirdsEnv:
    def __init__(self, agent_id=2888, auto_launch=True):
        self.agent_id = agent_id
        self.auto_launch = auto_launch
        self.java_process = None
        self.game_process = None

        self.model_cv = np.loadtxt(os.path.join(project_root, "model"), delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open(os.path.join(project_root, 'target_class')).readlines()))

        # ★ Action space: angle × tap_time = 20×10 = 200 (원본과 동일)
        self.num_actions = ANGLE_RESOLUTION * TAP_TIME_RESOLUTION

        self.max_shots_per_episode = 10
        self.shots_count = 0
        self.prev_score = 0
        self.current_level_birds = []
        self.current_level_num = None
        self.level_paths = self._load_level_paths()
        self.frozen_levels = set()  # 멈춤 발생한 레벨 블랙리스트

        if auto_launch:
            self._start_game()

        # 소켓 연결
        self.ar = AgentClient("127.0.0.1", "2004")
        self._connect_with_retry()
        self.ar.configure(agent_id)
        self.game_speed = 50
        self.ar.set_game_simulation_speed(self.game_speed)

    def _start_game(self):
        """Java 서버 + Science Birds를 시작합니다."""
        # Java 서버 시작
        java_exe = _find_java()
        if java_exe is None:
            print("[경고] Java를 찾을 수 없습니다. 수동으로 시작하세요.")
            return

        framework_dir = os.path.join(project_root, "ScienceBirds", "MacOS")
        jar_path = os.path.join(framework_dir, "game_playing_interface.jar")
        if not os.path.isfile(jar_path):
            print(f"[경고] jar 파일 없음: {jar_path}")
            return

        print("[시작] Java 서버 시작 중...")
        log_file = open(os.path.join(framework_dir, "java_server.log"), "w")
        self.java_process = subprocess.Popen(
            [java_exe, "-jar", "game_playing_interface.jar"],
            stdout=log_file, stderr=log_file,
            cwd=framework_dir,
        )
        print(f"  Java PID: {self.java_process.pid}")

        # Science Birds 시작 (macOS)
        if sys.platform == "darwin":
            app_path = os.path.join(framework_dir, "Science Birds.app")
            if os.path.exists(app_path):
                print("[시작] Science Birds 시작 중...")
                self.game_process = subprocess.Popen(
                    ["open", "-a", app_path, "--args", "-batchmode"],
                    cwd=framework_dir,
                )
                print(f"  Science Birds 시작됨")
            else:
                # .app이 없으면 실행파일 직접 시도
                exe_path = os.path.join(framework_dir, "Science Birds")
                if os.path.isfile(exe_path):
                    self.game_process = subprocess.Popen(
                        [exe_path], cwd=framework_dir,
                    )

        # 서버 준비 대기
        print("[대기] 서버 연결 대기 중 (최대 30초)...")
        time.sleep(5)

        # macOS: Science Birds 창 자동 최소화
        if sys.platform == "darwin":
            self._minimize_game_window()

    def _minimize_game_window(self):
        """macOS: Science Birds 창을 최소화합니다."""
        try:
            # 창이 뜰 때까지 잠시 대기
            time.sleep(2)
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to set miniaturized of '
                '(every window of every process whose name contains "Science Birds") to true'
            ], capture_output=True, timeout=5)
            print("[최소화] Science Birds 창 최소화 완료")
        except Exception as e:
            print(f"[최소화] 실패 (무시): {e}")

    def _connect_with_retry(self, timeout=60):
        """소켓 연결을 재시도합니다."""
        deadline = time.time() + timeout
        while True:
            try:
                self.ar.connect_to_server()
                return
            except socket.error:
                if time.time() >= deadline:
                    raise RuntimeError(f"서버 연결 실패 ({timeout}초 초과)")
                time.sleep(2)
                try:
                    self.ar.server_socket.close()
                except OSError:
                    pass
                self.ar.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _kill_game(self):
        """Java + Science Birds 프로세스를 종료합니다."""
        print("[종료] 게임 프로세스 종료 중...")

        # Java 서버 종료
        if self.java_process is not None:
            try:
                self.java_process.kill()
                self.java_process.wait(timeout=10)
            except Exception:
                pass
            self.java_process = None

        # Science Birds 종료
        if self.game_process is not None:
            try:
                self.game_process.kill()
                self.game_process.wait(timeout=10)
            except Exception:
                pass
            self.game_process = None

        # 혹시 남아있는 프로세스 정리
        _kill_process_by_name("game_playing_interface.jar")
        _kill_process_by_name("Science Birds")

        # 소켓 닫기
        try:
            self.ar.server_socket.close()
        except Exception:
            pass

        time.sleep(3)
        print("[종료] 완료")

    def restart_game(self):
        """게임 전체를 재시작합니다 (학습 데이터는 유지)."""
        # 멈춘 레벨을 블랙리스트에 추가
        if self.current_level_num is not None:
            actual_level = ((self.current_level_num - 1) % 500) + 1  # 1~500 범위로 변환
            self.frozen_levels.add(actual_level)
            print(f"\n[재시작] 레벨 {self.current_level_num} (실제: {actual_level}) 블랙리스트 추가")
            print(f"  블랙리스트: {sorted(self.frozen_levels)}")
        print("[재시작] Unity 멈춤 감지 → Java+게임 재시작...")
        self._kill_game()

        if self.auto_launch:
            self._start_game()

        # 소켓 재연결
        self.ar = AgentClient("127.0.0.1", "2004")
        self._connect_with_retry()
        self.ar.configure(self.agent_id)
        self.ar.set_game_simulation_speed(self.game_speed)
        self.shots_count = 0
        self.prev_score = 0
        print("[재시작] 완료\n")

    def reset(self):
        """환경 리셋 (타임아웃 시 빠르게 실패 → restart_game으로 복구)"""
        #print("Resetting environment...")
        self.shots_count = 0
        self.prev_score = 0

        # ★ Level 21+ 로드 직전에 이전 상태 명시적 정리
        if self.current_level_num and self.current_level_num >= 20:
            #print(f"  [정리] Level {self.current_level_num} → 다음 레벨 로드 전 상태 정리...")
            try:
                self.ar.server_socket.settimeout(5)
                # pending 상태 처리
                self.ar.get_game_state()
                #print(f"  [정리] ✓ 상태 정리 완료")
            except (socket.timeout, OSError, ValueError):
                print(f"  [정리] ⚠️ 상태 정리 중 타임아웃/에러 (무시)")
                try:
                    self.ar._clear_buffer()
                except Exception:
                    pass
            finally:
                self.ar.server_socket.settimeout(None)

        try:
            # 전체 reset을 더 길게 설정 (Level 21+ 로드 시 오래 걸림)
            # 개별 호출: 20초 타임아웃 (복잡한 레벨 로드 대비)
            self.ar.server_socket.settimeout(20)

            # 1단계: 게임 준비 상태(competition 프레임워크 핸드셰이크)
            reset_deadline = time.time() + 20
            for _ in range(10):
                if time.time() > reset_deadline:
                    print(f"  [경고] reset() 타임아웃 (20초 초과)")
                    raise socket.timeout("reset() 타임아웃")

                try:
                    state = self.ar.get_game_state()
                except (socket.timeout, OSError, ValueError):
                    try:
                        self.ar._clear_buffer()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    continue

                if state == GameState.EVALUATION_TERMINATED:
                    #print("[순환] 평가 종료 → 레벨 자동 순환...")
                    try:
                        self.ar.ready_for_new_set()
                        #print("  ✓ 레벨 순환 완료")
                    except Exception as e:
                        print(f"  ⚠️ 순환 실패: {e}")
                    time.sleep(0.15)
                    continue
                if state in [GameState.NEWTRAININGSET, GameState.NEWTESTSET,
                             GameState.NEWTRIAL, GameState.RESUMETRAINING]:
                    print(f"  Handling {state}...")
                    try:
                        self.ar.ready_for_new_set()
                    except (socket.timeout, OSError):
                        pass
                    time.sleep(0.15)
                    continue
                if state == GameState.REQUESTNOVELTYLIKELIHOOD:
                    try:
                        self.ar.report_novelty_likelihood(0.0, 1.0, [], 0, "")
                    except (socket.timeout, OSError):
                        pass
                    time.sleep(0.15)
                    continue
                # 로딩 가능한 상태 또는 이미 PLAYING → 다음 단계로
                break

            # 2단계: 레벨 로딩 — 서버가 순서대로 레벨을 제공 (플레이 후 자동 진행)
            try:
                state = self.ar.get_game_state()
            except (socket.timeout, OSError):
                print(f"  [경고] reset() 단계 2 get_game_state() 타임아웃")
                state = GameState.PLAYING

            if state != GameState.PLAYING:
                try:
                    # ★ 100,000개 레벨이 config.xml에 미리 생성됨
                    # → load_next_available_level() 호출로 순차 로드 (재연결 불필요)
                    level_num = self.ar.load_next_available_level()

                    # ★ 블랙리스트 레벨이면 건너뛰기 (최대 10회)
                    skip_count = 0
                    while self.frozen_levels and skip_count < 10:
                        actual = ((level_num - 1) % 500) + 1
                        if actual not in self.frozen_levels:
                            break
                        print(f"  [블랙리스트] 레벨 {level_num} (실제: {actual}) 건너뜀")
                        level_num = self.ar.load_next_available_level()
                        skip_count += 1

                    self.current_level_num = level_num  # 현재 레벨 번호 저장
                    self.current_level_birds = self._parse_level_birds(level_num)

                    #print(f"  Level: {level_num}, Birds: {self.current_level_birds}")
                except (socket.timeout, OSError):
                    print(f"  [경고] load_next_available_level() 타임아웃")

                # 3단계: PLAYING 상태가 될 때까지 대기 (최대 15초)
                deadline = time.time() + 15
                while time.time() < deadline:
                    try:
                        state = self.ar.get_game_state()
                    except (socket.timeout, OSError):
                        time.sleep(0.5)
                        continue
                    if state == GameState.PLAYING:
                        break
                    if state == GameState.REQUESTNOVELTYLIKELIHOOD:
                        try:
                            self.ar.report_novelty_likelihood(0.0, 1.0, [], 0, "")
                        except (socket.timeout, OSError):
                            pass
                    time.sleep(0.2)

            print(f"  Game PLAYING. Ready.")
            try:
                self.ar.fully_zoom_out()
            except (socket.timeout, OSError):
                print(f"  [경고] fully_zoom_out() 타임아웃")

            # ★ Level 21+ 같은 복잡한 레벨 로드 후 warmup
            # Java 서버가 물리 엔진을 완전히 초기화하는 시간 필요
            if self.current_level_num and self.current_level_num >= 20:
                #print(f"  [warmup] Level {self.current_level_num} 물리 엔진 초기화 중...")
                try:
                    self.ar.server_socket.settimeout(5)
                    # dummy get_state를 한 번 더 호출해서 물리 엔진 깨우기
                    _ = self.ar.get_game_state()
                    #print(f"  [warmup] ✓ 물리 엔진 준비 완료")
                except (socket.timeout, OSError):
                    print(f"  [경고] warmup get_game_state() 타임아웃 (무시)")
                finally:
                    self.ar.server_socket.settimeout(None)

            self.ar.server_socket.settimeout(None)  # 타임아웃 해제
            return self.get_state()[:2]

        except Exception as e:
            print(f"  [에러] reset() 실패: {e}")
            self.ar.server_socket.settimeout(None)
            raise

    def _load_level_paths(self):
        """config.xml에서 레벨 파일 경로 목록을 파싱 (1-based 인덱스)."""
        config_path = os.path.join(project_root, "ScienceBirds/MacOS/config.xml")
        try:
            content = open(config_path, encoding='utf-16').read()
            paths = re.findall(r'level_path="([^"]+)"', content)
            #print(f"Loaded {len(paths)} level paths from config.xml")
            return paths
        except Exception as e:
            print(f"[경고] config.xml 파싱 실패: {e}")
            return []

    def _parse_level_birds(self, level_num):
        """level_num(1-based)에 해당하는 XML에서 새 목록을 파싱."""
        if not self.level_paths or level_num < 1 or level_num > len(self.level_paths):
            return []
        xml_path = self.level_paths[level_num - 1]
        try:
            content = open(xml_path, encoding='utf-16').read()
            return re.findall(r'<Bird\s+type="([^"]+)"', content)
        except Exception as e:
            print(f"[경고] 레벨 XML 파싱 실패 ({xml_path}): {e}")
            return []

    def get_state(self):
        t0 = time.time()

        # Screenshot (타임아웃 설정: 기본 10초)
        try:
            self.ar.server_socket.settimeout(10)
            t1 = time.time()
            screenshot = self.ar.do_screenshot()
            t2 = time.time()
            if t2 - t1 > 0.2:
                print(f"    [경고] do_screenshot(): {(t2-t1)*1000:.1f}ms")
        except (socket.timeout, OSError) as e:
            print(f"  ❌ do_screenshot() 타임아웃 → 게임 창 확인 필요! ({e})")
            # ★ 핵심: 타임아웃 발생 시 특수한 오류 코드로 반환 (step에서 감지)
            raise RuntimeError("SCREENSHOT_TIMEOUT")
        finally:
            self.ar.server_socket.settimeout(None)

        # Crop UI elements (score bar top, left margin) then resize — matches AIBirds원본
        h, w = screenshot.shape[:2]
        crop = screenshot[min(75, h):min(400, h), min(40, w):]
        if crop.size == 0:
            crop = screenshot

        t3 = time.time()
        image = cv2.resize(crop, (128, 128))
        t4 = time.time()
        if t4 - t3 > 0.1:
            print(f"    [경고] cv2.resize(): {(t4-t3)*1000:.1f}ms")

        image = image.astype(np.float32) / 255.0

        # Bird info — 5차원 one-hot (원본 AIBirds와 동일, evaluate_ab.py 호환)
        # [Red=0, Blue=1, Yellow=2, Black=3, White=4]
        bird_vec = np.zeros(5, dtype=np.float32)
        if self.shots_count < len(self.current_level_birds):
            bird_type = self.current_level_birds[self.shots_count]
            idx = XML_BIRD_TYPE_IDX.get(bird_type)
            if idx is not None:
                bird_vec[idx] = 1.0

        has_birds = self.shots_count < len(self.current_level_birds)

        total = time.time() - t0
        if total > 0.5:
            print(f"    [경고] get_state() 전체: {total*1000:.1f}ms")

        return image, bird_vec, has_birds

    def get_action_mask(self):
        """현재 새에 대한 action mask 반환. Red만 tap_time=0 고정."""
        mask = np.ones(self.num_actions, dtype=np.float32)
        if self.shots_count < len(self.current_level_birds):
            bird_type = self.current_level_birds[self.shots_count]
            if bird_type == 'BirdRed':
                # Red: 스킬 없음 → tap_time=0인 action만 허용 (t_idx=0)
                for action in range(self.num_actions):
                    if action % TAP_TIME_RESOLUTION != 0:
                        mask[action] = 0.0
        return mask

    def step(self, action):
        try:
            return self._step_impl(action)
        except RuntimeError as e:
            if "SCREENSHOT_TIMEOUT" in str(e):
                # ★ 스크린샷 타임아웃 → 즉시 에피소드 스킵
                print(f"  🔴 스크린샷 타임아웃으로 에피소드 스킵")
                blank_img = np.zeros((128, 128, 3), dtype=np.float32)
                return (blank_img, np.zeros(5, dtype=np.float32)), -1.0, True, {"score": 0, "state": "timeout"}
            raise

    def _step_impl(self, action):
        self.shots_count += 1
        #print(f"  → Step 시작 (Shot {self.shots_count}, Action {action})")

        # ★ Action 디코딩: 각도 × tap_time (20 × 10 = 200, 원본과 동일)
        # tap_time = 스킬 발동 시간 (새가 날아가는 도중 화면 탭 시간)
        a_idx = action // TAP_TIME_RESOLUTION                          # 0~19
        t_idx = action % TAP_TIME_RESOLUTION                           # 0~9

        alpha = PHI + int(a_idx * (180 - PHI - PSI) / (ANGLE_RESOLUTION - 1))
        # tap_time = 스킬 발동 시간 (t2로 Java에 전달)
        # 원본 naive_agent: t2=tap_time, t1=0
        tap_time = int(t_idx / TAP_TIME_RESOLUTION * MAXIMUM_TAP_TIME)

        # 동적 타임아웃
        base_physics_time = 3.0 + (tap_time / MAXIMUM_TAP_TIME) * 8.0
        expected_physics_time = max(3.0, min(15.0, base_physics_time))  # 최대 15초
        socket_timeout = int(expected_physics_time + 25)  # 여유 25초 추가 (기존 15초 → 25초)

        # Find slingshot to get release point
        try:
            # ★ 핵심: get_ground_truth_without_screenshot() 호출에 타임아웃 설정
            self.ar.server_socket.settimeout(socket_timeout)

            gt = self.ar.get_ground_truth_without_screenshot()
            gtr = GroundTruthReader(gt, self.model_cv, self.target_class)
            slings = gtr.find_slingshot_mbr()
            if not slings:
                self.ar.fully_zoom_out()
                time.sleep(0.3)
                gt = self.ar.get_ground_truth_without_screenshot()
                gtr = GroundTruthReader(gt, self.model_cv, self.target_class)
                slings = gtr.find_slingshot_mbr()

            sling = slings[0]
            # Reference Point Calculation (Uppercase X, Y)
            cx = sling.X + int(sling.width * 0.45)
            cy = sling.Y + int(sling.height * 0.35)
        except (socket.timeout, OSError) as e:
            # Timeout 발생 시 버퍼 초기화 후 기본값 사용
            print(f"  [경고] get_ground_truth_without_screenshot() 타임아웃: {e}")
            try:
                self.ar._clear_buffer()
            except Exception:
                pass
            cx, cy = 180, 420
        except Exception as e:
            # 기타 에러
            print(f"  [경고] slingshot 검출 실패: {e}")
            cx, cy = 180, 420
        finally:
            self.ar.server_socket.settimeout(None)

        # Match AIBirds원본 angle_to_vector logic
        rad_alpha = np.deg2rad(alpha)
        # Pull distance: 80 pixels
        dx = - np.sin(rad_alpha) * 80
        dy = np.cos(rad_alpha) * 80

        release_x = int(cx + dx)
        release_y = int(cy + dy)

        # Handle any pending novelty likelihood request BEFORE shooting
        pre_state = self.ar.get_game_state()
        if pre_state == GameState.REQUESTNOVELTYLIKELIHOOD:
            self.ar.report_novelty_likelihood(0.0, 1.0, [], 0, "")
            time.sleep(0.3)

        # ★ 스킬 발동 로그 (원본: tap_time을 game_speed_factor 없이 직접 전달)
        bird_name = "없음"
        bird_skill = "없음"
        if self.shots_count - 1 < len(self.current_level_birds):
            bird_name = self.current_level_birds[self.shots_count - 1]
            skill_map = {
                'BirdRed': '없음', 'BirdBlue': '분열(3마리)',
                'BirdYellow': '가속', 'BirdBlack': '폭발(자동)',
                'BirdWhite': '계란낙하'
            }
            bird_skill = skill_map.get(bird_name, '없음')

        if tap_time > 0:
            print(f"  [스킬] Shot {self.shots_count}: {bird_name}({bird_skill}), 각도={alpha}°, t_tap={tap_time}ms")
        else:
            print(f"  [스킬] Shot {self.shots_count}: {bird_name}({bird_skill}), 각도={alpha}°, t_tap=0 (스킬 없음)")

        # 발사 시도 (최대 2회 재시도)
        max_retries = 2
        for attempt in range(max_retries + 1):
            self.ar.server_socket.settimeout(socket_timeout)  # 동적 타임아웃 적용
            try:
                # ★ 원본과 동일: t1=0 고정, t2=tap_time (스킬 발동 시간, 정규화 없음)
                result = self.ar.shoot(release_x, release_y, 0, tap_time, False)
                if tap_time > 0:
                    print(f"  [스킬] 발동 완료! {bird_name}, t_tap={tap_time}ms, result={result}")
                break  # 성공하면 빠져나옴
            except (socket.timeout, OSError) as e:
                if attempt < max_retries:
                    # ★ 핵심: 타임아웃 발생 시 버퍼를 초기화해서 프로토콜 재동기화
                    try:
                        self.ar._clear_buffer()
                    except Exception:
                        pass
                    time.sleep(expected_physics_time / 2 + 1)  # 더 길게 대기
                    state = self.get_state()
                    continue
                else:
                    # 최종 타임아웃, 에피소드 스킵
                    # 최후의 수단: 버퍼를 초기화하고 아예 다른 상태로 진행
                    try:
                        self.ar._clear_buffer()
                        self.ar.report_novelty_likelihood(0.0, 1.0, [], 0, "")
                    except Exception:
                        pass
                    blank_img = np.zeros((128, 128, 3), dtype=np.float32)
                    return (blank_img, np.zeros(5, dtype=np.float32)), -1.0, True, {"score": 0, "state": "timeout"}
            finally:
                self.ar.server_socket.settimeout(None)

        # 물리 엔진 안정화 대기 (단축: 2초 → 1초)
        #time.sleep(1)

        self.ar.server_socket.settimeout(socket_timeout)  # 동적 타임아웃
        try:
            state = self.ar.get_game_state()
            # Handle novelty likelihood request (competition framework checkpoint)
            if state == GameState.REQUESTNOVELTYLIKELIHOOD:
                self.ar.report_novelty_likelihood(0.0, 1.0, [], 0, "")
                time.sleep(0.3)
                state = self.ar.get_game_state()

            score = self.ar.get_current_score()
        except (socket.timeout, OSError) as e:
            print(f"  [에러] get_game_state/score 타임아웃: {e}")
            # ★ 타임아웃 시 버퍼 초기화로 프로토콜 복구 시도
            try:
                self.ar._clear_buffer()
            except Exception:
                pass
            state = GameState.PLAYING
            score = 0
        finally:
            self.ar.server_socket.settimeout(None)

        # Get next state and check if birds are left
        next_img, next_bird_vec, _ = self.get_state()

        # ★ 새로운 설계: 모든 새를 다 쓸 때까지 진행
        all_birds_used = (self.shots_count >= len(self.current_level_birds))

        # 방어: 새가 없으면 스킵
        if self.shots_count > len(self.current_level_birds):
            blank_img = np.zeros((128, 128, 3), dtype=np.float32)
            return (blank_img, np.zeros(5)), 0.0, True, {
                "score": 0,
                "state": GameState.PLAYING,
                "warning": "Step called after all birds used"
            }

        done = all_birds_used or state == GameState.WON or state == GameState.LOST

        # ★ 리워드: delta 기반 (매 샷의 점수 기여도)
        score_val = int(score) if not hasattr(score, '__len__') else int(score[0]) if len(score) > 0 else 0
        delta_score = score_val - self.prev_score
        self.prev_score = score_val

        SCORE_NORMALIZATION = 5000
        r_delta = float(delta_score) / SCORE_NORMALIZATION
        r_time  = -0.05                                      # 매 샷 시간 페널티
        r_no_op = -0.2 if delta_score == 0 else 0.0         # 빈 샷 페널티
        reward  = r_delta + r_time + r_no_op

        if state == GameState.WON or (all_birds_used and state != GameState.LOST):
            reward += 3.0            # 승리 보너스 (축소: 10→3)
        elif state == GameState.LOST:
            reward -= 1.0            # 실패 페널티

        return (next_img, next_bird_vec), reward, done, {
            "score": score,
            "state": state,
            "shots_count": self.shots_count,
            "total_birds": len(self.current_level_birds),
            "all_birds_used": all_birds_used
        }
