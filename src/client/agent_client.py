"""Agent to Server API"""

from enum import Enum
import socket
import struct
import json
import logging
import numpy as np
from PIL import Image
import sys
import time
import os
import cv2

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
class GameState(Enum):
    """The state of the game at a particular instant"""
    UNKNOWN = 0
    MAIN_MENU = 1
    EPISODE_MENU = 2
    LEVEL_SELECTION = 3
    LOADING = 4
    PLAYING = 5
    WON = 6
    LOST = 7
    NEWTESTSET = 8
    NEWTRAININGSET = 9
    RESUMETRAINING = 10
    NEWTRIAL = 11
    REQUESTNOVELTYLIKELIHOOD = 12
    EVALUATION_TERMINATED = 13


class PlayingMode(Enum):
    """Mode of play"""
    COMPETITION = 0
    TRAINING = 1


class RequestCodes(Enum):
    """Codes for different requests"""
    DoScreenShot = 11
    Configure = 1
    SetGameSimulationSpeed = 2
    LoadLevel = 51
    RestartLevel = 52
    LoadNextAvailableLevel = 53
    Cshoot = 31
    Pshoot = 32
    GTshoot = 38
    GetState = 12
    FullyZoomOut = 34
    GetNoOfLevels = 15
    GetCurrentLevel = 14
    ShootSeq = 11
    CFastshoot = 41
    PFastshoot = 42
    ShootSeqFast = 43
    GetAllLevelScores = 23
    ClickInCentre = 36
    FullyZoomIn = 35
    GetGroundTruthWithScreenshot = 61
    GetGroundTruthWithoutScreenshot = 62
    GetNoisyGroundTruthWithScreenshot = 63
    GetNoisyGroundTruthWithoutScreenshot = 64
    GetCurrentLevelScore = 65
    ReportNoveltyLikelihood = 66
    ReportNoveltyDescription = 67
    ReadyForNewSet = 68
    NoveltyInfo = 69
    BatchGT = 70
    GetInitialStateScreenShot = 71
    NoveltyHint = 72
class AgentClient:
    """Science Birds agent API"""

    def __init__(
            self,
            host,
            port,
            playing_mode=PlayingMode.TRAINING,
            **kwargs
    ):

        self.server_port = int(port)
        self.server_host = host
        self.playing_mode = playing_mode
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server_socket.settimeout(100)
        self._buffer = bytearray()

        self._extra_args = kwargs
        if "logger" in kwargs:
            self._logger = kwargs['logger']
        else:
            self._logger = logging.getLogger('Agent Client')

        logging.getLogger().setLevel(logging.INFO)
    def _clear_buffer(self):
        """Force clear the socket buffer to recover from protocol desynchronization"""
        self._buffer.clear()
        self._logger.warning("⚠️ Socket buffer cleared due to timeout/error recovery")

    def _read_raw_from_buff(self, size):
        """Read a specific number of bytes from server_socket"""
        self._logger.debug("Reading %s bytes from server", size)
        try:
            while len(self._buffer) < size:
                new_bytes = self.server_socket.recv(size - len(self._buffer))
                if not new_bytes:
                    raise ConnectionError("Server closed connection")
                self._buffer.extend(new_bytes)
            encoded = bytearray(self._buffer[:size])
            self._logger.debug(
                "Read: |%s|",
                encoded.hex()[:75] + (encoded.hex()[75:] and "...")
            )
            self._buffer = self._buffer[size:]
            return encoded
        except socket.timeout as e:
            # 버퍼에 부분 데이터가 남아있으면 프로토콜이 영구적으로 손상된다
            remaining = len(self._buffer)
            self._clear_buffer()
            self._logger.error(f"🔴 Socket timeout with {remaining} bytes in buffer - BUFFER CLEARED")
            raise e

    def _read_from_buff(self, fmt):
        """Read the struct fmt from server_socket"""
        fmt = "!" + fmt
        size = struct.calcsize(fmt)
        encoded = self._read_raw_from_buff(size)
        return struct.unpack(fmt, encoded)

    def _send_command(self, command, *args):
        """Send a command with formatted arguments to server"""
        fmt = args[0] if args else ""
        args = args[1:] if len(args) > 1 else []
        msg = bytearray(struct.pack("!B" + fmt, command.value, *args))
        self._logger.debug(
            "Sending Request %s with bytes: |%s|",
            command,
            msg.hex()[:75] + (msg.hex()[75:] and "...")
        )
        #print("Sending Request ${command} with bytes: ", msg.hex()[:75] + (msg.hex()[75:] and "..."))
        self.server_socket.sendall(msg)

    


    # INITIALIZATION
    def connect_to_server(self):
        try:
            self.server_socket.connect((self.server_host, self.server_port))
            self._logger.info(
                'Client connected to server on port: %d',
                self.server_port
            )
        except socket.error as e:
            self._logger.exception(
                'Client failed to connect to server.'
                + ' Requested HOST: %s'
                + ' Requested PORT: %d'
                + ' Error Message: %s',
                self.server_host, self.server_port, e)
            raise e

    def disconnect_from_server(self):
        try:
            self.server_socket.close()
            self._logger.info('Client disconnected from server.')
        except socket.error as e:
            self._logger.exception(
                'Client failed to disconnect from server.'
                + ' Requested HOST: %s'
                + ' Requested PORT: %d'
                + ' Error Message: %s',
                self.server_host, self.server_port, e)
            raise e

    def reconnect_to_server(self):
        """★ 강제 소켓 재연결: 기존 소켓 완전 정리 후 새로 연결"""
        import time
        try:
            print(f"[🔄 RECONNECT STEP 1] 기존 소켓 닫기...")
            # 1단계: 기존 소켓 강제 정리
            try:
                self.server_socket.close()
                print(f"[🔄 RECONNECT STEP 1] 소켓 닫음 ✓")
            except Exception as e:
                print(f"[🔄 RECONNECT STEP 1] 소켓 닫기 실패 (무시): {e}")

            # 2단계: 포트 해제 대기 (OS TCP TIME_WAIT 타임아웃)
            print(f"[🔄 RECONNECT STEP 2] 포트 해제 대기 중... (1초)")
            time.sleep(1)

            # 3단계: 새로운 socket 객체 생성
            print(f"[🔄 RECONNECT STEP 3] 새로운 socket 객체 생성...")
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._buffer.clear()
            print(f"[🔄 RECONNECT STEP 3] 새 socket 생성 완료 ✓")

            # 4단계: 서버에 다시 연결
            print(f"[🔄 RECONNECT STEP 4] {self.server_host}:{self.server_port} 연결 중...")
            self.server_socket.connect((self.server_host, self.server_port))
            print(f"[🔄 RECONNECT STEP 4] 서버 연결 성공 ✓✓✓")
            print(f"[🔄 RECONNECT COMPLETE] Socket 강제 재연결 완료!")
            self._logger.warning('🔄 Socket forcefully reconnected to server')

        except socket.error as e:
            print(f"[❌ RECONNECT FAILED] 재연결 실패: {e}")
            self._logger.error(f'❌ Reconnection failed: {e}')
            raise e

    # REQUESTS
    def configure(self, agent_id):
        """Send configure message to server"""
        self._logger.info("Sending configure request")
        self._send_command(
            RequestCodes.Configure,
            "IB",
            agent_id,
            self.playing_mode.value
        )

        (round_number, limit, levels) = self._read_from_buff("BBB")
        self._logger.info(
            'Received configuration: Round = %d, time_limit=%d, levels = %d',
            round_number, limit, levels
        )
        return (round_number, limit, levels)

    def ready_for_new_set(self):
        self._logger.info("Ready for new data set with appropriate agent.")
        self._send_command(RequestCodes.ReadyForNewSet)
        (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self._read_from_buff("IIIIBBB")
        return (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo)

    def report_novelty_likelihood(self,report_novelty_likelihood, non_novelty_likelihood, id_array, novelty_level, novelty_description, hierarchy_array={}):
        self._logger.info("report novelty likelihood")

        id_array_length = len(id_array)
        hierarchy_array_length = len(hierarchy_array) 
        encoded_description = novelty_description.encode('utf-8')
        des_bytearray = bytearray()
        des_bytearray.extend(encoded_description)
        msg_length = len(des_bytearray)
        
        if(hierarchy_array_length>0):
            self._send_command(RequestCodes.ReportNoveltyLikelihood,"ffi"+str(id_array_length)+"iii"+str(msg_length)+"si"+str(hierarchy_array_length)+"f",report_novelty_likelihood, non_novelty_likelihood,id_array_length,*id_array,novelty_level,msg_length,des_bytearray,hierarchy_array_length,*hierarchy_array)
        else:
            self._send_command(RequestCodes.ReportNoveltyLikelihood,"ffi"+str(id_array_length)+"iii"+str(msg_length)+"s",report_novelty_likelihood, non_novelty_likelihood,id_array_length,*id_array,novelty_level,msg_length,des_bytearray)            
        response = self._read_from_buff("B")[0]
        return response

    def report_novelty_description(self,novelty_description):
        self._logger.info("report novelty description")
        encoded_description = novelty_description.encode('utf-8')
        msg_length = len(encoded_description)
        self._send_command(RequestCodes.ReportNoveltyDescription,"I"+str(msg_length)+"s", msg_length, encoded_description)
        response = self._read_from_buff("B")[0]
        return response

    def set_game_simulation_speed(self, simulation_speed):
        self._logger.info("Sending set simulation speed request")
        self._send_command(RequestCodes.SetGameSimulationSpeed, "I", simulation_speed)
        response = self._read_from_buff("B")[0]
        self._logger.info("Simulation speed is set to %d", simulation_speed)
        return response

    def read_image_from_stream(self):
        """Read image from server_socket"""
        (width, height) = self._read_from_buff("II")
        total_bytes = width * height * 3
        # Read the raw RGB data
        read_bytes = 0
        # read first bytes
        image_bytes = self.server_socket.recv(2048)
        read_bytes += image_bytes.__len__()

        # read the rest
        while (read_bytes < total_bytes):
            byte_buffer = self.server_socket.recv(2048)
            byte_buffer_length = byte_buffer.__len__()
            if (byte_buffer_length != -1):
                image_bytes += byte_buffer
            else:
                break
            read_bytes += byte_buffer_length

        rgb_image = Image.frombytes("RGB", (width, height), image_bytes)  # check if  PIL is needed
        # TODO: Remove after Debug
        # rgb_image.save(os.path.join('./', 'test'), format='png')

        self._logger.info('Received screenshot')

        img = np.array(rgb_image)
        # Convert BGR to RGB
        rgb_image = img[:, :, ::-1].copy()
        #cv2.imwrite('image.png',rgb_image)
        return rgb_image

    def read_ground_truth_from_stream(self):
        """Read Ground Truth from sever_socket"""
        self._logger.debug("reading groundtruth from stream")
        msg_length = self._read_from_buff("I")[0]
        data = b''
        self._logger.debug("groundtruth length is %d bytes", msg_length)
        while len(data) < msg_length:
            packet = self.server_socket.recv(msg_length - len(data))
            if not packet:
                return None
            data += packet
        data_string = data.decode("UTF-8")
        data_string = data_string[:-5]
        return json.loads(data_string)

    def do_screenshot(self):
        """Request screenshot from server"""
        self._logger.info("Sending screenshot request")
        self._send_command(RequestCodes.DoScreenShot)
        return self.read_image_from_stream()

    def get_initial_state_screenshot(self):
        """Request screenshot from server"""
        self._logger.info("Sending screenshot request")
        self._send_command(RequestCodes.GetInitialStateScreenShot)
        return self.read_image_from_stream()


    def get_game_state(self):
        """Retrieve game state"""
        self._logger.info("Sending gamestate request")
        self._send_command(RequestCodes.GetState)
        state = GameState(self._read_from_buff("B")[0])
        self._logger.info("Got gamestate = %s", state)
        return state

    def load_level(self, level_number):
        """Load a specific level"""
        if level_number < 1:
            level_number = 1
        self._logger.info("Sending loadLevel request")
        self._send_command(RequestCodes.LoadLevel, "I", level_number)
        response = self._read_from_buff("B")[0]
        self._logger.info('Received loadLevel')
        return response

    def load_next_available_level(self):
        """Load the next available level"""
        self._logger.info("Sending load next available level request")
        self._send_command(RequestCodes.LoadNextAvailableLevel)
        level = self._read_from_buff("I")[0]
        self._logger.info('Received load next available level')
        return level

    def get_novelty_info(self):
        """query if novelty starts to appear"""
        self._send_command(RequestCodes.NoveltyInfo)
        novelty_info = self._read_from_buff("i")[0]
        self._logger.info("novelty existence is %d ", novelty_info)
        return novelty_info

    def shoot_and_record_ground_truth(self, fx, fy, t1, t2, gt_frequency, gt_option = 0):
        """ Request to execute a shot and record ground truth every gt_frequency frames
            Note: number of frames will be dependent on the set game simulation and gt_frequency
            the slower the game is -> more frequent ground truth snapshots are possible and vice verta.
        """
        start_time = time.time()

        code = RequestCodes.GTshoot
        should_read_images = False # for now turned off completely on the server and SB, in case needed - ask
        self._send_command(code, "iiiiii", fx, fy, t1, t2, gt_frequency, gt_option)

        # read how many ground truths to expect
        ground_truths_count_bytes = self._read_from_buff("I")[0]
        ground_truths_count = int(ground_truths_count_bytes)

        # read n ground truths
        gt_images = []
        gt_jsons = []
        self._logger.info("receiving ground truth batch")
        for i in range(0, ground_truths_count):

            gt = self.read_ground_truth_from_stream()
            if(should_read_images):
                im = self.read_image_from_stream()
            if(i%100 == 0):
                self._logger.info("received gt number %d", i)
            if (should_read_images):
                gt_images.append(im)
            gt_jsons.append(gt)
        self._logger.info("received %d ground truth frames ", ground_truths_count)
        self._logger.info("--- %s seconds ---", (time.time() - start_time))
        return gt_jsons

    def batch_ground_truth(self,gt_frequency,n_frames=300):
        
        code = RequestCodes.BatchGT
        should_read_images = False # for now turned off completely on the server and SB, in case needed - ask
        self._send_command(code, "ii", gt_frequency, n_frames)

        # read how many ground truths to expect
        ground_truths_count_bytes = self._read_from_buff("I")[0]
        ground_truths_count = int(ground_truths_count_bytes)

        # read n ground truths
        gt_images = []
        gt_jsons = []
        self._logger.info("receiving ground truth batch")
        for i in range(0, ground_truths_count):
            gt = self.read_ground_truth_from_stream()
            if(should_read_images):
                im = self.read_image_from_stream()
            if(i%100 == 0):
                self._logger.info("received gt number %d", i)
            if (should_read_images):
                gt_images.append(im)
            gt_jsons.append(gt)
        self._logger.info("received %d ground truth frames ", ground_truths_count)
#        print("received ground truth frames ", ground_truths_count)
        return gt_jsons


    def restart_level(self):
        """Request to restart level"""
        self._send_command(RequestCodes.RestartLevel)
        return self._read_from_buff("B")[0]

    def shoot(self, fx, fy, t1, t2, isPolar):
        code = RequestCodes.Pshoot if isPolar else RequestCodes.Cshoot
        self._send_command(code, "iiii", fx, fy, t1, t2)
        return self._read_from_buff("B")[0]

    def fast_shoot(self, fx, fy, t1, t2, isPolar):
        code = RequestCodes.PFastshoot if isPolar else RequestCodes.CFastshoot
        self._send_command(code, "iiii", fx, fy, t1, t2)
        return self._read_from_buff("B")[0]

    def get_all_level_scores(self):
        if self.playing_mode != PlayingMode.COMPETITION:
            self._logger.warning(
                "GetAllLevelScores is not recommended in %s",
                self.playing_mode
            )
        self._send_command(RequestCodes.GetAllLevelScores)
        n_levels = self._read_from_buff("I")[0]
        return self._read_from_buff("" + n_levels * "I")

    def get_current_score(self):
        self._send_command(RequestCodes.GetCurrentLevelScore)
        return self._read_from_buff("I")[0]

    def get_number_of_levels(self):
        self._logger.info("Requesting number of levels")
        self._send_command(RequestCodes.GetNoOfLevels)
        levels = self._read_from_buff("I")[0]
        self._logger.info("Number of Levels = %d", levels)
        return levels

    def get_current_level(self):
        self._send_command(RequestCodes.GetCurrentLevel)
        return self._read_from_buff("I")[0]

    def fully_zoom_in(self):
        self._send_command(RequestCodes.FullyZoomIn)
        return self._read_from_buff("B")[0]

    def fully_zoom_out(self):
        self._send_command(RequestCodes.FullyZoomOut)
        return self._read_from_buff("B")[0]

    def get_ground_truth_with_screenshot(self):
        self._logger.info("sending get_ground_truth_with_screenshot request")
        self._send_command(RequestCodes.GetGroundTruthWithScreenshot)
        gt = self.read_ground_truth_from_stream()
        im = self.read_image_from_stream()
        return (im, gt)

    def get_ground_truth_without_screenshot(self):
        self._logger.info("sending get_ground_truth_without_screenshot request")
        self._send_command(RequestCodes.GetGroundTruthWithoutScreenshot)
        return self.read_ground_truth_from_stream()

    def get_noisy_ground_truth_with_screenshot(self):
        self._logger.info("sending get_noisy_ground_truth_with_screenshot request")
        self._send_command(RequestCodes.GetNoisyGroundTruthWithScreenshot)
        gt = self.read_ground_truth_from_stream()
        im = self.read_image_from_stream()
        return (im, gt)

    def get_noisy_ground_truth_without_screenshot(self):
        self._logger.info("sending get_noisy_ground_truth_without_screenshot request")
        self._send_command(RequestCodes.GetNoisyGroundTruthWithoutScreenshot)
        gt = self.read_ground_truth_from_stream()
        return gt

    def get_novelty_hint(self,hint_level):
        """get novelty hint"""
        self._logger.info("getting novelty hint")
        self._send_command(RequestCodes.NoveltyHint, "I", hint_level)
        msg_length = self._read_from_buff("I")[0]
        data = b''
        self._logger.debug("novelty hint length is %d bytes", msg_length)
        while len(data) < msg_length:
            packet = self.server_socket.recv(msg_length - len(data))
            if not packet:
                return None
            data += packet
        data_string = data.decode("UTF-8")
        return json.loads(data_string)


if __name__ == "__main__":
    """ TEST AGENT """
    # ★ 로깅 활성화
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    with open('./server_client_config.json', 'r') as config:
        sc_json_config = json.load(config)

    client = AgentClient(**sc_json_config[0])
    try:
        client.connect_to_server()
        client.configure(2888)
        time.sleep(1)

        # ★ 1단계: env.py의 reset() 처럼 상태 처리 루프 (85-122줄)
        print("\n=== [준비] 게임 상태 준비 ===")
        for attempt in range(30):
            state = client.get_game_state()
            print(f"  [{attempt}] 상태: {state}")

            # NEWTRIAL, NEWTRAININGSET, NEWTESTSET 모두 처리
            if state in [GameState.NEWTRIAL, GameState.NEWTRAININGSET,
                        GameState.NEWTESTSET, GameState.RESUMETRAINING]:
                print(f"    → {state} 처리 중...")
                client.ready_for_new_set()
                time.sleep(0.15)
                continue

            # REQUESTNOVELTYLIKELIHOOD 처리
            if state == GameState.REQUESTNOVELTYLIKELIHOOD:
                print(f"    → REQUESTNOVELTYLIKELIHOOD 처리 중...")
                client.report_novelty_likelihood(0.0, 1.0, [], 0, "")
                time.sleep(0.15)
                continue

            # MAIN_MENU, LEVEL_SELECTION 처리 (env.py처럼)
            if state in [GameState.MAIN_MENU, GameState.LEVEL_SELECTION, GameState.EPISODE_MENU]:
                print(f"    → {state} 처리: load_next_available_level() 호출...")
                client.load_next_available_level()
                time.sleep(0.5)
                continue

            # PLAYING 또는 로드 가능한 상태 → 다음 단계
            if state == GameState.PLAYING:
                print(f"  ✓ 준비 완료: {state}")
                break

        print()

        # ★ 2단계: LoadLevel 호출
        print("\n=== [TEST] Message 51 (LoadLevel) 테스트 ===")
        response = client.load_level(10)
        print(f"✓ LoadLevel(10) 응답: {response}")

        # ★ LoadLevel 후 게임 상태 추적
        print("\n[진단] LoadLevel 후 상태 변화:")

        time.sleep(2)
        state = client.get_game_state()
        current_level = client.get_current_level()
        print(f"  [2초 후] 상태: {state}, 레벨: {current_level}")

        time.sleep(5)
        state = client.get_game_state()
        current_level = client.get_current_level()
        print(f"  [7초 후] 상태: {state}, 레벨: {current_level}")

        time.sleep(5)
        state = client.get_game_state()
        current_level = client.get_current_level()
        print(f"  [12초 후] 상태: {state}, 레벨: {current_level}")

        print()
        client.disconnect_from_server()
        exit(0)

        # ★ 1단계: 게임 준비 상태 처리 (env.py 85-122줄과 동일)
        print("[준비] 게임 상태 준비 중...")
        for attempt in range(20):
            try:
                state = client.get_game_state()
                print(f"  상태: {state}")
            except (socket.timeout, OSError):
                time.sleep(0.5)
                continue

            if state == GameState.EVALUATION_TERMINATED:
                print("  [순환] 평가 종료 → 레벨 순환 중...")
                try:
                    client.ready_for_new_set()
                except Exception as e:
                    print(f"    ⚠️ {e}")
                time.sleep(0.15)
                continue

            if state in [GameState.NEWTRAININGSET, GameState.NEWTESTSET,
                         GameState.NEWTRIAL, GameState.RESUMETRAINING]:
                print(f"  처리: {state}")
                try:
                    client.ready_for_new_set()
                except (socket.timeout, OSError):
                    pass
                time.sleep(0.15)
                continue

            if state == GameState.REQUESTNOVELTYLIKELIHOOD:
                print(f"  처리: REQUESTNOVELTYLIKELIHOOD")
                try:
                    client.report_novelty_likelihood(0.0, 1.0, [], 0, "")
                except (socket.timeout, OSError):
                    pass
                time.sleep(0.15)
                continue

            # 로드 가능한 상태 또는 PLAYING → 다음 단계
            print(f"  ✓ 준비 완료: {state}")
            break

        # ★ 2단계: 레벨 로드
        print("\n[레벨로드] 레벨 로드 중...")
        
        level_num = client.load_next_available_level()
        time.sleep(5)
        level_num = client.load_level(10)
        print(f"  [로드됨] Level: {level_num}")

        # 물리 엔진 안정화
        time.sleep(10)
        print(f"  ✓ 물리 안정화 완료")

        # ★ 3단계: PLAYING 상태 대기
        print("\n[PLAYING대기] 게임 로드 완료 대기 중...")
        max_wait = time.time() + 15
        while time.time() < max_wait:
            try:
                state = client.get_game_state()
            except (socket.timeout, OSError):
                time.sleep(0.5)
                continue

            print(f"  상태: {state}")
            if state == GameState.PLAYING:
                print(f"  ✓ 게임 PLAYING 상태")
                break
            if state == GameState.REQUESTNOVELTYLIKELIHOOD:
                try:
                    client.report_novelty_likelihood(0.0, 1.0, [], 0, "")
                except (socket.timeout, OSError):
                    pass
            time.sleep(0.2)

        # Level 20+ 물리 엔진 warmup
        if level_num >= 20:
            print(f"\n[warmup] Level {level_num} 물리 엔진 초기화 중...")
            try:
                client.server_socket.settimeout(5)
                _ = client.get_game_state()
                print(f"  ✓ 물리 엔진 준비 완료")
            except (socket.timeout, OSError):
                print(f"  ⚠️ warmup 타임아웃 (무시)")
            finally:
                client.server_socket.settimeout(None)

        # 이제 스크린샷 및 상호작용
        print(f"\n[테스트] 상호작용 테스트...")
        client.do_screenshot()

        print(f"[6] Getting current level...")
        level = client.get_current_level()
        print(f"[7] Current level: {level}")

        print(f"[8] Testing zoom...")
        client.fully_zoom_in()
        client.fully_zoom_out()

        print(f"[9] Testing shoot...")
        info = client.shoot(172, 276, 943, 264, False)

        print(f"[10] Getting ground truth...")
        image, ground_truth = client.get_ground_truth_with_screenshot()
        ground_truth = client.get_ground_truth_without_screenshot()
        noisy_image, noisy_truth = client.get_noisy_ground_truth_with_screenshot()
        noisy_truth = client.get_noisy_ground_truth_without_screenshot()

        print(f"[11] Restarting level...")
        info = client.restart_level()

        print(f"[12] Disconnecting...")
        client.disconnect_from_server()
        print(f"[✓] Test complete!")
    except socket.error as e:
        print("Error in client-server communication: " + str(e))
