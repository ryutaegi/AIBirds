"""
AIBirds원본 레벨 XML을 sciencebirdsframework Mac 게임 포맷으로 변환.

AIBirds원본 포맷:
  <Camera x="0" y="2" minWidth="20" maxWidth="30">   ← 닫히지 않음
  <Score highScore ="32000">                           ← 닫히지 않음
  <Slingshot x="-8" y="-2.5">                         ← 닫히지 않음

Mac 포맷:
  <Camera x="0" y="-1" minWidth="25" maxWidth="35" /> ← self-closing
  <Score highScore="0" />                              ← self-closing
  <Slingshot x="-8" y="-2.5" />                       ← self-closing (좌표 유지)
"""

import os
import re
import shutil
import glob

# 원본 레벨 폴더 (Windows 게임 내 레벨)
SRC_BASE = os.path.join(
    os.path.dirname(__file__),
    "../AIBirds원본/src/envs/ab/Science Birds 0.3.8/Science Birds_Data/StreamingAssets/Levels/novelty_level_0"
)

# 목적지: Mac 게임 레벨 폴더
DST_BASE = os.path.join(
    os.path.dirname(__file__),
    "ScienceBirds/MacOS/9001.app/Contents/Resources/Data/StreamingAssets/Levels/novelty_level_0/type_classic_blocks/Levels"
)


def convert_level_xml(src_path, dst_path):
    # 원본 파일은 XML 헤더에 utf-16 선언이 있지만 실제는 ASCII
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Camera: 닫히지 않는 태그 → self-closing, y를 2에서 -1로
    content = re.sub(
        r'<Camera\s+x="([^"]+)"\s+y="[^"]+"\s+minWidth="([^"]+)"\s+maxWidth="([^"]+)"\s*>',
        r'<Camera x="\1" y="-1" minWidth="\2" maxWidth="\3" />',
        content
    )

    # Score: 닫히지 않는 태그 → self-closing, 공백 제거
    content = re.sub(
        r'<Score\s+highScore\s*=\s*"[^"]*"\s*>',
        r'<Score highScore="0" />',
        content
    )

    # Slingshot: 닫히지 않는 태그 → self-closing, x를 Mac 게임 기본값 -12.0으로 변경
    content = re.sub(
        r'<Slingshot\s+x="[^"]+"\s+(y="[^"]+")(\s*)>',
        r'<Slingshot x="-12.0" \1 />',
        content
    )

    # Level 태그 속성 공백 정리: width ="2" → width="2"
    content = re.sub(
        r'<Level\s+width\s*=\s*"([^"]+)"',
        r'<Level width="\1"',
        content
    )

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-16") as f:
        f.write(content)

    print(f"  변환 완료: {os.path.basename(dst_path)}")


def main():
    src_base = os.path.abspath(SRC_BASE)
    dst_base = os.path.abspath(DST_BASE)

    if not os.path.exists(src_base):
        print(f"[오류] 원본 레벨 폴더를 찾을 수 없습니다: {src_base}")
        return

    # type01, type13~17 (블록 타워 레벨 타입만 선택)
    target_types = [
        "type01-1Pig-0TNT-standardBlocks",
        "type13-1Pig-1TNT-standard",
        "type14-5Pigs-0TNT-standardBlocks",
        "type15-4Pigs-0to2TNT-standardBlocks",
        "type17-5Pigs-0TNT-manyBirds",
    ]

    converted_paths = []
    total = 0

    for type_dir in target_types:
        src_levels_dir = os.path.join(src_base, type_dir, "Levels")
        if not os.path.exists(src_levels_dir):
            print(f"[건너뜀] {type_dir}/Levels 폴더 없음")
            continue

        xml_files = sorted(glob.glob(os.path.join(src_levels_dir, "level-*.xml")))
        print(f"\n{type_dir}: {len(xml_files)}개 레벨 변환 중...")

        for src_path in xml_files:
            fname = os.path.basename(src_path)
            # 파일명에 타입 prefix 추가해서 충돌 방지
            type_short = type_dir.split("-")[0]  # "type01"
            dst_fname = f"{type_short}_{fname}"
            dst_path = os.path.join(dst_base, dst_fname)

            convert_level_xml(src_path, dst_path)
            converted_paths.append(dst_path)
            total += 1

    if total == 0:
        print("\n[오류] 변환할 레벨 파일을 찾지 못했습니다.")
        return

    # config.xml 생성 (변환된 레벨들을 포함)
    config_path = os.path.join(
        os.path.dirname(__file__),
        "ScienceBirds/MacOS/config_classic.xml"
    )
    write_config(converted_paths, config_path)
    print(f"\n총 {total}개 레벨 변환 완료.")
    print(f"새 config 파일: {config_path}")
    print("\n사용법: config.xml을 config_classic.xml로 교체하거나,")
    print("  ScienceBirds/MacOS/config.xml을 config_classic.xml로 덮어씌우세요.")


def write_config(level_paths, config_path):
    lines = []
    lines.append('<?xml version="1.0" encoding="utf-16"?>')
    lines.append('<evaluation>')
    lines.append('  <novelty_detection_measurement step="1" measure_in_training="True" measure_in_testing="True" />')
    lines.append('  <trials>')
    lines.append('    <trial id="0" number_of_executions="1" checkpoint_time_limit="9999" checkpoint_interaction_limit="9999" notify_novelty="False">')
    lines.append('      <game_level_set mode="training" time_limit="999999" total_interaction_limit="999999" attempt_limit_per_level="3" allow_level_selection="False">')

    for p in level_paths:
        lines.append(f'        <game_levels level_path="{p}" />')

    lines.append('      </game_level_set>')
    lines.append('    </trial>')
    lines.append('  </trials>')
    lines.append('</evaluation>')

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-16") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  config 생성: {config_path}")


if __name__ == "__main__":
    main()
