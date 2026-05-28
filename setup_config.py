#!/usr/bin/env python3
"""
ScienceBirds config.xml 생성 스크립트
500개 레벨을 N배 반복해서 무한 순환하는 config.xml 생성

사용법:
    python3 setup_config.py --repeat 200  # 10만개 레벨 생성
"""

import os
import re
import argparse
from pathlib import Path

def extract_level_paths_from_config(config_path):
    """기존 config.xml에서 500개 레벨 경로 추출"""
    print(f"[1/3] config.xml 읽기: {config_path}")

    with open(config_path, 'r', encoding='utf-16') as f:
        content = f.read()

    # <game_levels level_path="..." /> 패턴 추출
    pattern = r'<game_levels\s+level_path="([^"]+)"\s*/>'
    level_paths = re.findall(pattern, content)

    print(f"  ✓ {len(level_paths)}개 레벨 경로 추출됨")
    return level_paths

def generate_config_xml(level_paths, repeat_times, output_path):
    """config.xml 동적 생성 (level_paths를 repeat_times번 반복)"""
    print(f"\n[2/3] config.xml 생성 중... ({len(level_paths)} × {repeat_times} = {len(level_paths) * repeat_times}개 레벨)")

    # XML 헤더 및 구조
    lines = [
        '<?xml version="1.0" encoding="utf-16"?>',
        '<evaluation>',
        '    <novelty_detection_measurement step="1" measure_in_training="False" measure_in_testing="True" />',
        '    <trials>',
        '        <trial id="0" number_of_executions="1" checkpoint_time_limit="2147483647" checkpoint_interaction_limit="2147483647" notify_novelty="False">',
        '            <game_level_set mode="training" time_limit="999999" total_interaction_limit="999999" attempt_limit_per_level="1" allow_level_selection="False">',
    ]

    # 레벨 경로를 repeat_times번 반복
    for repeat in range(repeat_times):
        for level_path in level_paths:
            lines.append(f'                <game_levels level_path="{level_path}" />')

    # XML 닫기
    lines += [
        '            </game_level_set>',
        '        </trial>',
        '    </trials>',
        '</evaluation>',
    ]

    # UTF-16으로 저장
    content = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-16') as f:
        f.write(content)

    print(f"  ✓ {len(level_paths) * repeat_times}개 레벨 config.xml 생성됨")
    print(f"  ✓ 저장 위치: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='ScienceBirds config.xml 동적 생성')
    parser.add_argument('--repeat', type=int, default=200,
                       help='레벨 반복 횟수 (기본값: 200 = 10만개 레벨)')
    args = parser.parse_args()

    # 경로 설정
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(
        project_root,
        'ScienceBirds/MacOS/config.xml'
    )
    output_path = config_path  # 원본 덮어쓰기

    print("=" * 70)
    print("ScienceBirds config.xml 동적 생성")
    print("=" * 70)
    print(f"반복 횟수: {args.repeat}회")
    print(f"총 레벨: 500 × {args.repeat} = {500 * args.repeat:,}개")
    print()

    # 1. 기존 config.xml에서 레벨 경로 추출
    if not os.path.exists(config_path):
        print(f"❌ 오류: config.xml을 찾을 수 없습니다: {config_path}")
        return False

    level_paths = extract_level_paths_from_config(config_path)

    if len(level_paths) == 0:
        print("❌ 오류: config.xml에서 레벨 경로를 추출할 수 없습니다")
        return False

    if len(level_paths) != 500:
        print(f"⚠️  경고: {len(level_paths)}개 레벨만 발견됨 (500개 예상)")

    # 2. 새 config.xml 생성
    generate_config_xml(level_paths, args.repeat, output_path)

    # 3. 완료
    print(f"\n[3/3] 완료!")
    print(f"✓ {500 * args.repeat:,}개 레벨로 학습 가능합니다")
    print(f"✓ train_v2_hybrid.py를 실행하세요")
    print()
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
