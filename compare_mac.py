"""동일 입력의 2차원/1차원 MAC 비교. 표준 라이브러리만 사용한다."""
import csv
import platform
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

import main as app
from patterns import generate_cross, generate_x


def _flatten_unchecked(matrix):
    """사전 검증된 배열만 받는 측정용 변환. 행 우선 순서를 유지한다."""
    result = []
    for row in matrix:
        for value in row:
            result.append(value)
    return result


def _batch_average_ms(operation, repeats):
    start = perf_counter_ns()
    for _ in range(repeats):
        operation()
    return (perf_counter_ns() - start) / repeats / 1_000_000


def compare_mac(pattern, filter_matrix, repeats=1000, rounds=6):
    for name, value, minimum in (("반복 횟수", repeats, 10), ("라운드 수", rounds, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
    if rounds % 3:
        raise ValueError("라운드 수는 실행 순서 균형을 위해 3의 배수여야 합니다.")

    score = app.calculate_score(pattern, filter_matrix)
    flat_pattern = app.flatten_matrix(pattern)
    flat_filter = app.flatten_matrix(filter_matrix)
    flat_score = app.calculate_flat_score(flat_pattern, flat_filter)
    if score != flat_score:
        raise ValueError("2차원과 1차원 점수가 일치하지 않습니다.")

    def two_d():
        return app.mac(pattern, filter_matrix)

    def one_d():
        return app.mac_flat(flat_pattern, flat_filter)

    def converted():
        return app.mac_flat(_flatten_unchecked(pattern),
                            _flatten_unchecked(filter_matrix))

    operations = [("2d", two_d), ("1d_ready", one_d),
                  ("1d_with_conversion", converted)]
    for _, operation in operations:
        if operation() != score:
            raise ValueError("측정 경로의 계산 결과가 일치하지 않습니다.")
        for _ in range(5):
            operation()

    samples = {name: [] for name, _ in operations}
    for round_index in range(rounds):
        offset = round_index % 3
        order = operations[offset:] + operations[:offset]
        for name, operation in order:
            samples[name].append(_batch_average_ms(operation, repeats))

    stages = {
        name: {"average_ms": mean(values), "min_ms": min(values),
               "max_ms": max(values), "samples_ms": values}
        for name, values in samples.items()
    }
    return {"size": len(pattern), "mac_count": len(pattern) ** 2,
            "score": score, "repeats": repeats, "rounds": rounds,
            "stages": stages}


def run_comparison():
    results = []
    print("=== 2차원 / 1차원 MAC 비교 ===")
    print("동일한 생성 X 패턴과 Cross 필터, 필터 1개 기준")
    print("검증·생성·I/O 제외. 배치 평균은 호출·반복 비용을 포함합니다.")
    print("변환 포함 경로는 매 호출마다 두 배열을 검증 없이 평탄화합니다.")
    print("각 경로: 준비 5회 + 1000회 × 6라운드, 실행 순서 순환")
    print("비율=2차원/준비된 1차원. 1보다 크면 준비된 1차원이 빠릅니다.")
    print("크기      2D(ms)    1D준비(ms)  변환포함(ms)   비율")
    for size in (3, 5, 13, 25):
        result = compare_mac(generate_x(size), generate_cross(size))
        results.append(result)
        stages = result["stages"]
        a = stages["2d"]["average_ms"]
        b = stages["1d_ready"]["average_ms"]
        c = stages["1d_with_conversion"]["average_ms"]
        ratio = f"{a / b:.3f}" if b > 0 else "측정불가"
        print(f"{size}×{size:<4} {a:>10.6f} {b:>12.6f} {c:>12.6f} {ratio:>8}")

    path = Path(__file__).resolve().parent / "mac_comparison.csv"
    fields = ["python", "platform", "size", "mac_count", "score", "stage",
              "repeats_per_round", "rounds", "average_ms", "min_ms", "max_ms",
              "round_averages_ms"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for name, stats in result["stages"].items():
                writer.writerow({
                    "python": platform.python_version(), "platform": platform.platform(),
                    "size": result["size"], "mac_count": result["mac_count"],
                    "score": result["score"], "stage": name,
                    "repeats_per_round": result["repeats"], "rounds": result["rounds"],
                    "average_ms": stats["average_ms"], "min_ms": stats["min_ms"],
                    "max_ms": stats["max_ms"],
                    "round_averages_ms": ";".join(map(str, stats["samples_ms"])),
                })
    print(f"측정 결과 저장: {path.name}")
    return results


if __name__ == "__main__":
    try:
        run_comparison()
    except (OSError, ValueError, OverflowError) as error:
        print(f"비교 실패: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n측정을 중단했습니다.", file=sys.stderr)
        sys.exit(130)
