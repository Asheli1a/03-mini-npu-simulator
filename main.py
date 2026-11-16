import json
import math
import re
from time import perf_counter_ns
from pathlib import Path
from patterns import generate_cross, generate_x, validate_pattern_size

class CalculationError(ValueError):
    """유효한 입력의 계산 중 발생한 수치 오류."""


def validate_matrix(matrix, name):
    """비어 있지 않은 정사각형 숫자 배열인지 확인하고 크기를 반환한다."""
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name}: 비어 있지 않은 2차원 리스트가 필요합니다.")

    size = len(matrix)

    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(
                f"{name}: {row_index + 1}행의 길이가 "
                f"배열 크기 {size}와 일치하지 않습니다."
            )

        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name}: ({row_index + 1}, {column_index + 1})의 "
                    "값은 숫자여야 합니다."
                )

            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"{name}: ({row_index + 1}, {column_index + 1})의 "
                    "값은 유한한 숫자여야 합니다."
                )

    return size


def mac(pattern, filter_matrix):
    """같은 크기로 검증된 정사각형 배열 두 개의 MAC 점수를 계산한다."""
    score = 0.0

    for row in range(len(pattern)):
        for column in range(len(pattern)):
            score += pattern[row][column] * filter_matrix[row][column]

    return score


def calculate_score(pattern, filter_matrix):
    """입력 배열을 검증한 뒤 MAC 점수를 반환한다."""
    pattern_size = validate_matrix(pattern, "패턴")
    filter_size = validate_matrix(filter_matrix, "필터")

    if pattern_size != filter_size:
        raise ValueError(
            f"크기 불일치: 패턴={pattern_size}×{pattern_size}, "
            f"필터={filter_size}×{filter_size}"
        )

    try:
        score = mac(pattern, filter_matrix)
    except OverflowError:
        raise CalculationError("계산 중 숫자 표현 범위를 초과했습니다.") from None

    if not math.isfinite(score):
        raise CalculationError("계산 결과가 유한하지 않습니다.")

    return score

def flatten_matrix(matrix):
    """정사각형 숫자 배열을 행 우선 순서의 새 리스트로 변환한다."""
    validate_matrix(matrix, "평탄화 배열")

    flattened = []

    for row in matrix:
        for value in row:
            flattened.append(value)

    return flattened


def validate_vector(vector, name):
    """비어 있지 않은 유한한 숫자 리스트인지 검증한다."""
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"{name}: 비어 있지 않은 1차원 리스트가 필요합니다.")

    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name}: {index + 1}번째 값은 숫자여야 합니다."
            )

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"{name}: {index + 1}번째 값은 유한한 숫자여야 합니다."
            )

    return len(vector)


def mac_flat(pattern, filter_vector):
    """검증된 동일 길이의 1차원 배열 두 개를 MAC 계산한다."""
    score = 0.0

    for index in range(len(pattern)):
        score += pattern[index] * filter_vector[index]

    return score


def calculate_flat_score(pattern, filter_vector):
    """1차원 배열을 검증하고 유한한 MAC 점수를 반환한다."""
    pattern_length = validate_vector(pattern, "1차원 패턴")
    filter_length = validate_vector(filter_vector, "1차원 필터")

    if pattern_length != filter_length:
        raise ValueError(
            f"길이 불일치: 패턴={pattern_length}, 필터={filter_length}"
        )

    try:
        score = mac_flat(pattern, filter_vector)
    except OverflowError:
        raise CalculationError(
            "1차원 계산 중 숫자 표현 범위를 초과했습니다."
        ) from None

    if not math.isfinite(score):
        raise CalculationError("1차원 계산 결과가 유한하지 않습니다.")

    return score

EPSILON = 1e-9

FAILURE_LABELS = {
    "tie": "동점 판정(UNDECIDED)",
    "prediction_mismatch": "예측 불일치",
    "data_schema": "데이터/스키마 오류",
    "numeric": "수치 계산 오류",
}


def normalize_label(label):
    """외부 라벨을 내부 표준 라벨 Cross 또는 X로 변환한다."""
    if not isinstance(label, str):
        raise ValueError("라벨은 문자열이어야 합니다.")

    cleaned = label.strip().lower()

    mapping = {
        "+": "Cross",
        "cross": "Cross",
        "x": "X",
    }

    if cleaned not in mapping:
        raise ValueError(f"지원하지 않는 라벨입니다: {label!r}")

    return mapping[cleaned]


def decide_label(score_cross, score_x):
    """유한한 두 점수를 epsilon 정책으로 비교한다."""
    if not all(
        math.isfinite(score)
        for score in (score_cross, score_x)
    ):
        raise CalculationError("판정할 점수는 유한한 숫자여야 합니다.")

    difference = score_cross - score_x

    if abs(difference) < EPSILON:
        return "UNDECIDED"

    if score_cross > score_x:
        return "Cross"

    return "X"

def parse_row(text, size):
    """한 줄을 지정된 개수의 유한한 실수 목록으로 변환한다."""
    parts = text.split()

    if len(parts) != size:
        raise ValueError(
            f"각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
        )

    try:
        values = [float(part) for part in parts]
    except ValueError:
        raise ValueError("숫자로 변환할 수 없는 값이 있습니다.") from None

    if not all(math.isfinite(value) for value in values):
        raise ValueError("nan, inf 대신 유한한 숫자를 입력하세요.")

    return values


def read_matrix(name, size=3):
    """배열을 한 행씩 입력받으며, 오류가 난 행만 다시 입력받는다."""
    print(f"\n{name} ({size}줄 입력, 각 줄에 숫자 {size}개)")
    matrix = []

    for row_index in range(size):
        while True:
            text = input(f"{row_index + 1}행> ")

            try:
                row = parse_row(text, size)
            except ValueError as error:
                print(f"입력 형식 오류: {error}")
                continue

            matrix.append(row)
            break

    return matrix


def print_matrix(name, matrix):
    print(f"\n[{name}]")

    for row in matrix:
        print(" ".join(f"{value:g}" for value in row))


def decide_ab(score_a, score_b):
    """기존 점수 비교 정책을 재사용하여 A/B 판정으로 변환한다."""
    result = decide_label(score_a, score_b)

    return {
        "Cross": "A",
        "X": "B",
        "UNDECIDED": "UNDECIDED",
    }[result]


def run_manual_mode():
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")

    print("\n=== 필터 저장 확인 ===")
    print_matrix("필터 A", filter_a)
    print_matrix("필터 B", filter_b)

    pattern = read_matrix("패턴")

    try:
        score_a = calculate_score(pattern, filter_a)
        score_b = calculate_score(pattern, filter_b)
        result = decide_ab(score_a, score_b)
    except (ValueError, OverflowError) as error:
        print(f"\n계산 오류: {error}")
        return

    print("\n=== MAC 결과 ===")
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    print(
        f"판정 기준: 두 점수 차이의 절댓값이 {EPSILON:g} 미만이면 "
        "판정 불가(UNDECIDED)입니다."
    )

    if result == "UNDECIDED":
        print(f"판정: 판정 불가 (UNDECIDED, 점수 차이 < {EPSILON:g})")
    else:
        print(f"판정: {result}")

    performance = benchmark_mac(pattern, filter_a)
    print_performance_table([performance])

def load_data(path):
    """JSON을 읽고 전체 분석에 필요한 최상위 구조를 검증한다."""
    with Path(path).open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("최상위 데이터는 객체여야 합니다.")

    for key in ("filters", "patterns"):
        if key not in data or not isinstance(data[key], dict):
            raise ValueError(f"{key}는 객체여야 합니다.")

    if not data["patterns"]:
        raise ValueError("patterns에 분석할 케이스가 없습니다.")

    return data


def parse_case_size(case_id):
    """size_N_idx 형식의 식별자에서 N을 추출한다."""
    if not isinstance(case_id, str):
        raise ValueError("케이스 식별자는 문자열이어야 합니다.")

    match = re.fullmatch(r"size_([1-9][0-9]*)_([1-9][0-9]*)", case_id)

    if match is None:
        raise ValueError(f"케이스 키 형식 오류: {case_id!r}")

    size = int(match.group(1))

    if size not in (5, 13, 25):
        raise ValueError(f"지원하지 않는 JSON 패턴 크기: {size}")

    return size


def get_filters(filters, size):
    """해당 크기의 필터를 검증하고 표준 라벨로 반환한다."""
    if not isinstance(filters, dict):
        raise ValueError("filters는 객체여야 합니다.")

    size_key = f"size_{size}"
    group = filters.get(size_key)

    if not isinstance(group, dict):
        raise ValueError(f"{size_key} 필터 그룹이 없거나 객체가 아닙니다.")

    normalized = {}

    for raw_label, matrix in group.items():
        label = normalize_label(raw_label)

        if label in normalized:
            raise ValueError(f"{size_key}: 정규화 후 중복 필터 {label}")

        actual_size = validate_matrix(matrix, f"{size_key}/{label}")

        if actual_size != size:
            raise ValueError(
                f"{size_key}/{label}: "
                f"필터 크기 불일치, 기대={size}, 실제={actual_size}"
            )

        normalized[label] = matrix

    if set(normalized) != {"Cross", "X"}:
        raise ValueError(f"{size_key}: Cross와 X 필터가 모두 필요합니다.")

    return normalized


def analyze_case(case_id, case, filters):
    """한 케이스를 분석한다. 데이터 오류는 호출한 쪽에서 처리한다."""
    size = parse_case_size(case_id)

    if not isinstance(case, dict):
        raise ValueError("케이스는 객체여야 합니다.")

    if "input" not in case or "expected" not in case:
        raise ValueError("input 또는 expected가 없습니다.")

    pattern = case["input"]
    actual_size = validate_matrix(pattern, f"{case_id}/input")

    if actual_size != size:
        raise ValueError(
            f"패턴 크기 불일치: 키의 크기={size}, 실제={actual_size}"
        )

    expected = normalize_label(case["expected"])
    selected_filters = get_filters(filters, size)

    score_cross = calculate_score(pattern, selected_filters["Cross"])
    score_x = calculate_score(pattern, selected_filters["X"])

    if not all(math.isfinite(score) for score in (score_cross, score_x)):
        raise CalculationError("계산 결과가 유한하지 않습니다.")

    prediction = decide_label(score_cross, score_x)
    passed = prediction == expected

    if passed:
        reason = ""
        failure_type = None
    elif prediction == "UNDECIDED":
        reason = "epsilon 동점 규칙에 따라 UNDECIDED"
        failure_type = "tie"
    else:
        reason = f"예측과 정답 불일치: {prediction} != {expected}"
        failure_type = "prediction_mismatch"

    return {
        "case_id": case_id,
        "score_cross": score_cross,
        "score_x": score_x,
        "prediction": prediction,
        "expected": expected,
        "passed": passed,
        "reason": reason,
        "failure_type": failure_type,
    }


def analyze_data(data):
    """데이터 오류가 있는 케이스도 FAIL로 기록하고 분석을 계속한다."""
    if not isinstance(data, dict):
        raise ValueError("최상위 데이터는 객체여야 합니다.")
    for key in ("filters", "patterns"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"{key}는 객체여야 합니다.")

    results = []

    for case_id, case in data["patterns"].items():
        try:
            result = analyze_case(case_id, case, data["filters"])
        except (ValueError, OverflowError) as error:
            if isinstance(error, (CalculationError, OverflowError)):
                category = "수치 계산 오류"
                failure_type = "numeric"
            else:
                category = "데이터/스키마 오류"
                failure_type = "data_schema"
            result = {
                "case_id": case_id,
                "score_cross": None,
                "score_x": None,
                "prediction": None,
                "expected": None,
                "passed": False,
                "reason": f"{category}: {error}",
                "failure_type": failure_type,
            }

        results.append(result)

    return results


def summarize_results(results):
    """analyze_data의 결과를 사유 문자열이 아닌 실패 유형으로 집계한다."""
    counts = {failure_type: 0 for failure_type in FAILURE_LABELS}
    passed = 0

    for result in results:
        if result["passed"]:
            if result["failure_type"] is not None:
                raise ValueError("PASS 결과에는 실패 유형을 지정할 수 없습니다.")
            passed += 1
        else:
            failure_type = result["failure_type"]
            if failure_type not in counts:
                raise ValueError(f"지원하지 않는 실패 유형: {failure_type!r}")
            counts[failure_type] += 1

    return {
        "total": len(results),
        "passed": passed,
        "failed": sum(counts.values()),
        "undecided": counts["tie"],
        "failure_counts": counts,
    }


def run_json_mode():
    data_path = Path(__file__).resolve().parent / "data.json"

    try:
        data = load_data(data_path)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"\nJSON 로드 실패: {error}")
        return

    results = analyze_data(data)

    print("\n=== JSON 분석 결과 ===")

    for result in results:
        print(f"\n--- {result['case_id']} ---")

        if result["prediction"] is not None:
            print(f"Cross 점수: {result['score_cross']!r}")
            print(f"X 점수: {result['score_x']!r}")
            print(
                f"판정: {result['prediction']} | "
                f"expected: {result['expected']}"
            )

        status = "PASS" if result["passed"] else "FAIL"
        print(f"결과: {status}")

        if result["reason"]:
            print(f"사유: {result['reason']}")

    run_json_performance(data)
    summary = summarize_results(results)
    failures = [result for result in results if not result["passed"]]

    print("\n=== 결과 요약 ===")
    print(f"총 테스트: {summary['total']}개")
    print(f"통과: {summary['passed']}개")
    print(f"실패: {summary['failed']}개")
    print("\n실패 유형별 집계 (동점은 실패에 포함):")
    for failure_type, label in FAILURE_LABELS.items():
        print(f"- {label}: {summary['failure_counts'][failure_type]}개")

    if failures:
        print("\n실패 케이스:")
        for result in failures:
            print(f"- {result['case_id']}: {result['reason']}")

def benchmark_mac(pattern, filter_matrix, repeats=1000):
    """검증을 제외한 MAC 함수 호출 시간을 반복 측정한다."""
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise ValueError("반복 횟수는 정수여야 합니다.")

    if repeats < 10:
        raise ValueError("반복 횟수는 10 이상이어야 합니다.")

    size = validate_matrix(pattern, "측정 패턴")
    filter_size = validate_matrix(filter_matrix, "측정 필터")

    if size != filter_size:
        raise ValueError("측정할 패턴과 필터의 크기가 다릅니다.")

    calculate_score(pattern, filter_matrix)

    # 첫 호출의 영향을 줄이기 위한 준비 실행
    for _ in range(5):
        mac(pattern, filter_matrix)

    elapsed_total_ns = 0

    for _ in range(repeats):
        start = perf_counter_ns()
        mac(pattern, filter_matrix)
        elapsed_ns = perf_counter_ns() - start
        elapsed_total_ns += elapsed_ns

    return {
        "size": size,
        "repeats": repeats,
        "average_ms": elapsed_total_ns / repeats / 1_000_000,
        "mac_count": size * size,
    }


def print_performance_table(rows):
    print("\n=== 성능 분석 ===")
    print("측정 대상: 필터 1개에 대한 MAC 함수 호출")
    print("크기       평균 시간(ms)    MAC 횟수(N²)    반복 횟수")
    print("-" * 60)

    for row in rows:
        size_text = f"{row['size']}×{row['size']}"
        print(
            f"{size_text:<10}"
            f"{row['average_ms']:>12.6f}"
            f"{row['mac_count']:>16}"
            f"{row['repeats']:>13}"
        )


def run_json_performance(data):
    cross_3 = [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ]

    rows = [benchmark_mac(cross_3, cross_3)]

    for size in (5, 13, 25):
        measured = False

        for case_id, case in data["patterns"].items():
            try:
                if parse_case_size(case_id) != size:
                    continue

                if not isinstance(case, dict) or "input" not in case:
                    continue

                pattern = case["input"]

                if validate_matrix(pattern, case_id) != size:
                    continue

                selected_filters = get_filters(data["filters"], size)
                row = benchmark_mac(pattern, selected_filters["Cross"])

            except ValueError:
                continue

            rows.append(row)
            measured = True
            break

        if not measured:
            print(f"\n성능 측정 제외: {size}×{size}의 유효한 입력이 없습니다.")

    print_performance_table(rows)
    return rows

def read_generated_size():
    """3 이상의 홀수 크기를 입력받는다."""
    while True:
        text = input("크기 N (3 이상의 홀수, 예: 3, 5, 13, 25): ").strip()

        try:
            size = int(text)
        except ValueError:
            print("입력 오류: 정수를 입력하세요.")
            continue

        try:
            return validate_pattern_size(size)
        except ValueError as error:
            print(f"입력 오류: {error}")


def run_generated_mode():
    size = read_generated_size()

    while True:
        choice = input("패턴 선택 (1: Cross, 2: X): ").strip()

        if choice in ("1", "2"):
            break

        print("입력 오류: 1 또는 2를 입력하세요.")

    filter_cross = generate_cross(size)
    filter_x = generate_x(size)

    if choice == "1":
        pattern = filter_cross
        expected = "Cross"
    else:
        pattern = filter_x
        expected = "X"

    print(f"\n=== 생성 패턴 분석: {size}×{size} ===")
    print("필터: 자동 생성한 이진 Cross / X")
    print("원본 data.json의 가중치 필터와는 별개입니다.")

    if size <= 25:
        print_matrix("생성 패턴", pattern)
    else:
        print("큰 배열의 전체 출력은 생략합니다.")

    try:
        score_cross = calculate_score(pattern, filter_cross)
        score_x = calculate_score(pattern, filter_x)
        prediction = decide_label(score_cross, score_x)
        performance = benchmark_mac(pattern, filter_cross)
    except (ValueError, OverflowError) as error:
        print(f"\n계산 오류: {error}")
        return

    print(f"\nCross 점수: {score_cross!r}")
    print(f"X 점수: {score_x!r}")
    print(f"판정: {prediction} | 생성한 패턴: {expected}")
    print_performance_table([performance])

def main():
    while True:
        print("\n=== Mini NPU Simulator ===")
        print("1. 사용자 입력 (3×3)")
        print("2. data.json 분석")
        print("3. 생성 패턴 분석")
        print("0. 종료")

        choice = input("선택: ").strip()

        if choice == "1":
            run_manual_mode()
        elif choice == "2":
            run_json_mode()
        elif choice == "3":
            run_generated_mode()
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("0, 1, 2, 3 중 하나를 입력하세요.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 프로그램을 종료합니다.")