import json
from pathlib import Path


def describe_matrix(matrix):
    if not isinstance(matrix, list):
        return "오류: 배열이 아닙니다."

    row_count = len(matrix)
    column_counts = [
        len(row) if isinstance(row, list) else None
        for row in matrix
    ]

    if not column_counts:
        return "오류: 빈 배열입니다."

    if None in column_counts:
        return "오류: 배열이 아닌 행이 있습니다."

    if len(set(column_counts)) != 1:
        return (
            f"오류: 행마다 열 개수가 다릅니다. "
            f"행={row_count}, 열 개수={column_counts}"
        )

    return f"{row_count}×{column_counts[0]}"


def main():
    data_path = Path(__file__).resolve().parent / "data.json"

    try:
        with data_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {data_path}")
        return
    except json.JSONDecodeError as error:
        print(f"JSON 형식 오류: {error}")
        return
    except (OSError, UnicodeError) as error:
        print(f"파일 읽기 오류: {error}")
        return

    if not isinstance(data, dict):
        print("최상위 데이터는 객체여야 합니다.")
        return

    print("=== 데이터 기본 정보 ===")
    print("최상위 키:", list(data.keys()))
    print("meta:", data.get("meta"))

    filters = data.get("filters")
    patterns = data.get("patterns")

    if not isinstance(filters, dict):
        print("filters는 객체여야 합니다.")
        return

    if not isinstance(patterns, dict):
        print("patterns는 객체여야 합니다.")
        return

    print("\n=== 필터 확인 ===")
    for size_key, filter_group in filters.items():
        print(f"[{size_key}]")

        if not isinstance(filter_group, dict):
            print("  오류: 필터 그룹이 객체가 아닙니다.")
            continue

        for label, matrix in filter_group.items():
            print(f"  {label}: {describe_matrix(matrix)}")

    print("\n=== 패턴 확인 ===")
    print(f"패턴 수: {len(patterns)}")

    for case_id, case in patterns.items():
        if not isinstance(case, dict):
            print(f"{case_id}: 오류: 케이스가 객체가 아닙니다.")
            continue

        shape = describe_matrix(case.get("input"))
        expected = case.get("expected")

        print(
            f"{case_id}: "
            f"크기={shape}, expected={expected!r}"
        )


if __name__ == "__main__":
    main()