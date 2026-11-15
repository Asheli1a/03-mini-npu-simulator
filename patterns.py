"""보너스 패턴 생성기. 원본 JSON과 필터는 변경하지 않는다."""


def validate_pattern_size(size):
    """3 이상의 홀수 정수만 허용한다."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("크기는 정수여야 합니다.")
    if size < 3 or size % 2 == 0:
        raise ValueError("크기는 3 이상의 홀수여야 합니다.")
    return size


def generate_cross(size):
    """가운데 행과 열을 1.0으로 채운 새 배열을 반환한다."""
    validate_pattern_size(size)
    middle = size // 2
    return [
        [1.0 if row == middle or column == middle else 0.0
         for column in range(size)]
        for row in range(size)
    ]


def generate_x(size):
    """두 대각선을 1.0으로 채운 새 배열을 반환한다."""
    validate_pattern_size(size)
    return [
        [1.0 if row == column or row + column == size - 1 else 0.0
         for column in range(size)]
        for row in range(size)
    ]
