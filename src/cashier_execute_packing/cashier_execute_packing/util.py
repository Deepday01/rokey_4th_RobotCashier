
# todo 로봇 중심 깊이 조정: 그리퍼 잡는 너비에 따른 높이차
def get_gripper_depth_offset(width: float) -> float:
    """
    width(mm)를 입력받아 depth 보정값 (26 - depth)을 반환한다.
    0~1100 구간은 선형 보간으로 계산한다.
    범위를 벗어나면 양 끝값으로 clamp 한다.
    """

    table = [
        (0.0, 26.0),
        (100.0, 26.0),
        (200.0, 26.0),
        (300.0, 25.9),
        (400.0, 25.8),
        (500.0, 25.6),
        (600.0, 25.3),
        (700.0, 24.9),
        (800.0, 24.6),
        (900.0, 24.1),
        (1000.0, 23.4),
        (1100.0, 22.3),
    ]

    if not isinstance(width, (int, float)):
        raise TypeError(f"width must be int or float, got {type(width).__name__}")

    width = float(width)

    if width <= table[0][0]:
        return 26.0 - table[0][1]

    if width >= table[-1][0]:
        return 26.0 - table[-1][1]

    for i in range(len(table) - 1):
        w1, d1 = table[i]
        w2, d2 = table[i + 1]

        if w1 <= width <= w2:
            ratio = (width - w1) / (w2 - w1)
            depth = d1 + ratio * (d2 - d1)
            return 26.0 - depth

    raise RuntimeError(f"Interpolation failed for width={width}")