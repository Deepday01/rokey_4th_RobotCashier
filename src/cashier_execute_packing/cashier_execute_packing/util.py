
# # todo 로봇 중심 깊이 조정: 그리퍼 잡는 너비에 따른 높이차
# def get_gripper_depth_offset(width: float) -> float:
#     """
#     width(mm)를 입력받아 depth 보정값 (26 - depth)을 반환한다.
#     0~1100 구간은 선형 보간으로 계산한다.
#     범위를 벗어나면 양 끝값으로 clamp 한다.
#     """

#     table = [
#         (0.0, 26.0),
#         (10.0, 26.0),
#         (20.0, 26.0),
#         (30.0, 25.9),
#         (40.0, 25.8),
#         (50.0, 25.6),
#         (60.0, 25.3),
#         (70.0, 24.9),
#         (80.0, 24.6),
#         (90.0, 24.1),
#         (100.0, 23.4),
#         (110.0, 22.3),
#     ]


#     # table = [
#     #     (0.0,  260.0),
#     #     (10.0, 260.0),
#     #     (20.0, 260.0),
#     #     (30.0, 259.0),
#     #     (40.0, 258.0),
#     #     (50.0, 256.0),
#     #     (60.0, 253.0),
#     #     (70.0, 249.0),
#     #     (80.0, 246.0),
#     #     (90.0, 241.0),
#     #     (100.0, 234.0),
#     #     (110.0, 223.0),
#     # ]

#     if not isinstance(width, (int, float)):
#         raise TypeError(f"width must be int or float, got {type(width).__name__}")

#     width = float(width)

#     if width <= table[0][0]:
#         return 26.0 - table[0][1]

#     if width >= table[-1][0]:
#         return 26.0 - table[-1][1]

#     for i in range(len(table) - 1):
#         w1, d1 = table[i]
#         w2, d2 = table[i + 1]

#         if w1 <= width <= w2:
#             ratio = (width - w1) / (w2 - w1)
#             depth = d1 + ratio * (d2 - d1)
#             return 26.0 - depth

#     raise RuntimeError(f"Interpolation failed for width={width}")



def get_gripper_depth_offset(width: float) -> float:
    table = [
        (0.0,  260.0),
        (10.0, 260.0),
        (20.0, 260.0),
        (30.0, 259.0),
        (40.0, 258.0),
        (50.0, 256.0),
        (60.0, 253.0),
        (70.0, 249.0),
        (80.0, 246.0),
        (90.0, 241.0),
        (100.0, 234.0),
        (110.0, 223.0),
    ]
    max_depth = 260.0

    if not isinstance(width, (int, float)):
        raise TypeError(f"width must be int or float, got {type(width).__name__}")

    width = float(width)

    if width <= table[0][0]:
        depth = table[0][1]
    elif width >= table[-1][0]:
        depth = table[-1][1]
    else:
        depth = None
        for i in range(len(table) - 1):
            w1, d1 = table[i]
            w2, d2 = table[i + 1]
            if w1 <= width <= w2:
                ratio = (width - w1) / (w2 - w1)
                depth = d1 + ratio * (d2 - d1)
                break

        if depth is None:
            raise RuntimeError(f"Interpolation failed for width={width}")

    return max_depth - depth