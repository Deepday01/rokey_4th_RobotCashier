
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









