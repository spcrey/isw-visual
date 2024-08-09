class XyCoordGenerator:
    def __call__(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int]]:
        coord = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                coord.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
            coord.append((x, y))
        else:
            err = dy / 2.0
            while y != y1:
                coord.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            coord.append((x, y))
        return coord
    