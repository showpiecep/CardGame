class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Position:
    def __init__(self, upper_left: Point, lower_right: Point):
        self.upper_left = upper_left
        self.lower_right = lower_right