from dynagroup.von_mises.util import degrees_to_radians


LABELS_OF_DIRECTIONS = ["E", "N", "W", "S"]
DEGREES_OF_DIRECTIONS = [0, 90, 180, 270]
RADIANS_OF_DIRECTIONS = degrees_to_radians(DEGREES_OF_DIRECTIONS)
DIRECTIONS_DICT = dict(zip(LABELS_OF_DIRECTIONS, RADIANS_OF_DIRECTIONS))
