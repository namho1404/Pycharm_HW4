# marker constants
ROCKY_ROBOT = 'rocky_robot'
SPORTY_ROBOT = 'sporty_robot'

# emotion constants
HAPPY = 'happy'
SAD = 'sad'
ANGRY = 'angry'

# search constants
ROCK = 'rock'
SPORT = 'sport'

# marker table
MARKER_TABLE = [[[[0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 1, 0, 0]], ROCKY_ROBOT], [
                    [[1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1],
                     [1, 0, 1, 0, 1, 0, 1, 0, 0]], SPORTY_ROBOT]]


# match marker pattern to database record
def match_glyph_pattern(marker_pattern):
    marker_found = False
    marker_rotation = None
    marker_name = None

    for marker_record in MARKER_TABLE:
        for idx, val in enumerate(marker_record[0]):
            if marker_pattern == val:
                marker_found = True
                marker_rotation = idx
                marker_name = marker_record[1]
                break
        if marker_found: break

    return (marker_found, marker_rotation, marker_name)