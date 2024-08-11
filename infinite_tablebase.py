import random
import itertools as iter

KING_SQUARE = (0, 0)
KING_THREATENS = {(-1, -1), (-1, 0), (-1, 1),
                   (0, 1),           (0, -1),
                   (1, -1), (1, 0), (1, 1)} # , (1, 2), (-1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, 1), (-2, -1)}
MATING_SQUARES = KING_THREATENS | {KING_SQUARE}

assert len(MATING_SQUARES) == 9
assert len(KING_THREATENS) == 8
WINNING_POSITIONS = set()


def move(piece, v):
    return piece[0] + v[0], piece[1] + v[1]


# return if v is a natural number multiple of u. Assumes u != (0,0)
def is_natural_multiple(v, direction):
    if direction[0] != 0:
        scalar = v[0] // direction[0]
    else:
        scalar = v[1] // direction[1]
    return (scalar > 0 and (scalar * direction[0], scalar * direction[1]) == v), scalar


def rider_threatens(direction, piece_square, target_square, board):
    works, distance = is_natural_multiple((target_square[0] - piece_square[0], target_square[1] - piece_square[1]), direction)
    if works:
        for piece in board:
            if piece is not None:
                a, b = is_natural_multiple((piece[0] - piece_square[0], piece[1] - piece_square[1]), direction)
                if a and b < distance:
                    return False
        return True
    return False


def move_on_board(board, index, v):
    ls = list(board)
    ls[index] = move(board[index], v)
    return tuple(ls)


class PieceType:
    def __init__(self, symbol, riders=(), jumpers=(), rider_bound=None, is_royal=False, is_pawn=False):
        assert len(riders) == len(set(riders))
        assert len(jumpers) == len(set(jumpers))
        assert len(set(jumpers).intersection(set(riders))) == 0
        self.riders = set(riders)
        self.jumpers = set(jumpers)
        self.is_royal = is_royal
        self.is_pawn = is_pawn
        self.symbol = symbol
        self.rider_bound = rider_bound if rider_bound is not None else 2 ** 64

    # We don't need to check if the Black king is in the way.
    def threatens_from(self, board, my_index, target_square):
        piece_square = board[my_index]
        if piece_square is None:
            return False
        if piece_square == target_square:
            return False
        if self.is_pawn:
            if move(piece_square, (-1, 1)) == target_square or move(piece_square, (-1, -1)) == target_square:
                return True
            return False

        if (target_square[0] - piece_square[0], target_square[1] - piece_square[1]) in self.jumpers:
            return True
        for v in self.riders:
            if rider_threatens(v, piece_square, target_square, board):
                return True
        return False

    def get_resulting_board_states(self, board, index, move_bound):
        if board[index] is None:
            return []
        moves = []
        for v in self.jumpers:
            new_board = move_on_board(board, index, v)
            # TODO with black pieces we need to check check and captures.
            if not new_board[index] in board:
                if (not self.is_royal) or (new_board[index] not in KING_THREATENS):
                    moves.append(new_board)
        for v in self.riders:
            k = 1
            new_pieces = move_on_board(board, index, v)
            while new_pieces[index] not in board and k < min(self.rider_bound, move_bound):
                k += 1
                if (not self.is_royal) or (new_pieces[index] not in KING_THREATENS):
                    moves.append(new_pieces)
                new_pieces = move_on_board(new_pieces, index, v)
        return moves


CHANCELLOR = PieceType("C", riders=[(1, 0), (-1, 0), (0, 1), (0, -1)],
                       jumpers=[(1, 2), (-1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, 1), (-2, -1)])
ARCHBISHOP = PieceType("A", riders=[(1, 1), (-1, 1), (1, -1), (-1, -1)],
                       jumpers=[(1, 2), (-1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, 1), (-2, -1)])
ROOK = PieceType("R", riders=[(1, 0), (-1, 0), (0, 1), (0, -1)])
QUEEN = PieceType("Q", riders=[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
AMAZON = PieceType("A", riders=[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
                   jumpers=[(1, 2), (-1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, 1), (-2, -1)])
KNIGHT = PieceType("N", jumpers=[(1, 2), (-1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, 1), (-2, -1)])
KNIGHTRIDER = PieceType("R", riders=[(1, 2), (-1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, 1), (-2, -1)])
BISHOP = PieceType("B", riders=[(1, 1), (-1, 1), (-1, -1), (1, -1)])

KING = PieceType("K", jumpers=[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)], is_royal=True)
GUARD = PieceType("G", jumpers=[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
HAWK = PieceType("H", jumpers=[(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, 2), (-2, -2), (2, -2),
                               (3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (-3, 3), (-3, -3), (3, -3)])
PAWN = PieceType("P", jumpers=[(-1, 0)], is_pawn=True)


def get_white_moves(board, move_bound):
    result = [board]
    for i in range(len(PIECES)):
        result += PIECES[i].get_resulting_board_states(board, i, move_bound)
    return result


def get_white_preimages(board, move_bound):
    result = [board]
    for i in range(len(PIECES)):
        if PIECES[i].is_pawn:
            if board[i] is not None:
                for v in PIECES[i].jumpers:
                    new_board = move_on_board(board, i, (-v[0], -v[1]))
                    if not new_board[i] in board:
                        result.append(new_board)
        else:
            result += PIECES[i].get_resulting_board_states(board, i, move_bound)

    return [w for w in result if KING_SQUARE not in w and not is_threatened(KING_SQUARE, w)]


def get_black_moves(board):
    result = []
    for v in KING_THREATENS:
        new_board = []
        for p in board:
            if p is not None:
                p = move(p, v)  # technically this moves the king along -v
                new_board.append(p if p != KING_SQUARE else None)
            else:
                new_board.append(None)
        if not is_threatened(KING_SQUARE, tuple(new_board)):
            result.append(tuple(new_board))
    return result


def get_black_preimages(board):
    if is_threatened(KING_SQUARE, board):
        return []
    result = []
    for v in KING_THREATENS:
        new_board = []
        for p in board:
            if p is not None:
                new_board.append(move(p, v))
            else:
                new_board.append(None)
        if KING_SQUARE not in new_board and all(piece is None or (not piece.is_royal) or (new_board[i] not in MATING_SQUARES) for i, piece in enumerate(PIECES)):
            result.append(tuple(new_board))
    return result


def is_threatened(square, board):
    return any(PIECES[i].threatens_from(board, i, square) for i in range(len(board)))


def is_mate(piece_positions):
    return all(is_threatened(v, piece_positions) for v in MATING_SQUARES)


def is_stalemate(piece_positions):
    return all(is_threatened(v, piece_positions) for v in KING_THREATENS) and not is_threatened(KING_SQUARE,
                                                                                                piece_positions)


def branch_value(x):
    return len(get_black_moves(x))


def pieces_same_color(s1, s2):
    if s1 is None or s2 is None:
        return True
    return (s1[0] + s1[1] + s2[0] + s2[1]) % 2 == 0


def pieces_different_color(s1, s2):
    return not pieces_same_color(s1, s2)


def get_mates(n, parity_condition=lambda x: True):
    coordinates = [(a, b) for a in range(-n, n + 1) for b in range(-n, n + 1)] + [None]
    coordinates.remove(KING_SQUARE)
    royalty = [i for i in range(len(PIECES)) if PIECES[i].is_royal]
    positions = {}

    for p in iter.permutations(coordinates, len(PIECES)):
        if is_mate(p):
            if parity_condition(p):
                if all(p[i] not in MATING_SQUARES for i in royalty):
                    positions[p] = 0

    return positions


def get_mates_faster(bound, parity_condition=lambda x: True):
    coordinates = [(a, b) for a in range(-bound, bound + 1) for b in range(-bound, bound + 1)] + [None]
    coordinates.remove(KING_SQUARE)
    ms = list(MATING_SQUARES)
    royalty = [i for i in range(len(PIECES)) if PIECES[i].is_royal]
    patterns = [dict() for _ in PIECES]
    keys = [set() for _ in PIECES]

    for piece_index, piece in enumerate(PIECES):
        for c in coordinates:
            pattern = 0
            for i, p in enumerate(ms):
                if piece.threatens_from((c,), 0, target_square=p):
                    pattern |= 2**i
            if pattern not in patterns[piece_index]:
                patterns[piece_index][pattern] = []
                keys[piece_index].add(pattern)
            patterns[piece_index][pattern].append(c)

    positions = dict()
    for p in iter.product(*keys):
        s = 0
        for k in p:
            s |= k
        if s == (1 << len(MATING_SQUARES)) - 1:
            for r in iter.product(*[patterns[i][key] for i, key in enumerate(p)]):
                # print(r)
                if all(pos is None or r.count(pos) == 1 for pos in r):
                    if is_mate(r):
                        if parity_condition(r):
                            if all(r[i] not in MATING_SQUARES for i in royalty):
                                positions[r] = 0
    # print(positions)
    return positions


def print_position(board, indent=0):
    min_x = min([b[0] for b in board if b is not None] + [0]) - 2
    min_y = min([b[1] for b in board if b is not None] + [0]) - 2
    gap_x = max([b[0] for b in board if b is not None] + [0]) - min_x + 2 + 1
    gap_y = max([b[1] for b in board if b is not None] + [0]) - min_y + 2 + 1
    arr = [['_' for _ in range(gap_y)] for _ in range(gap_x)]

    for i, b in enumerate(board):
        if b is not None:
            arr[b[0]-min_x][b[1]-min_y] = PIECES[i].symbol
    arr[KING_SQUARE[0]-min_x][KING_SQUARE[1]-min_y] = 'k'

    print("|  " * indent + str(board))
    # if indent < 4:
    for l in arr:
        print("|  " * indent + (" ".join(l)))


def print_variations(position, white_wins, black_wins, move_bound, indent=0, black_to_move=True):
    print_position(position, indent)
    if black_to_move:
        if black_wins[position] <= 0:
            return
        for p in sorted(get_black_moves(position), key=lambda x: -white_wins[x]):
            assert p in white_wins
            print_variations(p, white_wins, black_wins, move_bound, indent+1, black_to_move=False)
    else:
        if white_wins[position] <= 0:
            return
        found_position = False
        for p in get_white_moves(position, move_bound):
            if p in black_wins and black_wins[p] < white_wins[position]:
                assert black_wins[p] + 1 == white_wins[position]
                print_variations(p, white_wins, black_wins, move_bound, indent+1, black_to_move=True)
                found_position = True
                break
        assert found_position


def get_best_position(positions, position_score):
    best = -2**64
    best_pos = None
    for board in positions:
        if position_score(board) > best:
            best = position_score(board)
            best_pos = board
    return best_pos


def rotations_and_reflections(positions):
    new_positions = dict()
    for p in positions:
        for i in range(4):
            new_positions[p] = 0
            p = tuple((-b, a) for (a, b) in p)
        p = tuple((-a, b) for (a, b) in p)
        for i in range(4):
            new_positions[p] = 0
            p = tuple((-b, a) for (a, b) in p)
    return new_positions


def get_wins(winning_positions, move_bound, pos_score=None, should_print_variations=False):
    print(f"Starting the search {len(winning_positions)}")
    black_wins = winning_positions
    white_wins = {}
    unexplored = winning_positions
    # for p in winning_positions:
    #     print_position(p)
    i = 1
    while len(unexplored) > 0:
        forceable_position = ((-1,-2), (-2,3), (4,-2))
        print_position(forceable_position)
        if forceable_position in white_wins:
            print("White to move is a win!")
            print_variations(forceable_position, white_wins, black_wins, move_bound, black_to_move=False)
            print(i)
        if forceable_position in black_wins:
            print("Black to move is a win!")
            print_variations(forceable_position, white_wins, black_wins, move_bound, black_to_move=True)
    #   print(i)
        new_wins = {}
        if i % 2 == 0:  # Black just moved
            for p in unexplored:
                for q in get_black_preimages(p):
                    if q not in black_wins:
                        if all(r in white_wins or is_mate(r) for r in get_black_moves(q)):
                            # print("HI")
                            new_wins[q] = i
            unexplored = new_wins
            black_wins.update(new_wins)
            print(f"{i} {len(new_wins)}")

            # if len(new_wins) < 10:
            #     print("-" * 80)
            #     for p in new_wins:
            #         print_position(p)
            #     print("-"*80)
            pos_to_print = get_best_position(new_wins, pos_score) if pos_score is not None else random.choice(list(new_wins.keys()))
            if should_print_variations:
                print_variations(pos_to_print, white_wins, black_wins, move_bound, black_to_move=True)
            else:
                print_position(pos_to_print)
        else:  # White just moved
            for p in unexplored:
                for q in get_white_preimages(p, move_bound):
                    if q not in white_wins:
                        new_wins[q] = i
            unexplored = new_wins
            white_wins.update(new_wins)
            print(f"{i} {len(new_wins)}")
            pos_to_print = get_best_position(new_wins, pos_score) if (pos_score is not None) else random.choice(list(new_wins.keys()))
            if should_print_variations:
                print_variations(pos_to_print, white_wins, black_wins, move_bound, black_to_move=False)
            else:
                print_position(pos_to_print)
        i += 1


PIECES = [KING, ARCHBISHOP, HAWK]
if __name__ == "__main__":


    three_bishops_one_knight_traps = [((1, 0), (-1, 2), (-3, -2), (-1, 0)), ((2, -1), (0, 1), (-2, -3), (0, -1)), ((2, 1), (0, 3), (-2, -1), (0, 1)), ((3, 0), (1, 2), (-1, -2), (1, 0))]

    def score(position):
        return min(abs(position[3][0])/3 + abs(position[3][1])/3, max(abs(position[3][0]), abs(position[3][1]))/2) if position[0] is not None else 0
    # get_wins(rotations_and_reflections(rotations_and_reflections(traps)), 10, pos_score=score, should_print_variations=False)
    # get_wins(rotations_and_reflections(three_bishops_one_knight_traps), 10, pos_score=score, should_print_variations=False)
    get_wins(get_mates_faster(10), 10, pos_score=None, should_print_variations=False)
    # get_wins(rotations_and_reflections({((-1, -1), (2, -1), (-1, 3)), ((-1, -1), (2, -1), (-1, -3)), ((0, -1), (3, -1), (-4, -2))}), 20, pos_score=score, should_print_variations=False)
