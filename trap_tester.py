import copy
import math
import random
import ast
from prettytable import PrettyTable


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
    def __init__(self, symbol, riders=(), jumpers=(), is_royal=False, is_black=False):
        assert len(riders) == len(set(riders))
        assert len(jumpers) == len(set(jumpers))
        assert len(set(jumpers).intersection(set(riders))) == 0
        self.riders = set(riders)
        self.jumpers = set(jumpers)
        self.is_royal = is_royal
        self.symbol = symbol
        self.is_black = is_black

    # We don't need to check if the Black king is in the way.
    def threatens_from(self, board, my_index, target_square):
        piece_square = board[my_index]
        if piece_square is None:
            return False
        if piece_square == target_square:
            return False
        if (target_square[0] - piece_square[0], target_square[1] - piece_square[1]) in self.jumpers:
            return True
        for v in self.riders:
            if rider_threatens(v, piece_square, target_square, board):
                return True
        return False

    def get_resulting_board_states(self, board, index, move_bound, passing):
        if board[index] is None:
            return []
        if not all_pieces_different_squares(board):
            print("Tried to play a move on an illegal board!")
            assert False
            return []
        moves = [] if not passing else [board]
        for v in self.jumpers:
            new_pieces = move_on_board(board, index, v)
            if new_pieces[index] not in board:
                moves.append(new_pieces)
            else:
                capture_index = board.index(new_pieces[index])
                if PIECES[capture_index].is_black != self.is_black:
                    new_pieces = new_pieces[:capture_index] + (None,) + new_pieces[capture_index+1:]
                    moves.append(new_pieces)
        for v in self.riders:
            k = 1
            new_pieces = move_on_board(board, index, v)
            while new_pieces[index] not in board and k <= move_bound:
                k += 1
                moves.append(new_pieces)
                new_pieces = move_on_board(new_pieces, index, v)
            if new_pieces[index] in board and k <= move_bound:
                capture_index = board.index(new_pieces[index])
                if PIECES[capture_index].is_black != self.is_black:
                    new_pieces = new_pieces[:capture_index] + (None,) + new_pieces[capture_index + 1:]
                    moves.append(new_pieces)
        if self.is_royal:
            moves = [m for m in moves if not is_threatened_by_opponent(m[index], m, self.is_black)]
        if CORNER_MODE and self.is_black:
            moves = [m for m in moves if m[BLACK_KING_INDEX][0] >= CORNER_BOUND]
        return moves

    def get_preimages(self, board, index, move_bound, passing):
        # like moves, but we ignore checks and captures.
        if board[index] is None:
            return []
        if not all_pieces_different_squares(board):
            print("Tried to play a move on an illegal board!")
            return []
        moves = [] if not passing else [board]
        for v in self.jumpers:
            new_pieces = move_on_board(board, index, v)
            if new_pieces[index] not in board:
                moves.append(new_pieces)
        for v in self.riders:
            k = 1
            new_pieces = move_on_board(board, index, v)
            while new_pieces[index] not in board and k <= move_bound:
                k += 1
                moves.append(new_pieces)
                new_pieces = move_on_board(new_pieces, index, v)

        # undo potential captures.
        preimages_of_captures = []
        for i in range(len(PIECES)):
            if board[i] is None and PIECES[i].is_black != self.is_black:
                for m in moves:
                    preimages_of_captures.append(m[:i] + (board[index],) + m[i+1:])
        moves += preimages_of_captures

        # Neither side can come from a position where they threatened their opponent.
        if self.is_black and WHITE_KING_INDEX is not None:
            moves = [m for m in moves if not is_threatened_by_black(m[WHITE_KING_INDEX], m)]
        elif not self.is_black:
            moves = [m for m in moves if not is_threatened_by_white(m[BLACK_KING_INDEX], m)]
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
BLACK_KING = PieceType("k", jumpers=[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)], is_royal=True, is_black=True)
GUARD = PieceType("G", jumpers=[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
HAWK = PieceType("H", jumpers=[(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, 2), (-2, -2), (2, -2),
                               (3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (-3, 3), (-3, -3), (3, -3)])


def get_white_moves(board, move_bound, white_pass):
    result = set()
    for i in range(len(PIECES)):
        if not PIECES[i].is_black:
            result.update(PIECES[i].get_resulting_board_states(board, i, move_bound, passing=white_pass))
    # assert all(all_pieces_different_squares(b) for b in result)
    return result


def get_white_preimages(board, move_bound, white_pass):
    result = set()
    for i in range(len(PIECES)):
        if not PIECES[i].is_black:
            result.update(PIECES[i].get_preimages(board, i, move_bound, passing=white_pass))
    # assert all(all_pieces_different_squares(b) for b in result)
    return result


def get_black_moves(board, move_bound, black_pass=False):
    result = set()
    for i in range(len(PIECES)):
        if PIECES[i].is_black:
            result.update(PIECES[i].get_resulting_board_states(board, i, move_bound, passing=black_pass))
    # assert all(all_pieces_different_squares(b) for b in result)
    return result


# def get_black_preimages(board, move_bound, black_pass=False):
#     result = set()
#     for i in range(len(PIECES)):
#         if PIECES[i].is_black:
#             result.update(PIECES[i].get_preimages(board, i, move_bound, passing=black_pass))
#     # assert all(all_pieces_different_squares(b) for b in result)
#     return result


def is_threatened_by_opponent(square, board, is_black):
    return is_threatened_by_white(square, board) if is_black else is_threatened_by_black(square, board)


def is_threatened_by_white(square, board):
    return any(PIECES[i].threatens_from(board, i, square) for i in range(len(board)) if not PIECES[i].is_black)


def is_threatened_by_black(square, board):
    return any(PIECES[i].threatens_from(board, i, square) for i in range(len(board)) if PIECES[i].is_black)


def is_mate(piece_positions, move_bound=1):
    # TODO doesn't matter since this method is unused, but should we consider checking piece_positions[BLACK_KING_INDEX] == None?
    return is_threatened_by_white(piece_positions[BLACK_KING_INDEX], piece_positions) and len(get_black_moves(piece_positions, move_bound)) == 0


def is_stalemate(piece_positions, move_bound=1):
    return len(get_black_moves(piece_positions, move_bound)) == 0 and not is_threatened_by_white(piece_positions[BLACK_KING_INDEX], piece_positions)


def all_pieces_different_squares(piece_list):
    r = set()
    for p in piece_list:
        if p is not None and p in r:
            return False
        r.add(p)
    return True


def print_position(board, indent=0, coords_to_include=tuple(), show_white_threats=False):
    min_x = min([b[0] for b in board+coords_to_include if b is not None]) - 2
    min_y = min([b[1] for b in board+coords_to_include if b is not None]) - 2
    gap_x = max([b[0] for b in board+coords_to_include if b is not None]) - min_x + 2 + 1
    gap_y = max([b[1] for b in board+coords_to_include if b is not None]) - min_y + 2 + 1
    if show_white_threats:
        arr = [[' ' if is_threatened_by_white((b + min_x, a + min_y), board) else '-' for a in range(gap_y)] for b in range(gap_x)]
    else:
        arr = [['_' for _ in range(gap_y)] for _ in range(gap_x)]

    for i, b in enumerate(board):
        if b is not None:
            arr[b[0]-min_x][b[1]-min_y] = PIECES[i].symbol

    print("|  " * indent + str(board))
    # if indent < 4:
    for l in arr:
        print("|  " * indent + (" ".join(l)))


def print_all_coords(coords):
    min_x = min([b[0] for b in coords if b is not None] + [0]) - 2
    min_y = min([b[1] for b in coords if b is not None] + [0]) - 2
    gap_x = max([b[0] for b in coords if b is not None] + [0]) - min_x + 2 + 1
    gap_y = max([b[1] for b in coords if b is not None] + [0]) - min_y + 2 + 1
    arr = [['_' for _ in range(gap_y)] for _ in range(gap_x)]

    for i, b in enumerate(coords):
        if b is not None:
            arr[b[0] - min_x][b[1] - min_y] = "X"
    if (0,0) in coords:
        arr[ - min_x][ - min_y] = "+"
    else:
        arr[ - min_x][ - min_y] = "O"

    print(str(coords))
    # if indent < 4:
    for l in arr:
        print(" ".join(l))


def find_maximal_inescapable_trap(potential_traps, move_bound, remove_bad=False):
    ptp = copy.copy(potential_traps) # TODO this avoids side effects, but is the performance hit worth it?
    if remove_bad:
        bad_positions = set()
        for p in ptp:
            if not all_pieces_different_squares(p) or is_stalemate(p):
                bad_positions.add(p)
        ptp.difference_update(bad_positions)

    # print(f"bad {len(bad_positions)}")
    # print(f"good {len(ptp)}")

    keep_looping = True
    while keep_looping:
        print(len(ptp))
        keep_looping = False
        failures = set()
        for p in ptp:
            for f in get_black_moves(p, move_bound):
                if all((g not in ptp) or (g in failures) for g in get_white_moves(f, move_bound, white_pass=True)):
                    failures.add(p)
                    keep_looping = True
        ptp.difference_update(failures)

    return ptp


def find_maximal_inescapable_tempo_gaining_trap(ptp, move_bound, remove_bad=False):
    ptp = find_maximal_inescapable_trap(ptp, move_bound, remove_bad)
    print(f"Found inescapable Trap {len(ptp)}")
    keep_looping = True
    while keep_looping:
        keep_looping = False
        bad_squares = get_squares_allowing_forced_repetition(ptp, move_bound)
        if len(bad_squares) > 0:

            keep_looping = True
            ptp.difference_update(bad_squares)
    print(f"Found inescapable tempo gaining Trap {len(ptp)}")
    return ptp


def print_heatmap(trap):
    coords = []
    label_maps = []
    min_x = min(x for board in trap for (x, y) in board)
    min_y = min(x for board in trap for (x, y) in board)
    gap_x = max(x for board in trap for (x, y) in board) - min_x + 1
    gap_y = max(x for board in trap for (x, y) in board) - min_y + 1
    # TODO support more than 2 White pieces, more than 26 positions for first piece and arbitrary black king index
    labels = [None, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", [str(s) for s in range(500)]]
    trap_map = [['' for _ in range(gap_y)] for _ in range(gap_x)]
    for i in range(len(PIECES)):
        label_maps += [[['' for _ in range(gap_y)] for _ in range(gap_x)]]

        clist = list(sorted(list(set(board[i] for board in trap))))
        coords += [clist]
        if i != BLACK_KING_INDEX:
            for index, c in enumerate(clist):
                # print(i, index)
                label_maps[i][c[0]-min_x][c[1]-min_y] = labels[i][index]
    for p in trap:
        for i in range(len(PIECES)):
            if i != BLACK_KING_INDEX:
                trap_map[p[BLACK_KING_INDEX][0]-min_x][p[BLACK_KING_INDEX][1]-min_y] += labels[i][coords[i].index(p[i])]
        trap_map[p[BLACK_KING_INDEX][0]-min_x][p[BLACK_KING_INDEX][1]-min_y] += " "
    for i in range(len(PIECES)):
        print(PIECES[i].symbol)
        p1 = PrettyTable()
        p1.header = False
        if i != BLACK_KING_INDEX:
            for r in label_maps[i]:
                p1.add_row(r, divider=True)
            print(p1)
    p1 = PrettyTable()
    p1.header = False
    for r in trap_map:
        p1.add_row(r, divider=True)
    print(p1)


def find_coordinate_reduced_tempo_trap(ptp, move_bound, remove_bad=True):
    ptp = find_maximal_inescapable_tempo_gaining_trap(ptp, move_bound, remove_bad=remove_bad)
    print(f"Maximal tempo gaining trap is size {len(ptp)}")
    reduced_trap = copy.copy(ptp)  # TODO this avoids side effects, but is that right to do?
    white_coords = list(set(tuple(a for i, a in enumerate(board) if not PIECES[i].is_black) for board in ptp))
    black_coords = list(set(tuple(a for i, a in enumerate(board) if PIECES[i].is_black) for board in ptp))
    white_coords.sort(key=lambda x: -sum(l1_norm(y) for y in x))
    black_coords.sort(key=lambda x: -sum(l1_norm(y) for y in x))

    for c in black_coords:
        to_remove = {board for board in ptp if tuple(a for i, a in enumerate(board) if PIECES[i].is_black) == c}
        a = len(reduced_trap)
        reduced_trap.difference_update(to_remove)
        if a != len(reduced_trap):
            test = find_maximal_inescapable_tempo_gaining_trap(reduced_trap, move_bound)
            if len(test) > 0:
                print(f"Removed a black coordinate {c} and reduced trap to {len(test)}")
                reduced_trap = test
            else:
                reduced_trap.update(to_remove)

    for c in white_coords:
        to_remove = {board for board in ptp if tuple(a for i, a in enumerate(board) if not PIECES[i].is_black) == c}
        a = len(reduced_trap)
        reduced_trap.difference_update(to_remove)
        if a != len(reduced_trap):
            test = find_maximal_inescapable_tempo_gaining_trap(reduced_trap, move_bound)
            if len(test) > 0:
                print(f"Removed a white coordinate {c} and reduced trap to {len(test)}")
                reduced_trap = test
            else:
                reduced_trap.update(to_remove)
    print("Now reducing remaining positions")
    reduced_trap = find_reduced_tempo_trap(reduced_trap, move_bound)
    reduced_white_coords = set(tuple(a for i, a in enumerate(board) if not PIECES[i].is_black) for board in reduced_trap)
    reduced_black_coords = set(tuple(a for i, a in enumerate(board) if PIECES[i].is_black) for board in reduced_trap)

    print(f"Reduced trap to just {len(reduced_white_coords)} White positions and {len(reduced_black_coords)} Black positions")
    for c in reduced_black_coords:
        print(f"{c} : {[tuple(a for i, a in enumerate(board) if not PIECES[i].is_black) for board in reduced_trap if tuple(a for i, a in enumerate(board) if PIECES[i].is_black) == c]}")
    print("="*20)
    print(reduced_trap)
    print("="*20)
    return reduced_trap


def find_reduced_tempo_trap(ptp, move_bound, remove_bad=False):
    ptp = find_maximal_inescapable_tempo_gaining_trap(ptp, move_bound, remove_bad)
    reduced_trap = copy.copy(ptp)
    for p in ptp:
        if p in reduced_trap:
            reduced_trap.difference_update({p})
            test = find_maximal_inescapable_tempo_gaining_trap(reduced_trap, move_bound)
            if len(test) > 0:
                print(f"Found inescapable tempo gaining trap of size {len(test)}")
                reduced_trap = test
            else:
                reduced_trap.add(p)
    return reduced_trap


def get_squares_allowing_forced_repetition(trap, move_bound):
    btm_trap = copy.copy(trap)
    wtm_trap = set()
    for p in btm_trap:
        wtm_trap.update(get_black_moves(p, move_bound))
    white_passable_pos = wtm_trap.intersection(btm_trap)

    keep_looping = True
    while keep_looping:
        keep_looping = False
        pass_forced = set()
        for p in btm_trap:
            if all(f in white_passable_pos for f in get_black_moves(p, move_bound)):
                pass_forced.add(p)
                keep_looping = True
        btm_trap.difference_update(pass_forced)
        for p in pass_forced:
            white_passable_pos.update(wtm_trap.intersection(get_white_preimages(p, move_bound, white_pass=True)))
    return btm_trap


def translate_board(board, v):
    return tuple((b[0] + v[0], b[1] + v[1]) for b in board)


def l1_norm(a):
    return abs(a[0]) + abs(a[1])


def l2_norm(a):
    return math.sqrt(a[0]**2 + a[1]**2)


def inf_norm(a):
    return max(abs(a[0]), abs(a[1]))


def knight_norm(a):
    r, s = abs(a[0]), abs(a[1])
    return max(r, s)/2 if 2*min(r, s) < max(r, s) else (r+s)/3


def dist(a, b, norm):
    return norm((a[0]-b[0], a[1]-b[1]))


def play_vs_trap(trap, move_bound,box_size=10):
    if len(trap) == 0:
        print("YOU WIN!")
    position = random.choice(list(trap))
    input_to_vec = {"w": (0,-1),
                    "e": (0,1),
                    "n": (-1,0),
                    "s": (1,0),
                    "ne": (-1, 1),
                    "nw": (-1, -1),
                    "se": (1, 1),
                    "sw": (1, -1)
                    }
    seen_since_last_pass = set()
    other_moves = trap
    while True:
        print_position(position, coords_to_include=((box_size,0),(-box_size,0),(0,box_size),(0,-box_size)), show_white_threats=True)
        inp = input("Enter direction (NESW): ").lower()

        if inp == "restart":
            position = random.choice(list(trap))
        elif inp == "reroll":
            position = random.choice(list(other_moves))
        elif inp == "exit":
            return
        elif inp in input_to_vec:
            possibilities = get_black_moves(position, move_bound=move_bound)
            direction = input_to_vec[inp]
            v = move(position[BLACK_KING_INDEX], direction)
            found_move = False
            for p in possibilities:
                if p[BLACK_KING_INDEX] == v:
                    assert not found_move
                    print_position(p, coords_to_include=((box_size,0),(-box_size,0),(0,box_size),(0,-box_size)), show_white_threats=True)
                    found_move = True
                    if p in trap:
                        print("PASS!")
                        position = p
                        seen_since_last_pass = set()
                    else:
                        options = get_white_moves(p, move_bound=move_bound, white_pass=False).intersection(trap)
                        non_repeat_options = options - seen_since_last_pass
                        other_moves = options if len(non_repeat_options) == 0 else non_repeat_options
                        if len(options) == 0:
                            print("YOU WIN! \n" * 100)
                            assert False
                        print(f"Choosing from {len(options)} options {len(non_repeat_options)} of which don't repeat")
                        position = random.choice(list(options)) if len(non_repeat_options) == 0 else random.choice(list(non_repeat_options))
                        seen_since_last_pass.add(position)
                    # time.sleep(1)
            if not found_move:
                print("Not a legal move!")
        else:
            try:
                inp = inp.split(',')
                board = tuple((int(inp[2*k]), int(inp[2*k+1])) for k in range(len(PIECES)))
                if board not in trap:
                    print("Not in trap!")
                else:
                    position = board
            except:
                print("Not a legal command!")
                print("Type restart to get a new position, reroll to see other White responses to your last move, or exit to stop the game.")
                print("Legal directions are",*list(input_to_vec.keys()))


def get_potential_ArHa_traps(n):
    coordinates1 = [(n, 0), (-n, 0), (0, n), (0, -n)]
    coordinates2 = [(a, b) for a in range(-n-1, n+2) for b in range(-n, n+2) if l1_norm((a, b)) <= n+7]
    king_coordinates = [(a, b) for a in range(-n, n+1) for b in range(-n, n+1) if l1_norm((a, b)) < n]
    potential_trap_positions = {(a, b, c) for a in king_coordinates for b in coordinates1 for c in coordinates2}
    return potential_trap_positions


def get_potential_nbb_traps(n, knight_bound, edge_size):
    print(f"{n} {knight_bound} {edge_size}")
    if CORNER_MODE:
        print(f"Warning idiot this is the version with the black king clamped above y = {CORNER_BOUND}")

    special_bishops = [(1 - n, 0), (n - 1, 0), (0, 1 - n), (0, n - 1)]
    bishop_1 = [(a, b) for a in range(-n, n + 1) for b in range(-n, n + 1) if a + b in {n + 1, -n-1}] + special_bishops
    bishop_2 = [(a, b) for a in range(-n, n + 1) for b in range(-n, n + 1) if a - b in {n + 1, -n-1}] + special_bishops

    all_bishops = [(a, b) for a in bishop_1 for b in bishop_2 if len((set(a) | set(b)).intersection({n, -n, n - 1, 1 - n})) > 0]
    bishops_corner = [(a, b) for a in bishop_1 for b in bishop_2 if len(set(a).intersection({n, -n})) > 0 and len(set(b).intersection({n, -n})) > 0]

    # print_all_coords([p[0] for p in all_bishops])
    # print_all_coords([p[1] for p in all_bishops])
    # print_all_coords([p[0] for p in bishops_corner] )
    # print_all_coords([p[1] for p in bishops_corner] )

    knight = [(a, b) for a in range(CORNER_BOUND if CORNER_MODE else -n-3, n+4) for b in range(-n-3, n+4) if l1_norm((a, b)) <= n + 3]
    # print(knight)
    # print_all_coords(knight)

    edge_king_coordinates = [(a, b) for a in range(CORNER_BOUND if CORNER_MODE else -n, n + 1) for b in range(-n, n + 1) if n - edge_size <= l1_norm((a, b)) <= n]
    # print_all_coords(edge_king_coordinates)
    center_king_coordinates = [(a, b) for a in range(CORNER_BOUND if CORNER_MODE else -n, n + 1) for b in range(-n, n + 1) if l1_norm((a, b)) < n-edge_size]
    # print_all_coords(center_king_coordinates)
    # print_all_coords([(x, y) for x in range(-n, n) for y in range(-n, n) if dist((x, y), (0, 0), knight_norm) <= knight_bound])
    potential_trap_positions = {(a, b) + c for a in center_king_coordinates for b in knight for c in bishops_corner if dist(a, b, knight_norm) < knight_bound}
    potential_trap_positions.update({(a, b) + c for a in edge_king_coordinates for b in knight for c in all_bishops if dist(a, b, knight_norm) < knight_bound})

    return potential_trap_positions


def save_trap(trap, path):
    with open(path, 'w') as f:
        f.write(str(trap))


def load_trap(path):
    with open(path, 'r') as f:
        return ast.literal_eval(f.read())

# def get_potential_nbb_corners(n):
#     bishop_1 = [(a, b) for a in range(0, n + 1) for b in range(-n, n + 1) if a + b in {n + 1, -n - 1}] + [(1-n, 0),(n - 1, 0),(0, 1 - n),(0, n - 1)]
#     bishop_2 = [(a, b) for a in range(0, n + 1) for b in range(-n, n + 1) if a - b in {n + 1, -n - 1}] + [(1-n, 0),(n - 1, 0),(0, 1 - n),(0, n - 1)]
#
#     all_bishops = [(a, b) for a in bishop_1 for b in bishop_2 if len((set(a) | set(b)).intersection({n, -n, n - 1, 1 - n})) > 0]
#     bishops_corner = [(a, b) for a in bishop_1 for b in bishop_2 if a[0] == b[0] and a[0] in {n, -n} or a[1] == b[1] and a[1] in {n, -n}]
#
#     knight = [(a, b) for a in range(3, n + 7) for b in range(-2 * n, 2 * n + 1) if l1_norm((a, b)) <= n + 3]
#
#     edge_king_coordinates = [(a, b) for a in range(1, n + 2) for b in range(-n - 1, n + 2) if n - 3 <= l1_norm((a, b)) <= n]
#     center_king_coordinates = [(a, b) for a in range(1, n + 2) for b in range(-n - 1, n + 2) if l1_norm((a, b)) < n - 3]
#
#     potential_trap_positions = {(a, b) + c for a in center_king_coordinates for b in knight for c in bishops_corner if dist(a, b, knight_norm) < 2.5}
#     potential_trap_positions.update({(a, b) + c for a in edge_king_coordinates for b in knight for c in all_bishops if dist(a, b, knight_norm) < 2.5})
#
#     return potential_trap_positions


PIECES = [BLACK_KING] + [KNIGHT, BISHOP, BISHOP]  # first entry must always be Black king
BLACK_KING_INDEX = 0
WHITE_KING_INDEX = None
CORNER_MODE = False
CORNER_BOUND = None
if __name__ == "__main__":
    print(*[pp.symbol for pp in PIECES])
    # play_vs_trap(load_trap("kNBB_20_3_2.5_23.txt"), 23, box_size=24)
    n = 20
    move_bound = n+2
    edge_size = 3
    knight_bound = 2.5
    print(n)
    trap = load_trap("kNBB_20_3_2.5_23.txt")
    play_vs_trap(trap, move_bound, box_size=(n+1))

    # potential_trap_positions = get_potential_nbb_traps(n, edge_size=edge_size, knight_bound=knight_bound)
    # print(len(potential_trap_positions))
    # print("="*20)
    # trap = find_maximal_inescapable_tempo_gaining_trap(potential_trap_positions, move_bound, remove_bad=True)
    # if len(trap) > 0:
    #     save_trap(trap, "".join(pp.symbol for pp in PIECES)+f'_{n}_{edge_size}_{knight_bound}_{move_bound}_coord_reduced.txt')


    # print_heatmap(trap)



    # Archbishop Hawk Trap!


    # print("="*20)
    # n=6
    # PIECES = [BLACK_KING] + [ARCHBISHOP, HAWK]
    # trap = find_coordinate_reduced_tempo_trap(get_potential_ArHa_traps(n), n, remove_bad=True)
    # # trap = find_maximal_inescapable_tempo_gaining_trap(get_potential_ArHa_traps(n), n, remove_bad=True)
    # # trap = load_trap("".join([pp.symbol for pp in PIECES]) + f'_{n}.txt')
    # print_heatmap(trap)
    # play_vs_trap(trap, n, box_size=(n + 1))
    # save_trap(trap, "".join([pp.symbol for pp in PIECES]) + f'_{n}_coord_reduced.txt')

