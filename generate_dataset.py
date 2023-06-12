import cv2
import glob
import numpy as np
import os
import shogi
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from typing import List
from typing import Tuple
from webdriver_manager.chrome import ChromeDriverManager


def generate_dataset(n: int = 10) -> None:
    """
    Generate the shogi boards and tiles for learning.
    """
    # load cookies
    cookies_path = 'data/cookies.csv'
    boards_path = 'data/boards/'
    tiles_path = 'data/tiles/'
    try:
        with open(cookies_path, 'r') as file:
            for line in file:
                for _ in range(n):  # n screenshots with the board/piece theme
                    sfen = generate_random_sfen()
                    shogi_board = shogi.Board(sfen)
                    cookie = line.strip()
                    board_file_name = sfen.split()[0].replace(
                        '/', '-') + '.png'
                    take_board_screenshot(
                        sfen, cookie, boards_path, board_file_name)
                    v_lines, h_lines = generate_lines(
                        boards_path, board_file_name)
                    generate_board_tiles(
                        shogi_board, v_lines, h_lines, boards_path, tiles_path, board_file_name)
    except FileNotFoundError:
        print('File not found.')
    except IOError:
        print('IOError occurred.')


def generate_random_sfen() -> str:
    """
    Play a random shogi game and return its board SFEN representation.
    """
    board = shogi.Board()

    while not board.is_game_over():
        legal_moves = list(board.generate_legal_moves())
        move = np.random.choice(legal_moves)  # pick a random move
        board.push(move)  # play the move

    return board.sfen()


def take_board_screenshot(sfen: str, cookie_value: str, boards_path: str, board_file_name: str) -> None:
    """
    Take a screenshot of the shogi board and save it.
    """
    # set up the driver
    url = 'https://lishogi.org/editor/' + sfen
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    driver.add_cookie({'name': 'lila2', 'value': cookie_value})
    driver.get(url)

    # save the screenshot
    driver.save_screenshot(os.path.join(boards_path, board_file_name))

    # crop the screenshot and save it
    image = Image.open(os.path.join(boards_path, board_file_name))
    cropped_image = image.crop((209, 200, 542, 563))  # crop coordinates
    cropped_image.save(os.path.join(boards_path, board_file_name))

    # close the window
    driver.quit()


def generate_lines(boards_path: str, board_file_name: str) -> Tuple[int, int]:
    """
    Given a shogi board, detect the vertical and horizontal lines. Return the list of vertical and horizontal lines as a tuple.
    Detects lines with an accuracy of 86%.
    """
    # load the image as grayscale
    image = cv2.imread(boards_path + board_file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=200, minLineLength=200, maxLineGap=10)  # tuned for shogi boards

    v_lines, h_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:  # vertical line
            v_lines.append(x1)
        elif y1 == y2:  # horizontal line
            h_lines.append(y1)

    # remove clustering from line detection
    v_lines = cluster_lines(v_lines)
    h_lines = cluster_lines(h_lines)

    return v_lines, h_lines


def cluster_lines(lines: List[int], line_gap: int = 5) -> List[int]:
    """
    Cluster lines if they are closer than the line gap/threshold.
    """
    if len(lines) <= 0:
        return []
    lines.sort()
    clustered_lines = [lines[0]]
    for i in range(1, len(lines)):
        # more than {error}px gap between the lines
        if lines[i] - lines[i-1] > line_gap:
            clustered_lines.append(lines[i])
    return clustered_lines


def generate_board_tiles(shogi_board: shogi.Board, v_lines: List[int], h_lines: List[int], boards_path: str, tiles_path: str, board_file_name: str) -> None:
    """
    Crop the board and save the screenshot of the tile in its respective directory.
    """

    # can create more sophisticated line verification
    if len(v_lines) != 10 or len(h_lines) != 10:
        return

    for rank in range(len(v_lines) - 1):
        y1 = h_lines[rank]
        y2 = h_lines[rank + 1]
        for file in range(len(h_lines) - 1):
            x1 = v_lines[file]
            x2 = v_lines[file + 1]

            piece = str(shogi_board.piece_at(rank * 9 + file))  # indexed board
            color = 'b' if piece.isupper() else 'w'
            color = '' if piece == 'None' else color  # account for empty tile

            tile_file_path = tiles_path + color + piece.lower() + '/'
            # matches any file in the directory
            num_files = len(glob.glob(tile_file_path + '/*'))

            image = Image.open(boards_path + board_file_name)
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image = cropped_image.convert('L')  # convert to grayscale
            cropped_image = cropped_image.resize((32, 32))  # resize to 32 x 32
            cropped_image.save(tile_file_path + str(num_files) + '.png')


# generate the dataset
if __name__ == '__main__':
    generate_dataset()  # 10 boards of each board/piece theme (116640 tiles)
