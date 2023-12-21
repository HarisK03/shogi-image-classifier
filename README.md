# Shogi Image Classifier #

## About the Project ##

This project is inspired by the [TensorFlow Chessbot](https://github.com/Elucidation/tensorflow_chessbot).

The aim of the project is to take an image of a Shogi board with various pieces and return the Shogi Forsyth-Edwards Notation (SFEN) representation of the board.

## Defining SFEN ##

Given any Shogi board, the board can be represented as an SFEN string. SFEN is similar to the FEN representation where each piece has a corresponding symbol.

| Piece Name     | SFEN Notation | Promoted Notation |
| -------------- | ------------- | ----------------- | 
| King           | K             | None              |
| Rook           | R             | +R                |
| Bishop         | B             | +B                |
| Gold General   | G             | None              |
| Silver General | S             | +S                |
| Knight         | N             | +N                |
| Lance          | L             | +L                |
| Pawn           | P             | +P                |

Tiles that do not contain pieces are represented by a number. For example, 2 consecutive tiles along a rank will be represented by the number 2. Ranks of nine tiles are separated by a / character. 

An example of a valid SFEN string is lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL and the shogi board representation is found below.

![alt text](https://github.com/HarisK03/shogi-image-classifier/blob/readme/shogi.png)

## Generating the Dataset ##

The steps below outline the process of generating the dataset.

1. The [python-shogi](https://github.com/gunyarakun/python-shogi) is used to make moves from the starting position until a game over state is reached. A random SFEN string is generated.
2. Using Selenium and loading a cookie with combinations of board theme and piece sets, a screenshot is taken of the board. Ten screenshots of different positions are collected with a board theme and piece set combination.
3. Using OpenCV and Hough Transform, the board lines are detected and individual tile screenshots are saved to their corresponding data directories as 32x32 grayscale images.

## Training the Model ##

Using TensorFlow, the model is trained on the training dataset and validation dataset. The model that is defined can be found [here](https://github.com/HarisK03/shogi-image-classifier/blob/e89a0324d932e037243f797546743b6debb3c60e/train_model.py#L54C3-L54C3).

##  Testing and Results ##

To obtain the SFEN representation of a shogi board image, place the screenshot of the board into the boards path root directory and update the board_file_name in predict.py. Run predict.py and the resulting SFEN will be printed in the console. 

The **tile accuracy** is calculated at 99.94% and the **board line detection** is calculated at 86.43%.

## Contact ##

Haris Kamal - HarisKamal03@gmail.com

Project Link - https://github.com/HarisK03/shogi-image-classifier/
