import numpy as np
import os
import tensorflow as tf
from generate_dataset import generate_lines
from keras.utils import load_img, img_to_array
from PIL import Image
from train_model import generate_image_data_generators


def shogi_board_to_fen(model: tf.keras.Model, boards_path: str, tiles_path: str, board_file_name: str) -> str:
    """
    Return the SFEN representation from the board image.
    """
    v_lines, h_lines = generate_lines(boards_path, board_file_name)

    if len(v_lines) != 10 or len(h_lines) != 10:
        print(v_lines, h_lines)
        return

    train_data = generate_image_data_generators(32, 32, 32)[0]

    # get the class labels
    class_labels = train_data.class_indices
    class_labels = {v: k for k, v in class_labels.items()}

    sfen = ''
    none_count = 0
    tiles_processed = 0

    for rank in range(len(v_lines) - 1):
        y1 = h_lines[rank]
        y2 = h_lines[rank+1]
        for file in range(len(h_lines) - 1):
            x1 = v_lines[file]
            x2 = v_lines[file+1]

            image = Image.open(boards_path + board_file_name)
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image = cropped_image.convert('L')  # convert to grayscale
            cropped_image = cropped_image.resize((32, 32))  # resize to 32 x 32
            cropped_image.save(tiles_path + 'current_tile.png')  # temp tile

            image_path = tiles_path + 'current_tile.png'
            image = load_img(image_path, target_size=(32, 32))
            image_array = img_to_array(image)
            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]

            if tiles_processed > 0 and tiles_processed % 9 == 0:
                sfen += '/'

            if class_labels[predicted_class_index] == 'none':
                none_count += 1
            else:
                if none_count > 0:
                    sfen += str(none_count)
                    none_count = 0
                upper = True if class_labels[predicted_class_index][0] == 'b' else False
                piece_type = class_labels[predicted_class_index][1:]
                sfen += piece_type.upper() if upper else piece_type

            predicted_class = class_labels[predicted_class_index]
            print('Predicted Class:', predicted_class,
                  'Confidence:', confidence)

            tiles_processed += 1

        if none_count > 0:
            sfen += str(none_count)
            none_count = 0

    if none_count > 0:
        sfen += str(none_count)

    os.remove(tiles_path + 'current_tile.png')  # remove the temporary tile

    return sfen


# get the SFEN
if __name__ == '__main__':
    # Load the trained model
    model = tf.keras.models.load_model('shogi_board_to_sfen_model.h5')
    boards_path = 'data/boards/'
    tiles_path = 'data/tiles/'
    board_file_name = ''  # replace with the board image file name

    sfen = shogi_board_to_fen(model, boards_path, tiles_path, board_file_name)
    print('SFEN:', sfen)  # print sfen to stdout
