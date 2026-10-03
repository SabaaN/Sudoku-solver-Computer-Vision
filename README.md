# Sudoku Solver

An image-based Sudoku solver for my computer vision practice and learning. It detects a Sudoku grid in an image, recognizes the given digits with a small convolutional neural network, solves the puzzle with backtracking, and overlays the answers on the original image.

## Project files

- `sudoku_solver.ipynb` — notebook containing digit-model training, image processing, recognition, solving, and visualization.
- `sudoku.png` — image asset included with the project.

## Run in Google Colab

1. Upload `sudoku_solver.ipynb` to [Google Colab](https://colab.research.google.com/) and open it.
2. Run the notebook cells from top to bottom. The first training cell generates synthetic printed digits using available TrueType fonts and trains a CNN. This takes time and is repeated each time the notebook is run; the trained model is kept in memory for the current session.
3. Run the final cell and upload a clear image of a Sudoku puzzle when prompted.
4. The notebook prints the recognized starting board and displays the image with the missing digits filled in.

The notebook imports OpenCV, NumPy, Matplotlib, Pillow, TensorFlow/Keras, and Google Colab helpers. Colab provides the `google.colab` modules used for image upload and display. The first cell trains on 2,000 generated examples for each digit 1–9 for 8 epochs; it does not load a saved model or use an external MNIST download.

## Image requirements and options

The final function is `solve_sudoku_image(img, already_cropped=True)`. By default, the uploaded image is treated as a close, cropped view of the grid. For a photo where the grid sits within a larger scene, call it with `already_cropped=False` so the notebook attempts to locate and perspective-correct the grid:

```python
result = solve_sudoku_image(img, already_cropped=False)
cv2_imshow(result)
```

Use a well-lit, sharp image with the full 9×9 grid visible. Recognition errors can produce an invalid or unsolvable board; inspect the printed recognized board if the solver reports that it found no solution.

## How it works

1. Generates synthetic examples of printed digits 1–9 and trains a CNN classifier on 28×28 grayscale images.
2. Optionally detects the grid quadrilateral, then warps the grid to a square image.
3. Splits the image into 81 cells, filters out empty cells, and classifies the visible digits.
4. Solves the recognized board using recursive backtracking with row, column, and 3×3 box checks.
5. Draws answers only in cells recognized as empty and maps the overlay back onto the uploaded image.

The model classifies digits, while empty cells are identified by a lack of ink. The solver expects a standard 9×9 Sudoku.
