import numpy as np



kernels = {
    'guassian blur 3x3' : (1 / 16.0) * np.array([[1, 2, 1], # or blur 3x3
                                                 [2, 4, 2],
                                                 [1, 2, 1]]),

    'guassian blur 5x5' : (1 / 256.0) * np.array([[1, 4,  6,  4,  1], # or blur 5x5
                                                  [4, 16, 24, 16, 4],
                                                  [6, 24, 36, 24, 6],
                                                  [4, 16, 24, 16, 4],
                                                  [1, 4,  6,  4,  1]]),      
    'top sobel 3x3' : np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]),
    
    'top sobel 5x5' : np.array([[-1, -1, -2, -1, -1],
                                [-1, -1, -2, -1, -1],
                                [ 0,  0,  0,  0,  0],
                                [ 1,  1,  2,  1,  1],
                                [ 1,  1,  2,  1,  1]]),
        
    'right sobel 3x3' : np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]]),

    'right sobel 5x5' : np.array([[1, 1, 0, -1, -1],
                                  [2, 2, 0, -2, -2],
                                  [3, 3, 0, -3, -3],
                                  [2, 2, 0, -2, -2],
                                  [1, 1, 0, -1, -1]]),
        
    'left sobel 3x3' : np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]),

    'left sobel 5x5' : np.array([[-1, -1, 0, 1, 1],
                                 [-2, -2, 0, 2, 2],
                                 [-3, -3, 0, 3, 3],
                                 [-2, -2, 0, 2, 2],
                                 [-1, -1, 0, 1, 1]]),

    'bottom sobel 3x3' : np.array([[ 1,  2,  1],
                                   [ 0,  0,  0],
                                   [-1, -2, -1]]),
    
    'bottom sobel 5x5' : np.array([[ 1,  1,  2,  1,  1],
                                   [ 1,  1,  2,  1,  1],
                                   [ 0,  0,  0,  0,  0],
                                   [-1, -1, -2, -1, -1],
                                   [-1, -1, -2, -1, -1]]),

    'emboss 3x3' : np.array([[-2, -1, 0],
                             [-1,  1, 1],
                             [ 0,  1, 2]]),
    
    'emboss 5x5' : np.array([[-1, -1, -1, -1, 0],
                             [-1, -1, -1,  0, 1],
                             [-1, -1,  0,  1, 1],
                             [-1,  0,  1,  1, 1],
                             [ 0,  1,  1,  1, 2]]),

    'identity 3x3' : np.array([[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]),

    'identity 5x5' : np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]),

    'outline 3x3' : np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]]),

    'outline 5x5' : np.array([[0,  0, -1,  0, 0],
                              [0, -1, -1, -1, 0],
                              [1, -1,  8, -1, 1],
                              [0, -1, -1, -1, 0],
                              [0,  0, -1,  0, 0]]),
        
    'sharpen 3x3' : np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]]),
    
    'sharpen 5x5' : np.array([[ 0,  0,  0,  0,  0],
		              [ 0,  0, -1,  0,  0],
		              [ 0, -1,  5, -1,  0],
		              [ 0,  0, -1,  0,  0],
		              [ 0,  0,  0,  0,  0]]),

    'vertical edge 3x3' : np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]]),

    'vertical edge 5x5' : np.array([[-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1]]),
        
    'horizontal edge 3x3' : np.array([[ 1,  1, 1],
                                      [ 0,  0, 0],
                                      [-1, -1,-1]]),

    'horizontal edge 5x5' : np.array([[ 1,  1,  1,  1,  1],
                                      [ 1,  1,  1,  1,  1],
                                      [ 0,  0,  0,  0,  0],
                                      [-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]),
        
    'box blur 3x3' : (1 / 9.0) * np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]]),
    
    'box blur 5x5' : (1 / 25.0) * np.array([[1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1]]),
    'laplacian 3x3' : np.array([[0,  1, 0],
                                [1, -4, 1],
                                [0,  1, 0]]),

    'laplacian 5x5' : np.array([[ 0,  0, -1,   0,  0],
                                [ 0, -1, -2,  -1,  0],
                                [-1, -2,  16, -2, -1],
                                [ 0, -1, -2,  -1,  0], 
                                [ 0,  0, -1,   0,  0]]),
        
    'prewitt horizontal edge 3x3' : np.array([[-1, -1, -1],
                                              [ 0,  0,  0],
                                              [ 1,  1,  1]]),

    'prewitt horizontal edge 5x5' : np.array([[-1, -1, -1, -1, -1],
                                              [-1, -1, -1, -1, -1],
                                              [0,   0,  0,  0,  0],
                                              [1,   1,  1,  1,  1],
                                              [1,   1,  1,  1,  1]]),
        
    'prewitt vertical edge 3x3' : np.array([[-1, 0, 1],
                                            [-1, 0, 1],
                                            [-1, 0, 1]]),

    'prewitt vertical edge 5x5' : np.array([[-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1]]),
        
    'high-pass filter 3x3' : np.array([[-1, -1, -1],
                                       [-1,  8, -1],
                                       [-1, -1, -1]]),

    'high-pass filter 5x5' : np.array([[-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, 24, -1, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1]]),

    'unsharp masking 3x3' : np.array([[ -1, -1, -1],
                                      [ -1,  9, -1],
                                      [ -1, -1, -1]]),

    'unsharp masking 5x5' : (-1 / 256.0) * np.array([[1, 4,   6,   4,  1],
                                                     [4, 16,  24,  16, 4],
                                                     [6, 24, -476, 24, 6],
                                                     [4, 16,  24,  16, 4],
                                                     [1, 4,   6,   4,  1]]),
    'dilation 3x3' : np.array([[ 0, -1,  0],
                               [-1,  8, -1],
                               [ 0, -1,  0]]),

    'dilation 5x5' : np.array([[ 0,  0, -1,  0,  0],
		               [ 0,  0, -1,  0,  0],
		               [ 1, -1,  8, -1,  1],
		               [ 0,  0, -1,  0,  0],
		               [ 0,  0, -1,  0,  0]]),
        
    'soften 3x3' : (1 / 16.0) * np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]]),
    
    'soften 5x5' : (1 / 256.0) * np.array([[1,  4,  6,  4,  1],
                                           [4, 16, 24, 16,  4],
                                           [6, 24, 36, 24,  6],
                                           [4, 16, 24, 16,  4],
                                           [1,  4,  6,  4,  1]]),
        
    'scharr vertical edge 3x3' : np.array([[-3,  0, 3],
                                           [-10, 0, 10],
                                           [-3,  0, 3]]),

    'scharr vertical edge 5x5' : np.array([[-1, -3,  0, 3,  1],
                                           [-3, -10, 0, 10, 3],
                                           [-4, -12, 0, 12, 4],
                                           [-3, -10, 0, 10, 3],
                                           [-1, -3,  0, 3,  1]]),
         
    'scharr horizontal edge 3x3' : np.array([[-3, -10,  -3],
                                             [ 0,  0,    0],
                                             [ 3,  10,   3]]),

    'scharr horizontal edge 5x5' : np.array([[-1, -3,  -4,   -3, -1],
                                             [-3, -10, -12, -10, -3],
                                             [ 0,  0,   0,   0,   0],
                                             [ 3,  10,  12,  10,  3],
                                             [ 1,  3,   4,   3,   1]]),

    'motion blur 3x3' : (1 / 3.0) * np.array([[1,  0,   0],
                                              [0,  1,   0],
                                              [0,  0,   1]]),
    
    'motion blur 5x5' : (1 / 5.0) * np.array([[1,  0, 0, 0,  0],
                                              [0,  1, 0, 0,  0],
                                              [0,  0, 1, 0,  0],
                                              [0,  0, 0, 1,  0],
                                              [0,  0, 0, 0,  1]]),

    'robert horizontal edge 3x3': np.array([[0, 0,  0],
                                            [0, 1,  0],
                                            [0, 0, -1]]),

    'robert horizontal edge 5x5': np.array([[0, 0, 0,  0, 0],
                                            [0, 0, 0,  0, 0],
                                            [0, 0, 1,  0, 0],
                                            [0, 0, 0, -1, 0],
                                            [0, 0, 0,  0, 0]]),

    'robert vertical edge 3x3': np.array([[0,  0, 0],
                                          [0,  0, 1],
                                          [0, -1, 0]]),

    'robert vertical edge 5x5': np.array([[0, 0,  0, 0, 0],
                                          [0, 0,  0, 0, 0],
                                          [0, 0,  0, 1, 0],
                                          [0, 0, -1, 0, 0],
                                          [0, 0,  0, 0, 0]]),

    'ridge detection1 3x3' : np.array([[0, -1, 0], # or edge detection1 3x3
                                       [-1, 4,-1],
                                       [0, -1, 0]]),

    'ridge detection1 5x5' : np.array([[ 0,  0, -1,  0,  0], # or edge detection1 5x5
                                       [ 0,  0, -1,  0,  0],
                                       [ 1, -1,  4, -1,  1],
                                       [ 0,  0, -1,  0,  0],
                                       [ 0,  0, -1,  0,  0]]),

    'ridge detection2 3x3' : np.array([[-1, -1, -1], # or edge detection2 3x3
                                       [-1,  8, -1],
                                       [-1, -1, -1]]),

    'ridge detection2 5x5' : np.array([[ 0,  0, -1,  0,  0], # or edge detection2 5x5
                                       [ 0,  0, -1,  0,  0],
                                       [-1, -1,  8, -1, -1],
                                       [ 0,  0, -1,  0,  0],
                                       [ 0,  0, -1,  0,  0]])
    }