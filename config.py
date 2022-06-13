r'''
    Contiene variables de entorno para el proyecto.
    DATASET_DIR: Directorio donde se encuentran los datasets.
    RESULT_DIR: Directorio donde se guardan los resultados.

    También la configuración de los experimentos!
'''
#DATASET_DIR = r'C:\Users\mcastro\Documents\MCastro\2_Codigo\0_DATASETS'
#RESULT_DIR = r'C:\Users\mcastro\Documents\MCastro\2_Codigo\DermaModelOptimization\results'
DATASET_DIR = 'data' # in notebooks
RESULT_DIR = '..\\results' # it's another folder
RANDOM_SEED = 123

# Experiments configuration
MbV2 = {'inverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ],
     'attention': False,
    }

MbV2_CA = {'inverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ],
     'attention': True,
    }

MbV2_CA_Reduced = {'inverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 1, 2],
            [4, 32, 1, 2],
            [4, 64, 1, 2],
            [4, 96, 1, 1],
            [4, 160, 1, 2],
            [4, 320, 1, 1],
        ],
     'attention': True,
    }

experiment = {'MbV2': MbV2, 'MbV2_CA': MbV2_CA, 'MbV2_CA_Reduced': MbV2_CA_Reduced}

