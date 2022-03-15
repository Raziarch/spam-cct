import os
import sys

# Adding Lib Path
lib_path = os.path.abspath(os.path.join('lib'))
sys.path.append(lib_path)

# Adding Models Path
models_path = os.path.abspath(os.path.join('models'))
sys.path.append(models_path)