
export PYFLEXROOT=${PWD}/PyFlex
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export PYTHONPATH=${PWD}:$PYTHONPATH

username="robot_dev4"
source "/home/${username}/anaconda3/etc/profile.d/conda.sh"
conda activate softgym
