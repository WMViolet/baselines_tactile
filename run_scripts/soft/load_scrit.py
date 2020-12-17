try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

fullpath = '~/try/tactile-baselines/run_scripts/soft/soft_box.xml'
model = mujoco_py.load_model_from_path(fullpath)
