import os

from gym_trading_env.renderer import Renderer

PROJ_PATH = os.path.split(os.path.abspath(''))[0]
RENDER_LOGS_PATH = os.path.join(PROJ_PATH, 'render_logs')

renderer = Renderer(render_logs_dir=os.path.join(RENDER_LOGS_PATH, 'trading_indicators', 'PPO_ws0'))
renderer.run()
