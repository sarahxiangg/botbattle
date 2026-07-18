import os
path = os.environ.get('CANDIDATE_CONFIG')
if not path:
    raise RuntimeError('CANDIDATE_CONFIG missing')
os.environ['BOT_TUNING_PATH'] = path
from my_bot import main
if __name__ == '__main__':
    main()
