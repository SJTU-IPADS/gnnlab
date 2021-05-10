from .papers100M import Papers100M
from .comfriendster import ComFriendster


def dataset(name, path):
    if name == 'Papers100M':
        return Papers100M(path)
    elif name == 'Com-Friendster':
        return ComFriendster(path)
    else:
        print(f"Dataset {name} not exist")
        assert(False)


__all__ = ['dataset', 'Papers100M', 'ComFriendster']
