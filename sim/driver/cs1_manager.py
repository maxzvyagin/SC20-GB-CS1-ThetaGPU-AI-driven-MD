from fabric import Connection


def get_connection(host):
    return Connection(host)
