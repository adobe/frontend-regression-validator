from browsermobproxy import Server


def init(app):
    server = Server("utils/browsermob_proxy/bin/browsermob-proxy", options={'port': 8090})
    server.start()
    proxy = server.create_proxy()

    app.proxy = proxy
    app.server = server


def get_proxy(app):
    return app.proxy


def get_server(app):
    return app.server
