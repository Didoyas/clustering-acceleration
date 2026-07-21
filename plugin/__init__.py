def classFactory(iface):
    from .plugin import ClusteringPlugin
    return ClusteringPlugin(iface)
