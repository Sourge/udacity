# import gmaps
#
# def draw_heatmap(data):
#     gmaps.configure(api_key="AIzaSyDPWAl8lcrK9q-tOkrl64sGkxDnbWz47Ko")
#
#     locations = data[["lat", "long"]]
#     prices = data["price"]
#
#     heatmap_layer = gmaps.heatmap_layer(locations, weights=prices)
#     heatmap_layer.max_intensity = 7200000
#     heatmap_layer.point_radius = 4
#
#     fig = gmaps.figure()
#     fig.add_layer(heatmap_layer)