import json
from cashier_interfaces.msg import Item, Placement
################################ 물체 yaw만 회전함 #################################

json_data_1 = """
{
  "items": [
    {
      "item_id": "caramel_1",
      "name": "caramel",
      "pose": { "x": 422.0, "y": 55.3, "z": 233.0, "roll": 0.0, "pitch": 0.0, "yaw": -10.193912091088947 },
      "size": { "width": 82, "depth": 32, "height": 54 },
      "durability": 1
    },
    {
      "item_id": "eclipse_red",
      "name": "eclipse_red",
      "pose": { "x": 174.8, "y": 67.5, "z": 232.5, "roll": 0.0, "pitch": 0.0, "yaw": -29.2912109454698 },
      "size": { "width": 85, "depth": 31, "height": 48 },
      "durability": 5
    },
    {
      "item_id": "insect",
      "name": "insect",
      "pose": { "x": 286.2, "y": 90.9, "z": 242.5, "roll": 0.0, "pitch": 0.0, "yaw": -21.058553380712453 },
      "size": { "width": 91, "depth": 51, "height": 65 },
      "durability": 4
    },
    {
      "item_id": "eclipse_gre",
      "name": "eclipse_gre",
      "pose": { "x": 385.8, "y": 142.4, "z": 232.5, "roll": 0.0, "pitch": 0.0, "yaw": -30.249129003121794 },
      "size": { "width": 85, "depth": 31, "height": 48 },
      "durability": 5
    },
    {
      "item_id": "cream",
      "name": "cream",
      "pose": { "x": 178.5, "y": 190.2, "z": 234.0, "roll": 0.0, "pitch": 0.0, "yaw": 1.9372149982733282 },
      "size": { "width": 125, "depth": 34, "height": 44 },
      "durability": 3
    },
    {
      "item_id": "candy",
      "name": "candy",
      "pose": { "x": 410.0, "y": 259.9, "z": 243.0, "roll": 0.0, "pitch": 0.0, "yaw": -1.4570857404661126 },
      "size": { "width": 110, "depth": 52, "height": 85 },
      "durability": 2
    },
    {
      "item_id": "caramel_2",
      "name": "caramel",
      "pose": { "x": 185.7, "y": 272.1, "z": 233.0, "roll": 0.0, "pitch": 0.0, "yaw": -28.837750215872063 },
      "size": { "width": 82, "depth": 32, "height": 54 },
      "durability": 1
    }
  ],
  "placements": [
    { "object_index": 5, "pose": { "x": 257.5, "y": 75.0, "z": 25.0, "roll": 0.0, "pitch": 0.0, "yaw": 90.0 }},
    { "object_index": 2, "pose": { "x": 325.0, "y": 60.0, "z": 25.0, "roll": 0.0, "pitch": 0.0, "yaw": 90.0 }},
    { "object_index": 1, "pose": { "x": 377.5, "y": 60.0, "z": 17.5, "roll": 0.0, "pitch": 0.0, "yaw": 90.0 }},
    { "object_index": 3, "pose": { "x": 345.0, "y": 127.5, "z": 17.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }},
    { "object_index": 0, "pose": { "x": 345.0, "y": 127.5, "z": 42.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }},
    { "object_index": 6, "pose": { "x": 377.5, "y": 55.0, "z": 42.5, "roll": 0.0, "pitch": 0.0, "yaw": 90.0 }},
    { "object_index": 4, "pose": { "x": 282.5, "y": 37.5, "z": 55.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }}
  ]
}
"""
################################ 물체 yaw pitch 회전함 #################################
json_data_2 = """
{
  "items": [
    {
      "item_id": "caramel_1",
      "name": "caramel",
      "pose": { "x": 422.0, "y": 55.3, "z": 233.0, "roll": 0.0, "pitch": 0.0, "yaw": -10.193912091088947 },
      "size": { "width": 82, "depth": 32, "height": 54 },
      "durability": 1
    },
    {
      "item_id": "eclipse_red",
      "name": "eclipse_red",
      "pose": { "x": 174.8, "y": 67.5, "z": 232.5, "roll": 0.0, "pitch": 0.0, "yaw": -29.2912109454698 },
      "size": { "width": 85, "depth": 31, "height": 48 },
      "durability": 5
    },
    {
      "item_id": "insect",
      "name": "insect",
      "pose": { "x": 286.2, "y": 90.9, "z": 242.5, "roll": 0.0, "pitch": 0.0, "yaw": -21.058553380712453 },
      "size": { "width": 91, "depth": 51, "height": 65 },
      "durability": 4
    },
    {
      "item_id": "eclipse_gre",
      "name": "eclipse_gre",
      "pose": { "x": 385.8, "y": 142.4, "z": 232.5, "roll": 0.0, "pitch": 0.0, "yaw": -30.249129003121794 },
      "size": { "width": 85, "depth": 31, "height": 48 },
      "durability": 5
    },
    {
      "item_id": "cream",
      "name": "cream",
      "pose": { "x": 178.5, "y": 190.2, "z": 234.0, "roll": 0.0, "pitch": 0.0, "yaw": 1.9372149982733282 },
      "size": { "width": 125, "depth": 34, "height": 44 },
      "durability": 3
    },
    {
      "item_id": "candy",
      "name": "candy",
      "pose": { "x": 410.0, "y": 259.9, "z": 243.0, "roll": 0.0, "pitch": 0.0, "yaw": -1.4570857404661126 },
      "size": { "width": 110, "depth": 52, "height": 85 },
      "durability": 2
    },
    {
      "item_id": "caramel_2",
      "name": "caramel",
      "pose": { "x": 185.7, "y": 272.1, "z": 233.0, "roll": 0.0, "pitch": 0.0, "yaw": -28.837750215872063 },
      "size": { "width": 82, "depth": 32, "height": 54 },
      "durability": 1
    }
  ],
  "placements": [
    { "object_index": 2, "pose": { "x": 245.0, "y": 60.0, "z": 25.0, "roll": 0.0, "pitch": 0.0, "yaw": 90.0 }},
    { "object_index": 5, "pose": { "x": 330.0, "y": 57.5, "z": 25.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }},
    { "object_index": 1, "pose": { "x": 330.0, "y": 122.5, "z": 17.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }},
    { "object_index": 3, "pose": { "x": 330.0, "y": 122.5, "z": 42.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }},
    { "object_index": 4, "pose": { "x": 282.5, "y": 37.5, "z": 55.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }},
    { "object_index": 0, "pose": { "x": 252.5, "y": 127.5, "z": 40.0, "roll": 90.0, "pitch": 0.0, "yaw": 90.0 }},
    { "object_index": 6, "pose": { "x": 250.0, "y": 82.5, "z": 57.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }}
  ]
}
"""

# data = json.loads(json_data_1)
data = json.loads(json_data_2)

# ---------------------------
# Item 생성
# ---------------------------
items = []

for i in data["items"]:
    item = Item()

    item.item_id = i["item_id"]
    item.name = i["name"]

    item.width = i["size"]["width"]
    item.depth = i["size"]["depth"]
    item.height = i["size"]["height"]

    item.durability = i["durability"]

    item.x = i["pose"]["x"]
    item.y = i["pose"]["y"]
    item.z = i["pose"]["z"]

    item.roll = i["pose"]["roll"]
    item.pitch = i["pose"]["pitch"]
    item.yaw = i["pose"]["yaw"]

    items.append(item)


# ---------------------------
# Placement 생성
# ---------------------------
placements = []

for p in data["placements"]:
    place = Placement()

    place.object_index = p["object_index"]

    place.x = p["pose"]["x"]
    place.y = p["pose"]["y"]
    place.z = p["pose"]["z"]

    place.roll = p["pose"]["roll"]
    place.pitch = p["pose"]["pitch"]
    place.yaw = p["pose"]["yaw"]

    placements.append(place)


# 결과
pick_items = items
place_items = placements