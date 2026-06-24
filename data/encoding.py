import numpy as np

def tile_to_index(tile: str) -> int:
    suit = tile[0]
    num_char = tile[1]
    num = int(num_char)

    if suit == "m":
        base = 0
    elif suit == "p":
        base = 10
    elif suit == "s":
        base = 20
    elif suit == "z":
        base = 29
        if num == 0:
            raise ValueError("z0 は存在しない牌です")
    else:
        raise ValueError(f"未知のスートです: {suit}")

    idx = base + num

    if not (0 <= idx < 37):
        raise ValueError(f"インデックスが範囲外です: tile={tile}, idx={idx}")

    return idx


def create_tile_images(all_wdt_data):
    img_set = []
    for wdt in all_wdt_data:
        tile_image = [[0]*37 for _ in range(4)]
        for tile in wdt:
            idx = tile_to_index(tile)

            current_count = 0
            for row in range(4):
                if tile_image[row][idx] == 1:
                    current_count += 1

            if current_count < 4:
                place_row = current_count
                tile_image[place_row][idx] = 1

        img = np.array(tile_image, dtype=int)
        img_set.append(img)
    return img_set


def create_yaku_dict(all_roles):
    unique_yaku_list = []
    for round_roles in all_roles:
        for role in round_roles:
            if role not in unique_yaku_list:
                unique_yaku_list.append(role)

    yaku_dict = {i: yaku for i, yaku in enumerate(unique_yaku_list)}
    return yaku_dict


def create_multi_hot_labels(all_final_roles, yaku_dict):
    num_yaku = len(yaku_dict)

    multi_hot_labels = []
    for roles_in_round in all_final_roles:
        label_vector = np.zeros(num_yaku, dtype=int)
        for role in roles_in_round:
            for idx, yaku_name in yaku_dict.items():
                if yaku_name == role:
                    label_vector[idx] = 1
                    break
        multi_hot_labels.append(label_vector)

    return multi_hot_labels