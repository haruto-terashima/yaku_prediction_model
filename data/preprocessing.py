import zipfile
import json

def clean_pai_id(pai_id:str) -> str:
    if pai_id.endswith('_') or pai_id.endswith('*'):
        return pai_id[:-1]
    return pai_id

def extract_single_paifu_data(paifu_json_data):
    i_local = []
    u_local = []
    # 全局をループ
    for round_log in paifu_json_data['log']:
        winner = None
        final_roles = []

        # 上がった人とhuleイベントを探す（後ろから検索）
        for event in reversed(round_log):
            if 'hule' in event:
                winner = event['hule']['l']
                if 'hupai' in event['hule'] and event['hule']['hupai'] is not None:
                    final_roles = [r['name'] for r in event['hule']['hupai']]
                else:
                    final_roles = []
                break

        if winner is None:
            continue

        winner_dapai_tiles = []

        for event in round_log:
            if 'dapai' in event and event['dapai']['l'] == winner:
                winner_dapai_tiles.append(clean_pai_id(event['dapai']['p']))

        winner_dapai_tiles.sort()

        i_local.append(winner_dapai_tiles)
        u_local.append(final_roles)

    return i_local, u_local