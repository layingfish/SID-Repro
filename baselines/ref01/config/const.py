
import os
from pathlib import Path


def _setrec_path(*parts):
    root = os.environ.get("SEATER_SETREC_DATA_ROOT", "data/seater_setrec")
    return str(Path(root).joinpath(*parts))


class Books_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = '../open-source-data/Books'

        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'
        self.itemID_2_attr = 'vocab/item_2_attr_mapping.npy'


        self.tree_data_par_path = '../open-source-data/Books/tree_data_SASREC'

        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Books_SASREC_item_emb.npy'


        self.users_ID_num = 603671
        self.item_ID_num = 367982 + 1
        self.item_cate_num = 1600 + 1


        self.reco_his_max_length = 20

class Yelp_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = '../open-source-data/Yelp'

        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'


        self.tree_data_par_path = '../open-source-data/Yelp/tree_data_SASREC'

        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Yelp_SASREC_item_emb.npy'


        self.users_ID_num = 31668
        self.item_ID_num = 38048 + 1


        self.reco_his_max_length = 20

class MIND_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = '../open-source-data/MIND'

        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'


        self.tree_data_par_path = '../open-source-data/MIND/tree_data_SASREC'

        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/MIND_SASREC_item_emb.npy'


        self.users_ID_num = 50000
        self.item_ID_num = 39865 + 1


        self.reco_his_max_length = 20


class SETRec_beauty_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = _setrec_path("beauty")
        self.training_file = "dataset/training.tsv"
        self.validation_file = "dataset/validation.tsv"
        self.test_file = "dataset/test.tsv"
        self.itemID_2_attr = "vocab/item_2_attr_mapping.npy"


        self.tree_data_par_path = _setrec_path("beauty", "tree_data_SASREC")
        self.tree_based_itemID_2_indexID = "itemID_2_tree_indexID.npy"
        self.tree_based_prefix_tree = "tree_node_allowed_next_tokens.npy"

        self.two_tower_item_emb = "vocab/beauty_SASREC_item_emb.npy"


        self.users_ID_num = 18136
        self.item_ID_num = 12024 + 1


        self.reco_his_max_length = 50


class SETRec_toys_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = _setrec_path("toys")
        self.training_file = "dataset/training.tsv"
        self.validation_file = "dataset/validation.tsv"
        self.test_file = "dataset/test.tsv"
        self.itemID_2_attr = "vocab/item_2_attr_mapping.npy"


        self.tree_data_par_path = _setrec_path("toys", "tree_data_SASREC")
        self.tree_based_itemID_2_indexID = "itemID_2_tree_indexID.npy"
        self.tree_based_prefix_tree = "tree_node_allowed_next_tokens.npy"

        self.two_tower_item_emb = "vocab/toys_SASREC_item_emb.npy"


        self.users_ID_num = 15635
        self.item_ID_num = 11908 + 1


        self.reco_his_max_length = 50


class SETRec_sports_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = _setrec_path("sports")
        self.training_file = "dataset/training.tsv"
        self.validation_file = "dataset/validation.tsv"
        self.test_file = "dataset/test.tsv"
        self.itemID_2_attr = "vocab/item_2_attr_mapping.npy"


        self.tree_data_par_path = _setrec_path("sports", "tree_data_SASREC")
        self.tree_based_itemID_2_indexID = "itemID_2_tree_indexID.npy"
        self.tree_based_prefix_tree = "tree_node_allowed_next_tokens.npy"

        self.two_tower_item_emb = "vocab/sports_SASREC_item_emb.npy"


        self.users_ID_num = 27591
        self.item_ID_num = 18340 + 1


        self.reco_his_max_length = 50


class SETRec_steam_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = _setrec_path("steam")
        self.training_file = "dataset/training.tsv"
        self.validation_file = "dataset/validation.tsv"
        self.test_file = "dataset/test.tsv"
        self.itemID_2_attr = "vocab/item_2_attr_mapping.npy"


        self.tree_data_par_path = _setrec_path("steam", "tree_data_SASREC")
        self.tree_based_itemID_2_indexID = "itemID_2_tree_indexID.npy"
        self.tree_based_prefix_tree = "tree_node_allowed_next_tokens.npy"

        self.two_tower_item_emb = "vocab/steam_SASREC_item_emb.npy"


        self.users_ID_num = 32919
        self.item_ID_num = 14330 + 1


        self.reco_his_max_length = 50


class SETRec_microlens_50k_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = _setrec_path("microlens_50k")
        self.training_file = "dataset/training.tsv"
        self.validation_file = "dataset/validation.tsv"
        self.test_file = "dataset/test.tsv"
        self.itemID_2_attr = "vocab/item_2_attr_mapping.npy"


        self.tree_data_par_path = _setrec_path("microlens_50k", "tree_data_SASREC")
        self.tree_based_itemID_2_indexID = "itemID_2_tree_indexID.npy"
        self.tree_based_prefix_tree = "tree_node_allowed_next_tokens.npy"

        self.two_tower_item_emb = "vocab/microlens_50k_SASREC_item_emb.npy"


        self.users_ID_num = 42666
        self.item_ID_num = 14079 + 1


        self.reco_his_max_length = 50


class SETRec_amazon23_vg_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = _setrec_path("amazon23_vg")
        self.training_file = "dataset/training.tsv"
        self.validation_file = "dataset/validation.tsv"
        self.test_file = "dataset/test.tsv"
        self.itemID_2_attr = "vocab/item_2_attr_mapping.npy"


        self.tree_data_par_path = _setrec_path("amazon23_vg", "tree_data_SASREC")
        self.tree_based_itemID_2_indexID = "itemID_2_tree_indexID.npy"
        self.tree_based_prefix_tree = "tree_node_allowed_next_tokens.npy"

        self.two_tower_item_emb = "vocab/amazon23_vg_SASREC_item_emb.npy"


        self.users_ID_num = 74333
        self.item_ID_num = 25062 + 1


        self.reco_his_max_length = 50


class SETRec_yelp_Config(object):
    def __init__(self) -> None:


        self.datafile_par_path = _setrec_path("yelp")
        self.training_file = "dataset/training.tsv"
        self.validation_file = "dataset/validation.tsv"
        self.test_file = "dataset/test.tsv"
        self.itemID_2_attr = "vocab/item_2_attr_mapping.npy"


        self.tree_data_par_path = _setrec_path("yelp", "tree_data_SASREC")
        self.tree_based_itemID_2_indexID = "itemID_2_tree_indexID.npy"
        self.tree_based_prefix_tree = "tree_node_allowed_next_tokens.npy"

        self.two_tower_item_emb = "vocab/yelp_SASREC_item_emb.npy"


        self.users_ID_num = 221039
        self.item_ID_num = 109326 + 1


        self.reco_his_max_length = 50
