import os

# Global settings
# (same for all experiments)

MAX_SEQ_LENGTH = 300 #256  # max 512 (strongly affects GPU memory consumption)
HIDDEN_DIM = 768  # size of BERT hidden layer
MLP_DIM = 1024 #500  # size of multi layer perceptron (2 layers)
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 5

default_extra_cols = [
    
]
VALID_META = [
    "description",
    "host_since",
    "host_listings_count",
    "host_total_listings_count",
    "host_has_profile_pic",
    "host_identity_verified",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "amenities",
    "price",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "instant_bookable",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms" ]

# most_popular_label = 'Literatur & Unterhaltung'  # use this as default

most_popular_label = 'over'  # use this as default

if 'BERT_MODELS_DIR' not in os.environ:
    raise ValueError('You must define BERT_MODELS_DIR as environment variable!')

BERT_MODELS_DIR = os.environ['BERT_MODELS_DIR']

