"""TODO(google_local_places): Add a description here."""


import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import ast
import requests
# TODO(google_local_places): BibTeX citation
_CITATION = """
Translation-based factorization machines for sequential recommendation
Rajiv Pasricha, Julian McAuley
RecSys, 2018

Translation-based recommendation
Ruining He, Wang-Cheng Kang, Julian McAuley
RecSys, 2017

"""

# TODO(google_local_places):
_DESCRIPTION = """
Google local places dataset
"""


class GoogleLocalPlaces(tfds.core.GeneratorBasedBuilder):
  """TODO(google_local_places): Short description of my dataset."""

  # TODO(google_local_places): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(google_local_places): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
			"name" : tf.string,
			"price":tf.string,
			"address": tf.string,
			"hours": tf.string,
			"phone": tf.string,
			"closed": tf.string,
			"gPlusPlaceId": tf.string,
			"gps":tf.string

        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        
        # Homepage of the dataset for documentation
        homepage='https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    dl_paths = dl_manager.download_and_extract("http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/places.clean.json.gz")
    return [
      tfds.core.SplitGenerator(
            name="train", gen_kwargs={
                "file_path": dl_paths,
                "split_start":0,
                "split_end":10
            }),    
      tfds.core.SplitGenerator(
            name="test", gen_kwargs={
                "file_path": dl_paths,
                "split_start":0,
                "split_end":0
            }),    
    ]


  def _generate_examples(self,file_path,split_start,split_end):
    """Yields examples."""
    with tf.io.gfile.GFile(file_path) as f:
      i=-1
      while True:
        line = f.readline()
        if line == '':
          break
        if i==split_end:
          break
        i=i+1
        if i>=split_start:
          l = ast.literal_eval(line)
          if l!=None and 'gps' in l and l['gps']!=None:
            parameters = {"format":'jsonv2', "lat" : l['gps'][0], "lon": l['gps'][1]}
            address = requests.get("https://nominatim.openstreetmap.org/reverse", params=parameters).json()
            if 'address' in address:
              l['address'] = address['address']
              def replace_none_with_empty_str(some_dict):
                return { k: ('' if v is None else v) for k, v in some_dict.items() }
              l = replace_none_with_empty_str(l)
              yield i, {#"places_data":line
			    "name" : l["name"],
			    "price": l["price"],
			    "address": str(l['address']),
			    "hours": str(l['hours']),
			    "phone": str(l['phone']),
			    "closed": str(l['closed']),
			    "gPlusPlaceId": l['gPlusPlaceId'],
			    "gps":str(l['gps'])
			    }




