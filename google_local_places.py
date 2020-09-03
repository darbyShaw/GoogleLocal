"""TODO(google_local_places): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
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
            "review_data":tfds.features.Text()
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
                "split_end":50
            }),    
      tfds.core.SplitGenerator(
            name="test", gen_kwargs={
                "file_path": dl_paths,
                "split_start":50,
                "split_end":100
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
          yield i, {"places_data":line}




