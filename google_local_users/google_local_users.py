"""TODO(google_local_users): Add a description here."""


import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import ast
# TODO(google_local_users): BibTeX citation
_CITATION = """
Translation-based factorization machines for sequential recommendation
Rajiv Pasricha, Julian McAuley
RecSys, 2018

Translation-based recommendation
Ruining He, Wang-Cheng Kang, Julian McAuley
RecSys, 2017
"""

# TODO(google_local_users):
_DESCRIPTION = """
User information for google local dataset.
"""


class GoogleLocalUsers(tfds.core.GeneratorBasedBuilder):
  """TODO(google_local_users): User information for google local dataset."""

  # TODO(google_local_users): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(google_local_users): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            #"user_data": tfds.features.Text()
			"userName": tfds.features.Text(),
			"jobs": tfds.features.Text(),
			"currentPlace": tfds.features.Text(),
			"previousPlaces": tfds.features.Text(),
			"education": tfds.features.Text(),
			"gPlusUserId":tf.string
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=None,
        # Homepage of the dataset for documentation
        homepage='https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    dl_paths = dl_manager.download_and_extract("http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/users.clean.json.gz")
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    return [
      tfds.core.SplitGenerator(
            name="train", gen_kwargs={
                "file_path": dl_paths,
                "split_start":0,
                "split_end":100
            }),    
      tfds.core.SplitGenerator(
            name="test", gen_kwargs={
                "file_path": dl_paths,
                "split_start":100,
                "split_end":200
            }),    
    ]

  def _generate_examples(self, file_path,split_start,split_end):
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
          l=ast.literal_eval(line)
          yield i, {
            "userName": l["userName"],
            "jobs": str(l["jobs"]),
            "currentPlace": str(l["currentPlace"]),
            "previousPlaces": str(l["previousPlaces"]),
            "education": str(l["education"]),
            "gPlusUserId": l["gPlusUserId"]
		  }

