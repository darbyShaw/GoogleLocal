"""google_local_reviews dataset."""

import tensorflow_datasets as tfds
from . import google_local_reviews


class GoogleLocalReviewsTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for google_local_reviews dataset."""
  # TODO(google_local_reviews):
  DATASET_CLASS = google_local_reviews.GoogleLocalReviews
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
