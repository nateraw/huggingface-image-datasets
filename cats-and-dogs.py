import pickle
from pathlib import Path
from typing import List

import datasets

logger = datasets.logging.get_logger(__name__)


_HOMEPAGE = "https://www.microsoft.com/en-us/download/details.aspx?id=54765"
_URL = "https://huggingface.co/datasets/nateraw/cats-and-dogs/resolve/main/"
_URLS = {
    "train": _URL + "train.pt",
}
_DESCRIPTION = "A large set of images of cats and dogs. There are 1738 corrupted images that are dropped."
_CITATION = """\
@Inproceedings (Conference){asirra-a-captcha-that-exploits-interest-aligned-manual-image-categorization,
    author = {Elson, Jeremy and Douceur, John (JD) and Howell, Jon and Saul, Jared},
    title = {Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization},
    booktitle = {Proceedings of 14th ACM Conference on Computer and Communications Security (CCS)},
    year = {2007},
    month = {October},
    publisher = {Association for Computing Machinery, Inc.},
    url = {https://www.microsoft.com/en-us/research/publication/asirra-a-captcha-that-exploits-interest-aligned-manual-image-categorization/},
    edition = {Proceedings of 14th ACM Conference on Computer and Communications Security (CCS)},
}
"""


class CatsAndDogs(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "img_bytes": datasets.Value("binary"),
                    "labels": datasets.features.ClassLabel(names=["cat", "dog"]),
                }
            ),
            supervised_keys=("img_bytes", "labels"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with Path(filepath).open("rb") as f:
            examples = pickle.load(f)

        for i, ex in enumerate(examples):
            yield str(i), ex
