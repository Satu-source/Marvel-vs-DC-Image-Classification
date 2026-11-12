# Dataset provenance and availability

## Original study

The 2024 course report describes screenshots from *Doctor Strange in the Multiverse of Madness* (Marvel class) and *Aquaman* (DC class), sourced from a superhero movie screencap website. The first dataset included **440 images** (220 per class); an expanded version used **580 images** (290 per class).

Reported splits:

| Total images | Training | Validation | Test |
| ---: | ---: | ---: | ---: |
| 440 | 308 | 66 | 66 |
| 580 | 406 | 86 | 88 |

## Availability of the historical images

**No film screenshots are included in this public repository.** The image dataset used in the 2024 course experiment was a copyrighted third-party collection, and the author's original private archive is stored separately. That private archive may not include every image from the original training experiments.

The reported historical results cannot be reproduced solely from these public files. To run the demonstration code, supply your own suitable, legally usable images and create the required train/validation/test splits.

## Licensing

Movie screenshots may be copyright protected. Their appearance in an educational project does not grant a blanket right to redistribute them in a public GitHub repository. This project therefore provides a reference code path and documentation rather than publishing a complete public film-image dataset. Use images for which you have the necessary rights when reproducing the experiment.

## Expected local structure

```text
data/
    train/DC/       train/Marvel/
    valid/DC/       valid/Marvel/
    test/DC/        test/Marvel/
```

The `data/` directory is ignored by default in this public repository. See [README](../README.md) for the example command.
