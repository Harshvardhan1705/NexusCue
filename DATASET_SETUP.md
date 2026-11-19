# NexusCue Dataset Setup Guide

This guide explains how to download and place the datasets required by the NexusCue sign-generation pipeline.

> **Important:** The datasets are not stored in this GitHub repository. They must be downloaded separately and placed in the expected directories before running the complete sign-generation workflow.

## 1. Dataset sources

### A. ISL Alphabet Dataset

- **Source:** [RealSign — Indian Sign Language Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset)
- **Used for:** A–Z fingerspelling fallback when a required sign is unavailable.
- **Expected local folders:** `Letters`, `Training`, `Testing`, and `Validation`.

Download the dataset ZIP from the source repository. Extract it and place the four folders under:

```text
NexusCue/
└── Dataset/
    └── ISL_Alphabet/
        ├── Letters/
        ├── Training/
        ├── Testing/
        └── Validation/
```

If the downloaded ZIP creates an additional enclosing folder, move the four dataset folders so that they are directly inside `Dataset/ISL_Alphabet/`.

### B. ISL-CSLTR Corpus

- **Source:** [ISL-CSLTR Indian Sign Language Dataset — Kaggle](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset)
- **Used for:** Sentence-level frame sequences, word-level sign images, sentence/video metadata, and gloss-related corpus information.

Download the dataset from Kaggle and extract the relevant corpus contents under:

```text
NexusCue/
└── Dataset/
    └── ISL_CSLRT_Corpus/
        └── ISL_CSLRT_Corpus/
            ├── corpus_csv_files/
            ├── Frames_Sentence_Level/
            ├── Frames_Word_Level/
            ├── Videos_Sentence_Level/
            └── ISL_CSLRT.txt
```

The downloaded archive may have a different enclosing folder name or additional nesting. Arrange the extracted contents so that the folders and files listed above are directly inside the inner `ISL_CSLRT_Corpus` directory.

**Do not overwrite the Python implementation files already present in this directory** (for example, `create_sign_video.py`, `gloss_engine.py`, and `sentence_sequence_engine.py`). If the archive contains files with the same names, stop and compare them before replacing anything.

## 2. Expected project layout

After downloading and arranging both datasets, the relevant project structure should look like this:

```text
NexusCue/
├── Dataset/
│   ├── ISL_Alphabet/
│   │   ├── Letters/
│   │   ├── Training/
│   │   ├── Testing/
│   │   └── Validation/
│   │
│   └── ISL_CSLRT_Corpus/
│       └── ISL_CSLRT_Corpus/
│           ├── corpus_csv_files/
│           ├── Frames_Sentence_Level/
│           ├── Frames_Word_Level/
│           ├── Videos_Sentence_Level/
│           ├── mediapipe_models/
│           ├── face_models/
│           ├── create_sign_video.py
│           ├── gloss_engine.py
│           ├── sentence_sequence_engine.py
│           └── ISL_CSLRT.txt
│
├── docs/
├── README.md
├── DATASET_SETUP.md
├── TECHNICAL_DOCUMENTATION.md
├── config.py
├── main.py
└── model_loader.py
```

The model folders and Python files shown above are part of the implementation setup; they are not necessarily supplied by the dataset archives. Keep the model files and implementation files already provided with the NexusCue project.

## 3. Verify the dataset folders on Windows

Open PowerShell in the NexusCue project root and run:

```powershell
$required = @(
    ".\Dataset\ISL_Alphabet\Letters",
    ".\Dataset\ISL_Alphabet\Training",
    ".\Dataset\ISL_Alphabet\Testing",
    ".\Dataset\ISL_Alphabet\Validation",
    ".\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files",
    ".\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Sentence_Level",
    ".\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level",
    ".\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level"
)

foreach ($path in $required) {
    if (Test-Path $path) {
        Write-Host "[FOUND]  $path" -ForegroundColor Green
    }
    else {
        Write-Host "[MISSING] $path" -ForegroundColor Red
    }
}
```

All listed directories should report `[FOUND]`. Also check that the gloss CSV exists:

```powershell
Test-Path ".\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL Corpus sign glosses.csv"
```

This should return `True`.

## 4. Why the datasets are not committed to GitHub

The complete local datasets are large and contain many image and video files. The repository therefore excludes the alphabet dataset and the large sentence/word frame and video directories through `.gitignore`.

This keeps GitHub focused on the source code, documentation, metadata, and required implementation assets. Each developer must obtain the datasets from their respective sources and place them in the expected locations.

The `.gitignore` rules do not delete local dataset files. They only prevent the specified paths from being added to Git.

## 5. Licensing and attribution

Before downloading or reusing either dataset, read the licence and usage terms displayed on its original source page.

- **RealSign alphabet dataset:** The source repository identifies its dataset licence as **CC0 1.0**. Review the repository's licence and README for the exact terms and recommended citation.
- **ISL-CSLTR Kaggle copy:** Check the licence and usage terms shown on the Kaggle dataset page. Do not assume that a licence associated with another copy or publication automatically applies to this Kaggle upload.

Do not redistribute either dataset in another archive or mirror unless the applicable licence and source terms permit it. Retain required attribution and citations.

## 6. Troubleshooting

| Problem | What to check |
|---|---|
| Alphabet fallback cannot find a letter image | Confirm `Letters` and `Training` are in `Dataset/ISL_Alphabet/` and that the expected A–Z folders/files are present. |
| Sentence-level sequence is not found | Confirm `Frames_Sentence_Level/` is in the inner `ISL_CSLRT_Corpus` directory and that its sentence folders were extracted. |
| Word-level sign images are missing | Confirm `Frames_Word_Level/` is present and contains the expected word folders/images. |
| Gloss lookup cannot load the CSV | Confirm `corpus_csv_files/ISL Corpus sign glosses.csv` exists at the exact path shown above. |
| Dataset folders appear nested twice | Move the extracted contents into the expected inner directory; do not leave an unintended extra enclosing folder. |
| Extracting the corpus appears to replace Python files | Stop the extraction/overwrite operation and compare the conflicting files before proceeding. |

## 7. Important project limitation

NexusCue's current speech-to-sign pipeline uses dataset-backed sentence sequences and exact English-sentence-to-ISL-gloss mappings where available. If an exact supported mapping or sequence is unavailable, the fallback can spell the English sentence using alphabet images.

English fingerspelling is a spelling aid; it is **not** a verified ISL sentence translation. The current system should not be described as a general-purpose English-to-ISL translator.

---

**Dataset setup status:** Complete this guide's download, extraction, and verification steps before testing the full dataset-dependent sign-generation workflow.
