# ASTRA VISION — Starter Image Dataset

Labelled starter images for **Challenge 02: AI-Based Defence Object
Recognition System**.

## Layout

```text
images/
├── aircraft/          30 images
├── helicopter/        30 images
├── drone/             30 images
├── military-vehicle/  30 images
└── naval/             30 images
labels.csv             file_name,category — one row per image
credits.csv            file_name,source_title,artist,license,commons_page
```

**150 images total**, all ≥300 px on the short side, `.jpg`/`.png`.
Five classes match the categories suggested in the challenge statement:
aircraft · helicopter · drone · military-vehicle · naval.

## Suggested first steps

1. Look at the data before training (class balance, backgrounds, dupes).
2. Suggest a train/validation split (e.g. 80/20) — `labels.csv` lets you
   split programmatically.
3. Baseline first: a pre-trained model + transfer learning beats training
   from scratch in three days.

## Known characteristics (read this)

- Images come from **Wikimedia Commons community uploads** — mixed quality
  by design: museum pieces next to modern airframes, sim renders next to
  photos, some near-duplicate angles from the same event.
- The set was de-junked (charts, posters, line drawings and images without
  the object removed), but **not exhaustively reviewed** — treat inspecting
  and cleaning labels as part of the work, and say in your README what you
  cleaned.
- 30 per class is a *starting point*, not a benchmark. Adding more images
  (with sources and licences noted) is a legitimate and expected move.

## Licence & attribution

Every image is a Wikimedia Commons file under a free licence (CC BY-SA,
CC0, public domain, etc.). Per-file artist, licence and source page are in
`credits.csv` — **keep this file** and name your sources in your README.
